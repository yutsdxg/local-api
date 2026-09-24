from __future__ import annotations

import array
import ctypes as c
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import wave

from scripts import prepare_asr_evaluation as preparation


def write_wav(path: Path, samples: array.array, *, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as target:
        target.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        target.writeframes(samples.tobytes())


class TestExtractSegments(unittest.TestCase):
    def test_overlap_gap_and_mapping_preserve_actual_source_positions(self) -> None:
        samples = array.array("h", range(16000))
        output, segments, gaps = preparation.extract_segments(samples, [(10, 20), (50, 60)])
        self.assertEqual(samples[1600:4800], output[:3200])
        self.assertEqual(array.array("h", [0]) * 1600, output[3200:4800])
        self.assertEqual(samples[8000:9600], output[4800:])
        self.assertEqual(6400, len(output))
        self.assertEqual(4800, segments[0]["source"]["end_sample"])
        self.assertEqual(3200, segments[0]["source_before_overlap"]["end_sample"])
        self.assertEqual(1600, segments[0]["added_end_overlap_samples"])
        self.assertEqual(0, segments[1]["added_end_overlap_samples"])
        self.assertEqual(4800, segments[1]["output"]["start_sample"])
        self.assertEqual({"start_sample": 3200, "end_sample": 4800,
                          "start_seconds": 0.2, "end_seconds": 0.3}, gaps[0]["output"])
        self.assertIsNone(gaps[0]["source"])

    def test_tail_clamping_keeps_final_sample_and_handles_short_input(self) -> None:
        for length in (1, 319, 512, 1601):
            with self.subTest(length=length):
                samples = array.array("h", range(length))
                output, segments, gaps = preparation.extract_segments(samples, [(0, 200)])
                self.assertEqual(samples, output)
                self.assertEqual(length, segments[0]["source"]["end_sample"])
                self.assertEqual([], gaps)

    def test_empty_input_and_no_speech_return_no_samples_or_synthetic_gap(self) -> None:
        for samples, intervals in ((array.array("h"), []), (array.array("h"), [(0, 10)]),
                                   (array.array("h", [0]) * 320, [])):
            with self.subTest(length=len(samples), intervals=intervals):
                self.assertEqual((array.array("h"), [], []),
                                 preparation.extract_segments(samples, intervals))

    def test_rounded_zero_length_or_beyond_eof_intervals_are_omitted(self) -> None:
        samples = array.array("h", range(320))
        output, segments, gaps = preparation.extract_segments(samples, [(0, 0), (0, 2), (3, 4)])
        self.assertEqual(samples, output)
        self.assertEqual([1], [segment["segment_index"] for segment in segments])
        self.assertEqual([], gaps)

    def test_invalid_intervals_fail_before_extraction(self) -> None:
        for intervals in ([(-1, 2)], [(3, 2)], [(0, float("nan"))],
                          [(0, float("inf"))], [(0.5, 2)], [(3, 4), (1, 2)]):
            with self.subTest(intervals=intervals):
                with self.assertRaises(ValueError):
                    preparation.extract_segments(array.array("h", [0]) * 1000, intervals)


class TestWhisperVadBinding(unittest.TestCase):
    def library(self):
        library = mock.Mock()
        library.whisper_version.return_value = b"1.8.2"
        library.whisper_vad_default_context_params.return_value = preparation.ContextParams(3, True, 2)
        library.whisper_vad_default_params.return_value = preparation.VadParams(0.8, 250, 100, 123, 30, 0.2)
        library.whisper_vad_init_from_file_with_params.return_value = 100
        library.whisper_vad_segments_from_samples.return_value = 200
        library.whisper_vad_segments_n_segments.return_value = 1
        library.whisper_vad_segments_get_segment_t0.return_value = 0.0
        library.whisper_vad_segments_get_segment_t1.return_value = 1.0
        return library

    def test_version_gate_precedes_struct_abi_calls(self) -> None:
        library = self.library()
        library.whisper_version.return_value = b"1.9.0"
        with mock.patch.object(preparation.c, "CDLL", return_value=library):
            with self.assertRaisesRegex(ValueError, "Unsupported whisper.cpp ABI"):
                preparation.WhisperVad(Path("library"), Path("model"))
        library.whisper_vad_default_context_params.assert_not_called()
        library.whisper_vad_default_params.assert_not_called()

    def test_default_parameters_cpu_normalization_and_resource_cleanup(self) -> None:
        library = self.library()

        def infer(context, params, buffer, count):
            self.assertEqual(100, context)
            self.assertEqual(3, count)
            self.assertEqual([-1.0, 0.0, 32767 / 32768], list(buffer))
            self.assertEqual(100, params.min_speech_duration_ms)
            self.assertEqual(500, params.min_silence_duration_ms)
            self.assertEqual(200, params.speech_pad_ms)
            self.assertEqual(0.5, params.threshold)
            self.assertEqual(123, params.max_speech_duration_s)
            self.assertAlmostEqual(0.1, params.samples_overlap)
            return 200

        library.whisper_vad_segments_from_samples.side_effect = infer
        with mock.patch.object(preparation.c, "CDLL", return_value=library):
            vad = preparation.WhisperVad(Path("library"), Path("model"))
            self.assertEqual([(0.0, 1.0)], vad.detect(array.array("h", [-32768, 0, 32767])))
        params = library.whisper_vad_init_from_file_with_params.call_args.args[1]
        self.assertFalse(params.use_gpu)
        self.assertEqual(3, params.n_threads)
        library.whisper_vad_free_segments.assert_called_once_with(200)
        library.whisper_vad_free.assert_called_once_with(100)
        self.assertEqual(c.c_char_p, library.whisper_version.restype)

    def test_failed_detection_frees_context_and_never_frees_null_segments(self) -> None:
        library = self.library()
        library.whisper_vad_segments_from_samples.return_value = None
        with mock.patch.object(preparation.c, "CDLL", return_value=library):
            vad = preparation.WhisperVad(Path("library"), Path("model"))
            with self.assertRaisesRegex(RuntimeError, "failed to return"):
                vad.detect(array.array("h", [0]))
        library.whisper_vad_free.assert_called_once_with(100)
        library.whisper_vad_free_segments.assert_not_called()

    def test_empty_input_skips_native_inference(self) -> None:
        library = self.library()
        with mock.patch.object(preparation.c, "CDLL", return_value=library):
            vad = preparation.WhisperVad(Path("library"), Path("model"))
            self.assertEqual([], vad.detect(array.array("h")))
        library.whisper_vad_init_from_file_with_params.assert_not_called()
        library.whisper_vad_segments_from_samples.assert_not_called()


class TestPrepareEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.wav"
        self.output = self.root / "new-output"
        self.library = self.root / "library"
        self.model = self.root / "model"
        self.library.write_bytes(b"library-content")
        self.model.write_bytes(b"model-content")
        self.samples = array.array("h", range(1600))
        write_wav(self.source, self.samples)

    def run_preparation(self, intervals):
        with mock.patch.object(preparation, "WhisperVad") as factory:
            factory.return_value.version = "1.8.2"
            factory.return_value.detect.return_value = intervals
            factory.return_value.settings.return_value = {"cpu": True}
            manifest = preparation.prepare_evaluation(self.source, self.output, self.library, self.model)
        return manifest, factory

    def test_manifest_hashes_match_all_inputs_and_output(self) -> None:
        manifest, _ = self.run_preparation([(0, 10)])
        self.assertEqual("ready", manifest["status"])
        self.assertEqual(self.samples, preparation.read_pcm16(self.output / "prepared.wav"))
        saved = json.loads((self.output / "manifest.json").read_text())
        for key in ("source", "output", "whisper_library", "vad_model"):
            path = Path(saved[key]["path"])
            self.assertTrue(path.is_absolute())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), saved[key]["sha256"])
        self.assertEqual(1600, saved["output"]["samples"])
        self.assertTrue(any("one-sample overrun" in note for note in saved["implementation_notes"]))

    def test_existing_output_directory_is_refused_even_if_empty(self) -> None:
        self.output.mkdir()
        with mock.patch.object(preparation, "WhisperVad") as factory:
            with self.assertRaises(FileExistsError):
                preparation.prepare_evaluation(self.source, self.output, self.library, self.model)
        self.assertEqual([], list(self.output.iterdir()))
        factory.assert_not_called()

    def test_empty_and_silent_input_produce_valid_empty_wav_with_explicit_status(self) -> None:
        for length, status in ((0, "empty_input"), (512, "no_speech")):
            with self.subTest(length=length):
                self.output = self.root / f"output-{length}"
                write_wav(self.source, array.array("h", [0]) * length)
                manifest, _ = self.run_preparation([])
                self.assertEqual(status, manifest["status"])
                self.assertEqual(array.array("h"), preparation.read_pcm16(self.output / "prepared.wav"))
                self.assertEqual([], manifest["segments"])
                self.assertEqual(0, manifest["output"]["samples"])

    def test_wrong_sample_rate_is_rejected_before_vad_or_output_creation(self) -> None:
        write_wav(self.source, self.samples, sample_rate=48000)
        with mock.patch.object(preparation, "WhisperVad") as factory:
            with self.assertRaisesRegex(ValueError, "16000 Hz"):
                preparation.prepare_evaluation(self.source, self.output, self.library, self.model)
        factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_truncated_wav_is_rejected(self) -> None:
        raw = self.source.read_bytes()
        self.source.write_bytes(raw[:-2])
        with self.assertRaisesRegex(ValueError, "Truncated WAV"):
            preparation.read_pcm16(self.source)


if __name__ == "__main__":
    unittest.main()
