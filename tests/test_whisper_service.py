from __future__ import annotations

import asyncio
import io
import shutil
import struct
import subprocess
import threading
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi import UploadFile

from app import main
from app.config import Settings
from app.services import whisper as whisper_service


class TestWhisperService(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.archive = self.base / "whisper"
        self.archive.mkdir()
        self.vad_model = self.base / "vad.bin"
        self.vad_model.write_bytes(b"test model")
        self.settings = replace(
            _settings(self.archive, whisper_args=("-ng", "-nt", "-np")),
            whisper_vad_model_path=self.vad_model,
        )
        self.calls: list[list[str]] = []
        self.converted_frames = 16000
        self.transcription = "transcribed"

    def fake_run(self, cmd: list[str], check: bool) -> None:
        self.assertTrue(check)
        self.calls.append(cmd)
        if "-of" in cmd:
            output_prefix = Path(cmd[cmd.index("-of") + 1])
            output_prefix.with_suffix(".txt").write_text(self.transcription, encoding="utf-8")
        elif cmd[0] == "ffmpeg":
            _write_wav(Path(cmd[-1]), self.converted_frames, int(cmd[cmd.index("-ar") + 1]))
        else:
            source = Path(cmd[-1])
            output_dir = Path(cmd[cmd.index("-o") + 1])
            shutil.copyfile(source, output_dir / source.name)

    def transcribe(self, settings: Settings, filename: str = "audio.m4a") -> str:
        upload = UploadFile(filename=filename, file=io.BytesIO(b"audio"))
        with mock.patch.object(whisper_service.subprocess, "run", side_effect=self.fake_run):
            return asyncio.run(whisper_service.transcribe_upload(upload, settings))

    def modern_settings(self, profile: str = "vad", **overrides: object) -> Settings:
        return replace(self.settings, whisper_preprocessing=profile,
                       whisper_vad_model_path=self.vad_model, **overrides)

    def deepfilter_settings(self, **overrides: object) -> Settings:
        binary = self.base / "deep-filter"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)
        model = self.base / "dfn.tar.gz"
        model.write_bytes(b"test model")
        return self.modern_settings("deepfilter", whisper_deepfilter_bin=str(binary),
                                    whisper_deepfilter_model_path=model, **overrides)

    def test_transcribe_upload_passes_configured_whisper_args(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            whisper_tmp_dir = tmp_path / "whisper"
            whisper_tmp_dir.mkdir()
            settings = replace(
                _settings(whisper_tmp_dir, whisper_args=("-ng", "-nt", "-np")),
                whisper_vad_model_path=self.vad_model,
            )
            upload = UploadFile(filename="audio.m4a", file=io.BytesIO(b"audio"))
            calls: list[list[str]] = []

            def fake_run(cmd: list[str], check: bool) -> None:
                self.assertTrue(check)
                calls.append(cmd)
                if "-of" in cmd:
                    output_prefix = Path(cmd[cmd.index("-of") + 1])
                    output_prefix.with_suffix(".txt").write_text("transcribed", encoding="utf-8")
                else:
                    _write_wav(Path(cmd[-1]), 16000)

            with mock.patch.object(whisper_service.subprocess, "run", side_effect=fake_run):
                text = asyncio.run(
                    whisper_service.transcribe_upload(upload, settings, language="ja")
                )

            self.assertEqual("transcribed", text)
            self.assertEqual(2, len(calls))
            whisper_cmd = calls[1]
            self.assertIn("-ng", whisper_cmd)
            self.assertIn("-nt", whisper_cmd)
            self.assertIn("-np", whisper_cmd)
            self.assertEqual(["-l", "ja", "-ng", "-nt", "-np"], whisper_cmd[-5:])

    def test_legacy_keeps_filters_and_does_not_require_optional_models(self) -> None:
        settings = replace(self.settings, whisper_preprocessing="legacy", whisper_normalize=False,
                           whisper_vad_model_path=self.base / "missing-vad.bin",
                           whisper_deepfilter_bin="missing-deep-filter")
        self.assertEqual("transcribed", self.transcribe(settings))
        conversion, whisper = self.calls
        self.assertEqual(
            "highpass=f=120, lowpass=f=8000, dynaudnorm=f=200:g=7, "
            "silenceremove=start_periods=1:start_duration=0.8:"
            "start_threshold=-50dB:stop_periods=-1:stop_duration=0.8:stop_threshold=-50dB",
            conversion[conversion.index("-af") + 1],
        )
        self.assertNotIn("--vad", whisper)

    def test_default_vad_preserves_audio_without_silence_or_band_filters(self) -> None:
        settings = replace(self.settings, whisper_deepfilter_bin="missing-deep-filter")
        self.assertEqual("transcribed", self.transcribe(settings))
        conversion, whisper = self.calls
        self.assertNotIn("-af", conversion)
        self.assertEqual("16000", conversion[conversion.index("-ar") + 1])
        self.assertIn("--vad", whisper)
        for option, value in (
            ("--vad-model", str(self.vad_model)),
            ("--vad-threshold", "0.5"),
            ("--vad-min-speech-duration-ms", "100"),
            ("--vad-min-silence-duration-ms", "500"),
            ("--vad-speech-pad-ms", "200"),
        ):
            self.assertEqual(value, whisper[whisper.index(option) + 1])
        self.assertEqual(["-l", "ja", "-ng", "-nt", "-np"], whisper[-5:])

    def test_vad_normalization_is_optional_and_has_no_silence_filter(self) -> None:
        self.transcribe(self.modern_settings(whisper_normalize=True))
        conversion = self.calls[0]
        self.assertEqual("dynaudnorm=f=200:g=7", conversion[conversion.index("-af") + 1])

    def test_missing_selected_model_fails_before_running_commands(self) -> None:
        settings = replace(self.modern_settings(), whisper_vad_model_path=self.base / "missing.bin")
        with self.assertRaisesRegex(whisper_service.WhisperError, "VAD model not found"):
            self.transcribe(settings)
        self.assertEqual([], self.calls)

    def test_missing_selected_deepfilter_executable_is_reported(self) -> None:
        settings = self.modern_settings("deepfilter", whisper_deepfilter_bin=str(self.base / "missing"))
        with self.assertRaisesRegex(whisper_service.WhisperError, "DeepFilterNet executable not found"):
            self.transcribe(settings)
        self.assertEqual([], self.calls)

    def test_invalid_profiles_and_vad_parameters_are_rejected(self) -> None:
        cases = [
            replace(self.settings, whisper_preprocessing="invalid"),
            self.modern_settings(whisper_vad_threshold=1.5),
            self.modern_settings(whisper_vad_threshold=float("nan")),
            self.modern_settings(whisper_vad_speech_pad_ms=-1),
        ]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(whisper_service.WhisperError):
                self.transcribe(settings)
        self.assertEqual([], self.calls)

    def test_empty_converted_wav_skips_whisper_and_preserves_empty_archive(self) -> None:
        self.converted_frames = 0
        self.assertEqual("", self.transcribe(self.modern_settings()))
        self.assertEqual(1, len(self.calls))
        self.assertEqual(1, len(list(self.archive.glob("whisper_input_*.wav"))))
        outputs = list(self.archive.glob("whisper_output_*.txt"))
        self.assertEqual(1, len(outputs))
        self.assertEqual("", outputs[0].read_text(encoding="utf-8"))

    def test_no_speech_result_is_empty_text_success(self) -> None:
        self.transcription = ""
        self.assertEqual("", self.transcribe(self.modern_settings()))

    def test_upload_filename_cannot_escape_temporary_directory(self) -> None:
        outside = self.base / "should-not-be-written"
        self.transcribe(self.settings, filename=str(outside))
        actual_input = Path(self.calls[0][self.calls[0].index("-i") + 1])
        self.assertEqual("upload.audio", actual_input.name)
        self.assertFalse(outside.exists())
        self.assertFalse(actual_input.parent.exists())

    def test_deepfilter_runs_at_48k_then_trims_padding_before_normalizing(self) -> None:
        self.transcribe(self.deepfilter_settings(whisper_normalize=True))
        self.assertEqual(4, len(self.calls))
        decode, denoise, convert, whisper = self.calls
        self.assertEqual("48000", decode[decode.index("-ar") + 1])
        self.assertNotIn("-af", decode)
        self.assertEqual("12.0", denoise[denoise.index("-a") + 1])
        self.assertIn("-D", denoise)
        self.assertNotIn("--pf", denoise)
        self.assertEqual("atrim=end_sample=16000,dynaudnorm=f=200:g=7",
                         convert[convert.index("-af") + 1])
        self.assertEqual("16000", convert[convert.index("-ar") + 1])
        self.assertIn("--vad", whisper)
        self.assertFalse(Path(denoise[-1]).parent.exists())
        self.assertEqual(2, len(list(self.archive.iterdir())))

    def test_deepfilter_reflection_padding_keeps_original_samples_and_flushes_short_tail(self) -> None:
        for frames in (1, 47, 2000):
            with self.subTest(frames=frames):
                source = self.base / "source.wav"
                padded = self.base / "padded.wav"
                original = struct.pack(f"<{frames}h", *range(1, frames + 1))
                with wave.open(str(source), "wb") as output:
                    output.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                    output.writeframes(original)
                whisper_service._pad_deepfilter_input(source, padded)
                with wave.open(str(padded), "rb") as result:
                    self.assertEqual(frames + 1920, result.getnframes())
                    self.assertEqual(original, result.readframes(frames))
                    padding = result.readframes(1920)
                    self.assertEqual(struct.pack("<h", frames), padding[:2])
                    self.assertNotEqual(b"\0" * len(padding), padding)

    def test_deepfilter_failure_cleans_intermediate_files_and_raises_whisper_error(self) -> None:
        settings = self.deepfilter_settings()
        captured_work_dirs = []

        def fail_denoise(cmd: list[str], check: bool) -> None:
            if cmd[0] == settings.whisper_deepfilter_bin:
                captured_work_dirs.append(Path(cmd[-1]).parent)
                raise subprocess.CalledProcessError(1, cmd)
            self.fake_run(cmd, check)

        upload = UploadFile(filename="audio.wav", file=io.BytesIO(b"audio"))
        with mock.patch.object(whisper_service.subprocess, "run", side_effect=fail_denoise):
            with self.assertRaisesRegex(whisper_service.WhisperError, "DeepFilterNet failed"):
                asyncio.run(whisper_service.transcribe_upload(upload, settings))
        self.assertEqual(1, len(captured_work_dirs))
        self.assertFalse(captured_work_dirs[0].exists())
        self.assertEqual([], list(self.archive.iterdir()))

    def test_deepfilter_bypass_attenuation_is_rejected_to_avoid_delay_shift(self) -> None:
        for attenuation in (0, 0.005, -1, float("nan"), float("inf")):
            with self.subTest(attenuation=attenuation):
                with self.assertRaisesRegex(whisper_service.WhisperError, "attenuation"):
                    self.transcribe(self.deepfilter_settings(
                        whisper_deepfilter_attenuation_limit_db=attenuation
                    ))
        self.assertEqual([], self.calls)

    def test_deepfilter_minimum_non_bypass_attenuation_is_allowed(self) -> None:
        self.assertEqual("transcribed", self.transcribe(self.deepfilter_settings(
            whisper_deepfilter_attenuation_limit_db=0.01
        )))

    def test_blocking_preparation_allows_event_loop_to_progress(self) -> None:
        release = threading.Event()

        async def scenario() -> None:
            entered = asyncio.Event()
            loop = asyncio.get_running_loop()
            event_loop_thread = threading.get_ident()

            def block_preparation(*args: object) -> None:
                self.assertNotEqual(event_loop_thread, threading.get_ident())
                loop.call_soon_threadsafe(entered.set)
                if not release.wait(5):
                    raise AssertionError("The event loop did not release preparation.")
                _write_wav(args[1], 160)

            upload = UploadFile(filename="audio.wav", file=io.BytesIO(b"audio"))
            with mock.patch.object(whisper_service, "_prepare_audio", side_effect=block_preparation), \
                    mock.patch.object(whisper_service.subprocess, "run", side_effect=self.fake_run):
                task = asyncio.create_task(whisper_service.transcribe_upload(upload, self.settings))
                try:
                    await asyncio.wait_for(entered.wait(), timeout=5)
                    self.assertFalse(task.done())
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
                self.assertEqual("transcribed", task.result())

        asyncio.run(scenario())

    def test_concurrent_requests_execute_one_heavy_pipeline_at_a_time(self) -> None:
        release = threading.Event()
        real_lock = threading.Lock()
        entered_payloads = []

        async def scenario() -> None:
            first_entered = asyncio.Event()
            second_waiting = asyncio.Event()
            loop = asyncio.get_running_loop()

            class ObservedLock:
                def __enter__(self) -> None:
                    if not real_lock.acquire(blocking=False):
                        loop.call_soon_threadsafe(second_waiting.set)
                        real_lock.acquire()

                def __exit__(self, *args: object) -> None:
                    real_lock.release()

            def heavy(audio: bytes, *args: object) -> str:
                entered_payloads.append(audio)
                if audio == b"first":
                    loop.call_soon_threadsafe(first_entered.set)
                    if not release.wait(5):
                        raise AssertionError("First pipeline was not released.")
                return audio.decode()

            with mock.patch.object(whisper_service, "_TRANSCRIPTION_LOCK", ObservedLock()), \
                    mock.patch.object(whisper_service, "_transcribe_audio", side_effect=heavy):
                first = asyncio.create_task(whisper_service.transcribe_upload(
                    UploadFile(file=io.BytesIO(b"first")), self.settings))
                tasks = [first]
                try:
                    await asyncio.wait_for(first_entered.wait(), timeout=5)
                    tasks.append(asyncio.create_task(whisper_service.transcribe_upload(
                        UploadFile(file=io.BytesIO(b"second")), self.settings)))
                    await asyncio.wait_for(second_waiting.wait(), timeout=5)
                    self.assertEqual([b"first"], entered_payloads)
                finally:
                    release.set()
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                self.assertEqual(["first", "second"], results)

        asyncio.run(scenario())

    def test_cancelled_request_keeps_temporary_files_until_worker_finishes(self) -> None:
        release = threading.Event()
        captured_inputs = []

        async def scenario() -> None:
            entered = asyncio.Event()
            loop = asyncio.get_running_loop()

            def block_preparation(source: Path, destination: Path, *args: object) -> None:
                captured_inputs.append(source)
                loop.call_soon_threadsafe(entered.set)
                if not release.wait(5):
                    raise AssertionError("Cancelled request's worker was not released.")
                self.assertTrue(source.is_file())
                _write_wav(destination, 160)

            with mock.patch.object(whisper_service, "_prepare_audio", side_effect=block_preparation), \
                    mock.patch.object(whisper_service.subprocess, "run", side_effect=self.fake_run):
                task = asyncio.create_task(whisper_service.transcribe_upload(
                    UploadFile(file=io.BytesIO(b"audio")), self.settings))
                try:
                    await asyncio.wait_for(entered.wait(), timeout=5)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertTrue(captured_inputs[0].is_file())
                    self.assertTrue(whisper_service._TRANSCRIPTION_LOCK.locked())
                finally:
                    release.set()
                    # Cancellation only detaches the await. Keep the mocks alive
                    # until the worker exits and cleans up its temporary files.
                    await loop.shutdown_default_executor()

        asyncio.run(scenario())
        self.assertFalse(captured_inputs[0].parent.exists())
        self.assertFalse(whisper_service._TRANSCRIPTION_LOCK.locked())
        self.assertEqual(1, len(list(self.archive.glob("whisper_output_*.txt"))))

    def test_whisper_endpoint_response_contract_is_unchanged(self) -> None:
        upload = mock.Mock(spec=UploadFile)

        with mock.patch.object(
            main.whisper_service,
            "transcribe_upload",
            new_callable=mock.AsyncMock,
            return_value="transcribed",
        ) as mocked_transcribe:
            result = asyncio.run(main.whisper(file=upload, language="ja"))

        mocked_transcribe.assert_awaited_once_with(upload, main.settings, language="ja")
        self.assertEqual({"text": "transcribed"}, result)


def _write_wav(path: Path, frames: int, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        output.writeframes(struct.pack("<h", 1000) * frames)


def _settings(
    whisper_tmp_dir: Path,
    whisper_args: tuple[str, ...],
) -> Settings:
    base = whisper_tmp_dir.parent
    return Settings(
        whisper_bin="whisper",
        whisper_model_path="model",
        whisper_args=whisper_args,
        ffmpeg_bin="ffmpeg",
        whisper_tmp_dir=whisper_tmp_dir,
        ytdlp_bin="ytdlp",
        ytdlp_output_dir=base / "tmp/yt",
        rsync_bin="rsync",
        obsidian_vault_root=base / "vault",
        obsidian_export_dir=base / "exports",
        obsidian_target_dirs=("inbox", "journal"),
        obsidian_exclude_tags=("type/snippet", "type/account"),
        obsidian_journal_tag="type/journal",
        obsidian_topic_prefix="topic/",
        obsidian_others_group_name="others",
        google_docs_credentials_path=None,
        google_docs_folder_id=None,
        google_oauth_client_id=None,
        google_oauth_client_secret=None,
        google_oauth_refresh_token=None,
        google_oauth_token_uri="https://oauth2.googleapis.com/token",
    )
