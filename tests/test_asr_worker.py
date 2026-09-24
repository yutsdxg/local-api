from __future__ import annotations

import io
import json
import os
import struct
import subprocess
import sys
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from scripts import asr_worker


class TestASRWorker(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.model = self.base / "model"
        self.model.mkdir()
        self.audio = self.base / "prepared.wav"
        with wave.open(str(self.audio), "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(struct.pack("<h", 1000) * 160)
        self.backend = mock.Mock()
        self.backend.transcribe.return_value = {"text": "日本語", "segments": []}

    def run_worker(self, requests: list[object], *, backend: str = "qwen",
                   model: str | None = None) -> tuple[int, list[dict]]:
        source = io.StringIO("\n".join(json.dumps(value) for value in requests) + "\n")
        output = io.StringIO()
        with mock.patch.object(asr_worker, "load_backend", return_value=self.backend) as load:
            with mock.patch.dict(os.environ):
                code = asr_worker.serve(backend, model or str(self.model), source, output)
            self.load_calls = load.call_count
        return code, [json.loads(line) for line in output.getvalue().splitlines()]

    def request(self, request_id: str, **kwargs: object) -> dict:
        return {"id": request_id, "audio_path": str(self.audio), **kwargs}

    def test_model_is_loaded_once_and_requests_remain_in_order(self) -> None:
        code, events = self.run_worker([
            self.request("first"), self.request("second", prompt="語彙"),
            {"id": "end", "action": "shutdown"}, self.request("never"),
        ])
        self.assertEqual(0, code)
        self.assertEqual(1, self.load_calls)
        self.assertEqual(["ready", "result", "result", "stopped"], [e["event"] for e in events])
        self.assertEqual(["first", "second"], [e["id"] for e in events[1:3]])
        self.assertGreaterEqual(events[0]["load_seconds"], 0)
        self.assertEqual(0.01, events[1]["audio_seconds"])
        self.assertGreaterEqual(events[1]["elapsed_seconds"], 0)
        self.assertEqual("語彙", self.backend.transcribe.call_args_list[1].args[3])

    def test_request_failure_does_not_prevent_next_request(self) -> None:
        self.backend.transcribe.side_effect = [RuntimeError("decoding failed"),
                                               {"text": "回復", "segments": []}]
        code, events = self.run_worker([self.request("bad"), self.request("good")])
        self.assertEqual(0, code)
        self.assertEqual("error", events[1]["event"])
        self.assertEqual("bad", events[1]["id"])
        self.assertEqual("RuntimeError", events[1]["error"]["type"])
        self.assertEqual("回復", events[2]["text"])

    def test_bad_protocol_and_unsupported_options_are_not_silently_accepted(self) -> None:
        code, events = self.run_worker([
            [], self.request("bad-option", options={"model": "remote/repo"}),
            self.request("nan", options={"temperature": float("nan")}),
            self.request("bad-limit", options={"max_tokens": 0}),
            self.request("wrong-key", audio="elsewhere"), self.request("good"),
        ])
        self.assertEqual(0, code)
        self.assertEqual(["error"] * 5, [e["event"] for e in events[1:6]])
        self.assertEqual("result", events[6]["event"])
        self.backend.transcribe.assert_called_once()

    def test_relative_or_missing_models_fail_before_dependency_import(self) -> None:
        for model in ("org/model", str(self.base / "absent")):
            with self.subTest(model=model):
                code, events = self.run_worker([], model=model)
                self.assertEqual(1, code)
                self.assertEqual("startup_error", events[0]["event"])
                self.assertEqual(0, self.load_calls)

    def test_startup_exception_is_json_and_offline_flags_precede_load(self) -> None:
        output = io.StringIO()

        def fail_load(*args: object) -> None:
            self.assertEqual("1", os.environ["HF_HUB_OFFLINE"])
            self.assertEqual("1", os.environ["TRANSFORMERS_OFFLINE"])
            print("dependency log")
            raise ImportError("dependency missing")

        errors = io.StringIO()
        with mock.patch.object(asr_worker, "load_backend", side_effect=fail_load), \
                mock.patch.dict(os.environ), mock.patch("sys.stderr", errors):
            code = asr_worker.serve("qwen", str(self.model), io.StringIO(), output)
        self.assertEqual(1, code)
        self.assertEqual("ImportError", json.loads(output.getvalue())["error"]["type"])
        self.assertIn("dependency log", errors.getvalue())

    def test_malformed_json_recovers(self) -> None:
        output = io.StringIO()
        source = io.StringIO('{\n' + json.dumps(self.request("good")) + '\n')
        with mock.patch.object(asr_worker, "load_backend", return_value=self.backend), \
                mock.patch.dict(os.environ):
            code = asr_worker.serve("qwen", str(self.model), source, output)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(0, code)
        self.assertEqual(["ready", "error", "result"], [e["event"] for e in events])

    def test_audio_format_and_truncation_are_rejected(self) -> None:
        for rate, channels, width, frames in ((8000, 1, 2, 10), (16000, 2, 2, 10),
                                             (16000, 1, 4, 10), (16000, 1, 2, 0)):
            with self.subTest(rate=rate, channels=channels, width=width, frames=frames):
                with wave.open(str(self.audio), "wb") as writer:
                    writer.setparams((channels, width, rate, 0, "NONE", "not compressed"))
                    writer.writeframes(bytes(frames * width * channels))
                with self.assertRaises(ValueError):
                    asr_worker.read_pcm(self.audio)
        with wave.open(str(self.audio), "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(bytes(20))
        self.audio.write_bytes(self.audio.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, "truncated"):
            asr_worker.read_pcm(self.audio)

    def test_timestamp_padding_is_clamped_and_nan_is_rejected(self) -> None:
        result = asr_worker.normalize_segments([
            {"start": 0, "end": 1, "text": "短音"},
            SimpleNamespace(start=0.2, end=0.4, text="別"),
        ], 0.5)
        self.assertEqual([0.5, 0.4], [s["end"] for s in result])
        with self.assertRaisesRegex(ValueError, "timestamps"):
            asr_worker.normalize_segments([{"start": 0, "end": float("nan"), "text": ""}], 1)

    def local_backend(self, name: str) -> asr_worker.LocalBackend:
        # Exercise adapter contracts without importing numpy, MLX, or model code.
        backend = asr_worker.LocalBackend.__new__(asr_worker.LocalBackend)
        backend.name = name
        backend.model_path = self.model
        backend.np = mock.MagicMock()
        backend.mx = mock.Mock()
        backend.mx.get_peak_memory.return_value = 4096
        backend.model = mock.Mock()
        return backend

    def test_qwen_language_prompt_timing_and_token_budget_are_reported(self) -> None:
        backend = self.local_backend("qwen")
        backend.model.generate.return_value = SimpleNamespace(
            text="発話", segments=[{"start": 0, "end": 1, "text": "発話"}],
            generation_tokens=20,
        )
        options = asr_worker.request_options("qwen", {"max_tokens": 20})
        result = backend.transcribe(b"\0\0", 0.5, "ja", "語彙", options)
        keywords = backend.model.generate.call_args.kwargs
        self.assertEqual("Japanese", keywords["language"])
        self.assertEqual("語彙", keywords["system_prompt"])
        self.assertEqual(1, keywords["batch_size"])
        self.assertFalse(keywords["stream"])
        self.assertEqual("input-chunk", result["timestamp_kind"])
        self.assertEqual(0.5, result["segments"][0]["end"])
        self.assertEqual(20, result["generation_tokens"])
        self.assertEqual(4096, result["peak_memory_bytes"])
        self.assertEqual(1, len(result["warnings"]))
        backend.mx.reset_peak_memory.assert_called_once()

    def test_parakeet_does_not_silently_ignore_prompt_or_language(self) -> None:
        backend = self.local_backend("parakeet")
        options = asr_worker.request_options("parakeet", {})
        for language, prompt in (("en", None), ("ja", "ignored?")):
            with self.subTest(language=language, prompt=prompt), self.assertRaises(ValueError):
                backend.transcribe(b"\0\0", 1, language, prompt, options)
        backend.model.generate.assert_not_called()
        backend.model.generate.return_value = SimpleNamespace(
            text="発話", sentences=[SimpleNamespace(start=0.1, end=0.5, text="発話")],
        )
        result = backend.transcribe(b"\0\0", 1, "ja", None, options)
        self.assertEqual(backend.mx.float32, backend.model.generate.call_args.kwargs["dtype"])
        self.assertIsNone(backend.model.generate.call_args.kwargs["chunk_duration"])
        self.assertEqual("parakeet-sentence", result["timestamp_kind"])
        self.assertEqual(0.1, result["segments"][0]["start"])

    def test_whisper_reuses_preloaded_holder_model(self) -> None:
        modules = {"numpy": mock.MagicMock(), "mlx.core": mock.Mock(),
                   "mlx_whisper.transcribe": mock.Mock()}
        modules["mlx.core"].get_peak_memory.return_value = 4096
        whisper = modules["mlx_whisper.transcribe"]
        whisper.transcribe.return_value = {"text": "発話", "segments": []}
        with mock.patch.object(asr_worker.importlib, "import_module", side_effect=modules.__getitem__):
            backend = asr_worker.LocalBackend("mlx-whisper", self.model)
        for _ in range(2):
            result = backend.transcribe(b"\0\0", 1, "ja", None,
                                        asr_worker.request_options("mlx-whisper", {}))
        whisper.ModelHolder.get_model.assert_called_once_with(str(self.model), modules["mlx.core"].float16)
        modules["mlx.core"].eval.assert_called_once()
        self.assertEqual(2, whisper.transcribe.call_count)
        keywords = whisper.transcribe.call_args.kwargs
        self.assertEqual(str(self.model), keywords["path_or_hf_repo"])
        self.assertTrue(keywords["fp16"])
        self.assertNotIn("beam_size", keywords)
        self.assertEqual("whisper-segment", result["timestamp_kind"])

    def test_native_and_python_logs_cannot_corrupt_protocol_stdout(self) -> None:
        command = [sys.executable, "-c", (
            "import os; from scripts.asr_worker import protocol_stdout, write_event; "
            "output=protocol_stdout(); print('python log'); os.write(1,b'native log\\n'); "
            "write_event(output, {'event':'ready'}); output.close()"
        )]
        result = subprocess.run(command, check=True, capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1])
        self.assertEqual({"event": "ready"}, json.loads(result.stdout))
        self.assertIn("python log", result.stderr)
        self.assertIn("native log", result.stderr)


if __name__ == "__main__":
    unittest.main()
