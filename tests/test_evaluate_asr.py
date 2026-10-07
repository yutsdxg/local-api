from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import time
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts import evaluate_asr


class TestEvaluateASR(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory(prefix="asr evaluation ")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.audio = self.base / "日本語 audio 'sample'.wav"
        with wave.open(str(self.audio), "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(b"\x01\x00" * 160)
        self.model = self.base / "model with spaces.bin"
        self.model.write_bytes(b"fake local model")

    def executable(self, name: str, body: str) -> Path:
        path = self.base / name
        path.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
        path.chmod(0o700)
        return path

    def cpp_condition(self, binary: Path, name: str = "cpp") -> dict:
        return {"name": name, "backend": "whisper.cpp", "binary": str(binary),
                "model": str(self.model)}

    def case(self, **kwargs: object) -> dict:
        return {"name": "japanese", "prepared_audio": str(self.audio),
                "speech_audio": str(self.audio), "language": "ja", **kwargs}

    def vad_manifest(self, audio: Path, samples: int, **overrides: object) -> Path:
        manifest = audio.with_suffix(".vad.json")
        manifest.write_text(json.dumps({"output": {
            "sha256": evaluate_asr.digest(audio), "samples": samples, **overrides,
        }}), encoding="utf-8")
        return manifest

    def run_manifest(self, conditions: list[dict], cases: list[dict] | None = None,
                     *extra: str) -> tuple[int, Path, dict | None]:
        manifest = self.base / "manifest.json"
        manifest.write_text(json.dumps({"conditions": conditions, "cases": cases or [self.case()]},
                                       ensure_ascii=False), encoding="utf-8")
        output = self.base / "evaluation results"
        args = ["evaluate_asr.py", str(manifest), "--output-dir", str(output), *extra]
        with mock.patch.object(sys, "argv", args), \
                mock.patch.object(evaluate_asr, "timed_command", side_effect=lambda command: command), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                evaluate_asr.main()
        report_path = output / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        return stopped.exception.code, output, report

    def test_unverified_reference_never_opens_or_scores_a_transcript(self) -> None:
        missing = self.base / "reference not yet checked.txt"
        for flag in (False, None, "true", 1):
            with self.subTest(flag=flag):
                result = evaluate_asr.metrics("参考結果", {"path": str(missing),
                                                          "human_verified": flag})
                self.assertNotIn("cer", result)
                self.assertNotIn("reference_sha256", result)
        self.assertNotIn("cer", evaluate_asr.metrics("参考結果", None))

    def test_verified_reference_scores_edit_errors_and_records_its_hash(self) -> None:
        reference = self.base / "verified.txt"
        reference.write_text("Ａ社は１２万円。", encoding="utf-8")
        result = evaluate_asr.metrics("A社は１３万円。", {"path": str(reference), "human_verified": True})
        self.assertEqual(1, result["edit_distance"])
        self.assertEqual(8, result["reference_characters"])
        self.assertAlmostEqual(1 / 8, result["cer"])
        self.assertEqual(evaluate_asr.digest(reference), result["reference_sha256"])

    def test_cer_preserves_numeric_decimal_points_and_signs(self) -> None:
        reference = self.base / "verified numeric.txt"
        for truth, hypothesis in (("12.5万円", "125万円"), ("-3度", "3度"),
                                  ("100円。", "100円")):
            with self.subTest(truth=truth):
                reference.write_text(truth, encoding="utf-8")
                result = evaluate_asr.metrics(hypothesis, {"path": str(reference),
                                                           "human_verified": True})
                self.assertEqual(1, result["edit_distance"])
                self.assertGreater(result["cer"], 0)

    def test_empty_verified_reference_does_not_claim_zero_error_rate(self) -> None:
        reference = self.base / "verified empty.txt"
        reference.write_text("", encoding="utf-8")
        result = evaluate_asr.metrics("無音への挿入", {"path": str(reference), "human_verified": True})
        self.assertIsNone(result["cer"])
        self.assertGreater(result["edit_distance"], 0)

    def test_sentence_diagnostics_find_repetition_within_one_line(self) -> None:
        text = "確認します。確認します。確認します。終了します。確認します。"
        result = evaluate_asr.metrics(text, None)
        self.assertEqual(0, result["adjacent_duplicate_lines"])
        self.assertEqual(5, result["sentence_count"])
        self.assertEqual(4, result["max_sentence_occurrences"])
        self.assertEqual(3, result["max_consecutive_sentence_occurrences"])
        self.assertNotIn("cer", result)
        self.assertTrue(all(isinstance(value, int) for value in result.values()))

    def test_sentence_diagnostics_preserve_decimal_values_and_handle_empty_input(self) -> None:
        result = evaluate_asr.metrics("１２．５万円です。\n12.5万円です！125万円です？", None)
        self.assertEqual(3, result["sentence_count"])
        self.assertEqual(2, result["max_sentence_occurrences"])
        self.assertEqual(2, result["max_consecutive_sentence_occurrences"])
        empty = evaluate_asr.metrics(" \n。！？", None)
        for key in ("sentence_count", "max_sentence_occurrences",
                    "max_consecutive_sentence_occurrences"):
            self.assertEqual(0, empty[key])

    def test_common_vad_requires_manifest_before_starting_cli(self) -> None:
        condition = {**self.cpp_condition(self.model), "vad_mode": "common-silero",
                     "input_key": "speech_audio"}
        with mock.patch.object(evaluate_asr.subprocess, "Popen") as process:
            with self.assertRaisesRegex(ValueError, "requires a VAD manifest"):
                evaluate_asr.run_cpp(condition, self.case(), self.base, 1)
        process.assert_not_called()

    def test_speech_input_records_matching_vad_manifest_and_honors_default_key(self) -> None:
        manifest = self.vad_manifest(self.audio, 160)
        case = self.case(vad_manifest=str(manifest), prepared_audio=str(self.base / "unused.wav"))
        # The persistent worker defaults to speech_audio without requiring backend metadata here.
        info = evaluate_asr.input_info({"vad_mode": "common-silero"}, case, "speech_audio")
        self.assertEqual(str(self.audio), info["path"])
        self.assertEqual(160, info["frames"])
        self.assertEqual(evaluate_asr.digest(manifest), info["vad_manifest_sha256"])

    def test_speech_input_rejects_independent_vad_hash_and_sample_mismatches(self) -> None:
        for override in ({"sha256": "0" * 64}, {"samples": 161}):
            with self.subTest(override=override):
                manifest = self.vad_manifest(self.audio, 160)
                contents = json.loads(manifest.read_text())
                contents["output"].update(override)
                manifest.write_text(json.dumps(contents), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "does not match the recorded VAD output"):
                    evaluate_asr.input_info({"input_key": "speech_audio"},
                                            self.case(vad_manifest=str(manifest)))

    def test_cli_success_preserves_paths_arguments_and_repeat_outputs(self) -> None:
        binary = self.executable("fake whisper cli", (
            "import json, pathlib, sys\n"
            "prefix = sys.argv[sys.argv.index('-of') + 1]\n"
            "pathlib.Path(prefix + '.txt').write_text('日本語42', encoding='utf-8')\n"
            "print(json.dumps(sys.argv[1:], ensure_ascii=False))\n"
        ))
        condition = self.cpp_condition(binary)
        condition["args"] = ["--prompt", "数値 '42' の説明"]
        code, output, report = self.run_manifest([condition], None, "--repeat", "2")
        self.assertEqual(0, code)
        self.assertEqual(["ok", "ok"], [row["status"] for row in report["results"]])
        for repeat, row in enumerate(report["results"], 1):
            transcript = output / "cpp" / f"japanese-{repeat}" / "transcript.txt"
            self.assertEqual("日本語42", transcript.read_text())
            self.assertEqual(str(transcript), row["transcript"])
            self.assertEqual(evaluate_asr.digest(self.audio), row["input"]["sha256"])
            self.assertEqual(0.01, row["input"]["seconds"])
            self.assertNotIn("cer", row["metrics"])
            arguments = json.loads(transcript.with_name("process.log").read_text())
            self.assertEqual(str(self.model), arguments[arguments.index("-m") + 1])
            self.assertEqual(str(self.audio), arguments[arguments.index("-f") + 1])
            self.assertEqual("数値 '42' の説明", arguments[-1])
        self.assertEqual(evaluate_asr.digest(self.model),
                         report["conditions"][0]["model"]["files"][0]["sha256"])

    def test_nonzero_cli_is_reported_and_next_condition_still_runs(self) -> None:
        bad = self.executable("fails", "import sys\nprint('decoder failed', file=sys.stderr)\nsys.exit(7)\n")
        good = self.executable("succeeds", (
            "import pathlib, sys\n"
            "pathlib.Path(sys.argv[sys.argv.index('-of')+1]+'.txt').write_text('成功')\n"
        ))
        code, output, report = self.run_manifest([self.cpp_condition(bad, "bad"),
                                                self.cpp_condition(good, "good")])
        self.assertEqual(1, code)
        self.assertEqual(["error", "ok"], [row["status"] for row in report["results"]])
        self.assertIn("exit 7", report["results"][0]["error"])
        self.assertIn("decoder failed", (output / "bad/japanese-1/process.log").read_text())
        self.assertNotIn("metrics", report["results"][0])

    def test_success_exit_without_output_is_an_error_not_empty_transcription(self) -> None:
        binary = self.executable("no output", "print('no transcript was produced')\n")
        code, _, report = self.run_manifest([self.cpp_condition(binary)])
        self.assertEqual(1, code)
        self.assertEqual("error", report["results"][0]["status"])
        self.assertIn("transcript.txt", report["results"][0]["error"])
        self.assertNotIn("transcript_sha256", report["results"][0])

    def test_missing_model_is_saved_and_does_not_abort_later_conditions(self) -> None:
        binary = self.executable("success", (
            "import pathlib, sys\n"
            "pathlib.Path(sys.argv[sys.argv.index('-of')+1]+'.txt').write_text('成功')\n"
        ))
        missing = self.cpp_condition(binary, "missing")
        missing["model"] = str(self.base / "absent model")
        code, _, report = self.run_manifest([missing, self.cpp_condition(binary, "good")])
        self.assertEqual(1, code)
        self.assertIn("error", report["conditions"][0])
        self.assertEqual("ok", report["results"][-1]["status"])
        self.assertEqual("good", report["results"][-1]["condition"])

    def test_cli_timeout_kills_its_process_group_and_records_failure(self) -> None:
        binary = self.executable("slow cli", "import time\ntime.sleep(30)\n")
        with mock.patch.object(evaluate_asr.os, "killpg", wraps=os.killpg) as kill:
            started = time.monotonic()
            code, _, report = self.run_manifest([self.cpp_condition(binary)], None, "--timeout", "0.2")
        self.assertEqual(1, code)
        self.assertIn("exceeded", report["results"][0]["error"])
        self.assertLess(time.monotonic() - started, 3)
        kill.assert_called_once()
        self.assertEqual(signal.SIGKILL, kill.call_args.args[1])

    def worker(self, source: str, *, timeout: float = 2) -> evaluate_asr.Worker:
        script = self.executable("fake worker", source)
        condition = {"python": sys.executable, "backend": "qwen", "model": str(self.model)}
        with mock.patch.object(evaluate_asr, "timed_command",
                               return_value=[sys.executable, str(script)]):
            worker = evaluate_asr.Worker(condition, self.base, timeout)
        self.addCleanup(worker.close)
        return worker

    def test_worker_client_passes_request_and_keeps_protocol_in_order(self) -> None:
        worker = self.worker(
            "import json, os, sys\n"
            "print(json.dumps({'event':'ready', 'offline':os.environ['HF_HUB_OFFLINE']}), flush=True)\n"
            "for line in sys.stdin:\n"
            "    request = json.loads(line)\n"
            "    if request.get('action') == 'shutdown': break\n"
            "    print(json.dumps({'event':'result', 'id':request['id'], 'text':'認識結果', "
            "'segments':[], 'request':request}), flush=True)\n"
        )
        self.assertEqual("1", worker.ready["offline"])
        condition = {"input_key": "speech_audio", "prompt": "専門語", "options": {"max_tokens": 512}}
        for run_id in ("one", "two"):
            result = worker.transcribe(condition, self.case(), run_id)
            self.assertEqual(run_id, result["id"])
            self.assertEqual(str(self.audio), result["request"]["audio_path"])
            self.assertEqual("ja", result["request"]["language"])
            self.assertEqual("専門語", result["request"]["prompt"])
            self.assertEqual({"max_tokens": 512}, result["request"]["options"])
            self.assertEqual(evaluate_asr.digest(self.audio), result["input"]["sha256"])

    def test_empty_common_vad_input_skips_worker_request_and_preserves_next_response(self) -> None:
        worker = self.worker(
            "import json, sys\n"
            "print('{\"event\":\"ready\"}', flush=True)\n"
            "count = 0\n"
            "for line in sys.stdin:\n"
            "    request = json.loads(line)\n"
            "    if request.get('action') == 'shutdown': break\n"
            "    count += 1\n"
            "    print(json.dumps({'event':'result', 'id':request['id'], "
            "'text':'認識結果', 'segments':[], 'request_count':count}), flush=True)\n"
        )
        empty = self.base / "no speech.wav"
        with wave.open(str(empty), "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(b"")
        manifest = self.vad_manifest(empty, 0)
        condition = {"vad_mode": "common-silero"}
        result = worker.transcribe(condition,
                                   self.case(speech_audio=str(empty), vad_manifest=str(manifest)), "empty")
        self.assertEqual("empty", result["id"])
        self.assertEqual("", result["text"])
        self.assertEqual([], result["segments"])
        self.assertTrue(result["skipped_empty"])
        self.assertEqual(0, result["elapsed_seconds"])
        self.assertEqual(0, result["wall_seconds"])
        self.assertEqual(evaluate_asr.digest(manifest), result["input"]["vad_manifest_sha256"])
        nonempty_manifest = self.vad_manifest(self.audio, 160)
        subsequent = worker.transcribe(condition, self.case(vad_manifest=str(nonempty_manifest)), "speech")
        self.assertEqual("speech", subsequent["id"])
        self.assertEqual(1, subsequent["request_count"])

    def test_worker_client_rejects_wrong_request_id(self) -> None:
        worker = self.worker(
            "import json, sys\n"
            "print('{\"event\":\"ready\"}', flush=True)\n"
            "for line in sys.stdin:\n"
            "    if json.loads(line).get('action') == 'shutdown': break\n"
            "    print('{\"event\":\"result\",\"id\":\"stale\",\"text\":\"wrong\"}', flush=True)\n"
        )
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            worker.transcribe({}, self.case(), "wanted")

    def test_worker_stdout_noise_cannot_be_accepted_as_a_result(self) -> None:
        worker = self.worker(
            "import json, sys\n"
            "print('{\"event\":\"ready\"}', flush=True)\n"
            "for line in sys.stdin:\n"
            "    if json.loads(line).get('action') == 'shutdown': break\n"
            "    print('unexpected library log', flush=True)\n"
        )
        with self.assertRaises(json.JSONDecodeError):
            worker.transcribe({}, self.case(), "wanted")

    def test_worker_unexpected_exit_reports_error(self) -> None:
        worker = self.worker("print('{\"event\":\"ready\"}', flush=True)\n")
        with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
            worker.receive()

    def test_worker_timeout_and_failed_graceful_shutdown_kill_process(self) -> None:
        worker = self.worker(
            "import time\nprint('{\"event\":\"ready\"}', flush=True)\ntime.sleep(30)\n",
            timeout=0.2,
        )
        with self.assertRaisesRegex(TimeoutError, "exceeded"):
            worker.transcribe({}, self.case(), "slow")
        original_wait = worker.process.wait
        # Exercise the forced-shutdown branch without spending its full grace period.
        def short_wait(timeout: float | None = None) -> int:
            return original_wait(timeout=0.05 if timeout is not None else None)

        with mock.patch.object(worker.process, "wait", side_effect=short_wait), \
                mock.patch.object(evaluate_asr.os, "killpg", wraps=os.killpg) as kill:
            worker.close()
        kill.assert_called_once_with(worker.process.pid, signal.SIGKILL)
        self.assertIsNotNone(worker.process.returncode)

    def test_path_traversal_names_are_rejected_before_output_creation(self) -> None:
        for field in ("condition", "case"):
            with self.subTest(field=field):
                condition = self.cpp_condition(self.model)
                case = self.case()
                (condition if field == "condition" else case)["name"] = "../escape"
                code, output, report = self.run_manifest([condition], [case])
                self.assertEqual(2, code)
                self.assertFalse(output.exists())
                self.assertIsNone(report)

    def test_duplicate_names_are_rejected_before_output_creation(self) -> None:
        condition = self.cpp_condition(self.model)
        for conditions, cases in (([condition, condition], [self.case()]),
                                  ([condition], [self.case(), self.case()])):
            with self.subTest(conditions=len(conditions), cases=len(cases)):
                code, output, report = self.run_manifest(conditions, cases)
                self.assertEqual(2, code)
                self.assertFalse(output.exists())
                self.assertIsNone(report)


    def test_interrupted_cli_is_killed_and_reaped_before_interrupt_propagates(self) -> None:
        for error in (KeyboardInterrupt(), RuntimeError("wait failed")):
            with self.subTest(error=type(error).__name__):
                process = mock.Mock()
                process.pid = 123456
                process.poll.return_value = None
                process.wait.side_effect = [error, -signal.SIGKILL]
                with mock.patch.object(evaluate_asr.subprocess, "Popen", return_value=process), \
                        mock.patch.object(evaluate_asr.os, "killpg") as kill:
                    with self.assertRaises(type(error)) as raised:
                        evaluate_asr.run_cpp(self.cpp_condition(self.model), self.case(), self.base, 1)
                self.assertIs(error, raised.exception)
                kill.assert_called_once_with(process.pid, signal.SIGKILL)
                self.assertEqual([mock.call(timeout=1), mock.call()], process.wait.call_args_list)


if __name__ == "__main__":
    unittest.main()
