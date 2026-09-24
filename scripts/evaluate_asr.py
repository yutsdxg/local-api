"""Run sequential, local ASR comparisons from an explicit JSON manifest.

Inputs, transcripts and reports belong under ignored data/. References are
scored only when the manifest explicitly marks a human-verified transcript.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import unicodedata
import wave
from collections import Counter
from itertools import groupby
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audio_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"Expected 16 kHz mono PCM16: {path}")
        return {"path": str(path), "sha256": digest(path),
                "frames": reader.getnframes(), "seconds": reader.getnframes() / 16000}


def input_info(condition: dict, case: dict, default_key: str = "prepared_audio") -> dict:
    key = condition.get("input_key", default_key)
    info = audio_info(Path(case[key]).resolve(strict=True))
    if condition.get("vad_mode") == "common-silero" and not case.get("vad_manifest"):
        raise ValueError("common-silero requires a VAD manifest")
    if key == "speech_audio" and case.get("vad_manifest"):
        manifest_path = Path(case["vad_manifest"]).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text())
        if manifest["output"]["sha256"] != info["sha256"] or manifest["output"]["samples"] != info["frames"]:
            raise ValueError("Speech input does not match the recorded VAD output")
        info["vad_manifest_sha256"] = digest(manifest_path)
    return info


def normalized(text: str) -> str:
    # Preserve punctuation too: 12.5 / 125 and -3 / 3 must remain different.
    return "".join(c for c in unicodedata.normalize("NFKC", text) if not c.isspace())


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def metrics(text: str, reference: dict | None) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Heuristic sentence boundaries include Japanese stops and !/?, not decimal
    # points. Real speech can repeat too: these counts diagnose output, not accuracy.
    sentences = [part for part in re.split(r"[。!?]+", normalized(text)) if part]
    result = {"characters": len(text), "nonempty_lines": len(lines),
              "adjacent_duplicate_lines": sum(a == b for a, b in zip(lines, lines[1:])),
              "sentence_count": len(sentences),
              "max_sentence_occurrences": max(Counter(sentences).values(), default=0),
              "max_consecutive_sentence_occurrences": max(
                  (sum(1 for _ in group) for _, group in groupby(sentences)), default=0)}
    if reference and reference.get("human_verified") is True:
        path = Path(reference["path"]).resolve(strict=True)
        truth = normalized(path.read_text(encoding="utf-8"))
        hypothesis = normalized(text)
        errors = edit_distance(truth, hypothesis)
        result.update({"reference_sha256": digest(path), "reference_characters": len(truth),
                       "edit_distance": errors, "cer": errors / len(truth) if truth else None,
                       "normalization": "NFKC; remove whitespace; preserve punctuation, signs and digits"})
    return result


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def peak_rss(log: Path) -> int | None:
    matches = re.findall(r"^\s*(\d+)\s+maximum resident set size\s*$", log.read_text(errors="replace"), re.M)
    return int(matches[-1]) if matches else None


def timed_command(command: list[str]) -> list[str]:
    return ["/usr/bin/time", "-l", *command] if sys.platform == "darwin" else command


def run_cpp(condition: dict, case: dict, output: Path, timeout: float) -> dict:
    info = input_info(condition, case)
    path = Path(info["path"])
    prefix = output / "transcript"
    command = [condition["binary"], "-m", condition["model"], "-f", str(path),
               "-otxt", "-of", str(prefix), "-l", case.get("language", "ja"),
               *condition.get("args", [])]
    started = time.monotonic()
    log = output / "process.log"
    if info["frames"] == 0:
        prefix.with_suffix(".txt").write_text("", encoding="utf-8")
        return {"text": "", "elapsed_seconds": 0.0, "input": info, "skipped_empty": True}
    with log.open("w") as stream:
        process = subprocess.Popen(timed_command(command), stdout=stream, stderr=stream,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate(process)
            raise TimeoutError(f"ASR exceeded {timeout}s; see {log}")
        except BaseException:
            # The child has its own session, so interrupting this runner does
            # not interrupt ASR. Reap it before another evaluation can start.
            terminate(process)
            raise
    elapsed = time.monotonic() - started
    if code:
        raise RuntimeError(f"ASR exit {code}; see {log}")
    text = prefix.with_suffix(".txt").read_text(encoding="utf-8")
    return {"text": text, "elapsed_seconds": elapsed, "peak_process_rss_bytes": peak_rss(log),
            "command": command, "input": info, "timing_scope": "CLI startup + model load + VAD if enabled + decode"}


class Worker:
    def __init__(self, condition: dict, output: Path, timeout: float):
        self.timeout = timeout
        self.log_path = output / "worker.log"
        self.log = self.log_path.open("w")
        command = [condition["python"], str(Path(__file__).with_name("asr_worker.py")),
                   "--backend", condition["backend"], "--model", condition["model"]]
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "HF_HOME": str(output / "offline-hf"), "TOKENIZERS_PARALLELISM": "false"}
        started = time.monotonic()
        self.process = subprocess.Popen(timed_command(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1, env=env, start_new_session=True)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        try:
            self.ready = self.receive()
            if self.ready.get("event") != "ready":
                raise RuntimeError(f"Worker startup failed: {self.ready}")
            self.startup_seconds = time.monotonic() - started
        except BaseException:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def receive(self) -> dict:
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError(f"Worker exceeded {self.timeout}s; see {self.log_path}")
        if line is None:
            raise RuntimeError(f"Worker exited unexpectedly; see {self.log_path}")
        return json.loads(line)

    def transcribe(self, condition: dict, case: dict, run_id: str) -> dict:
        info = input_info(condition, case, "speech_audio")
        path = Path(info["path"])
        if info["frames"] == 0:
            return {"event": "result", "id": run_id, "text": "", "segments": [], "input": info,
                    "elapsed_seconds": 0.0, "wall_seconds": 0.0, "skipped_empty": True,
                    "timing_scope": "empty common-VAD input; decoder skipped"}
        request = {"id": run_id, "audio_path": str(path), "language": case.get("language", "ja"),
                   "options": condition.get("options", {})}
        if condition.get("prompt"):
            request["prompt"] = condition["prompt"]
        started = time.monotonic()
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = self.receive()
        result["wall_seconds"] = time.monotonic() - started
        if result.get("event") != "result" or result.get("id") != run_id:
            raise RuntimeError(f"Worker request failed: {result}")
        result.update({"input": info, "timing_scope": "persistent worker decode; excludes startup"})
        return result

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write('{"action":"shutdown"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            terminate(self.process)
        finally:
            for stream in (self.process.stdin, self.process.stdout, self.log):
                if stream:
                    stream.close()


def fingerprint(path: str) -> dict:
    candidate = Path(path).resolve(strict=True)
    files = [candidate] if candidate.is_file() else sorted(
        p for p in candidate.rglob("*") if p.is_file() and ".cache" not in p.relative_to(candidate).parts)
    return {"path": str(candidate), "files": [
        {"path": str(p.relative_to(candidate)) if candidate.is_dir() else p.name,
         "size": p.stat().st_size, "sha256": digest(p)} for p in files]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    if args.repeat < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("repeat and timeout must be positive and finite")
    manifest = json.loads(args.manifest.read_text())
    for key in ("conditions", "cases"):
        names = [item["name"] for item in manifest[key]]
        if len(set(names)) != len(names):
            parser.error(f"duplicate {key} names are not allowed")
    for selected, key in ((args.only, "conditions"), (args.cases, "cases")):
        if selected and set(selected) - {item["name"] for item in manifest[key]}:
            parser.error(f"unknown {key} requested")
    conditions = [c for c in manifest["conditions"] if not args.only or c["name"] in args.only]
    cases = [c for c in manifest["cases"] if not args.cases or c["name"] in args.cases]
    if not conditions or not cases:
        parser.error("no matching conditions or cases")
    for name in [c["name"] for c in conditions] + [c["name"] for c in cases]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            parser.error("names must use letters, digits, underscore or hyphen")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"manifest": manifest, "manifest_sha256": digest(args.manifest), "system": platform.platform(),
              "python": sys.version, "repeat": args.repeat, "conditions": [], "results": [],
              "limitations": "Sequential execution. No accuracy claims without human-verified references. Native and common-VAD inputs are separate conditions. CLI and persistent worker timing scopes differ."}
    report_path = output / "report.json"
    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save()
    failed = False
    for condition in conditions:
        condition_dir = output / condition["name"]
        condition_dir.mkdir()
        metadata = {"configuration": condition}
        report["conditions"].append(metadata)
        worker = None
        try:
            metadata["model"] = fingerprint(condition["model"])
            if condition.get("binary"):
                metadata["binary"] = fingerprint(condition["binary"])
            if condition["backend"] != "whisper.cpp":
                worker = Worker(condition, condition_dir, args.timeout)
                metadata.update({"worker_ready": worker.ready, "startup_seconds": worker.startup_seconds})
            for repeat in range(args.repeat):
                for case in cases:
                    run_id = f'{case["name"]}-{repeat + 1}'
                    run_dir = condition_dir / run_id
                    run_dir.mkdir()
                    entry = {"condition": condition["name"], "case": case["name"], "repeat": repeat + 1}
                    try:
                        result = worker.transcribe(condition, case, run_id) if worker else run_cpp(condition, case, run_dir, args.timeout)
                        text = result.pop("text")
                        (run_dir / "transcript.txt").write_text(text, encoding="utf-8")
                        entry.update(result)
                        entry.update({"status": "ok", "transcript": str(run_dir / "transcript.txt"),
                                      "transcript_sha256": digest(run_dir / "transcript.txt"),
                                      "metrics": metrics(text, case.get("reference"))})
                    except Exception as exc:
                        failed = True
                        entry.update({"status": "error", "error": str(exc)})
                    report["results"].append(entry)
                    save()
                    print(json.dumps({k:entry[k] for k in ("condition", "case", "repeat", "status")}), flush=True)
                    # A failed request can leave the protocol out of sync.
                    if worker and entry["status"] != "ok":
                        raise RuntimeError(entry["error"])
        except Exception as exc:
            failed = True
            metadata["error"] = str(exc)
            print(json.dumps({"condition": condition["name"], "error": str(exc)}), flush=True)
        finally:
            if worker:
                worker.close()
                metadata["peak_worker_process_rss_bytes"] = peak_rss(worker.log_path)
            save()
    print(report_path)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
