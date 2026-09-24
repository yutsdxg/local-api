"""Explicitly download pinned, public evaluation models; never used by the API.

Run with the ASR environment. Model files and provenance stay under data/.
No audio is read or uploaded. Use --list to inspect the catalog without network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


CATALOG = {
    "whisper-ggml": {
        "repo": "ggerganov/whisper.cpp", "revision": "5359861c739e955e79d9a303bcbc70fb988958b1",
        "files": ["ggml-large-v3.bin", "ggml-large-v3-turbo.bin", "README.md"],
    },
    "whisper-large-v3-mlx": {
        "repo": "mlx-community/whisper-large-v3-mlx", "revision": "49e6aa286ad60c14352c404340ded53710378a11",
        "files": ["config.json", "weights.npz", "README.md"],
    },
    "qwen3-asr-1.7b-bf16": {
        "repo": "mlx-community/Qwen3-ASR-1.7B-bf16", "revision": "e1f6c266914abc5a46e8756e02580f834a6cf8a7",
        "files": ["*.json", "*.safetensors", "merges.txt", "README.md"],
    },
    "parakeet-ja": {
        "repo": "mlx-community/parakeet-tdt_ctc-0.6b-ja", "revision": "e3810190ff521dcd208bc444a71bf877f3864566",
        "files": ["config.json", "model.safetensors", "tokenizer.model", "tokenizer.vocab", "vocab.txt", "README.md"],
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--models", nargs="+", choices=CATALOG)
    parser.add_argument("--output-dir", type=Path, default=Path("data/asr/models"))
    args = parser.parse_args()
    if args.list:
        print(json.dumps(CATALOG, ensure_ascii=False, indent=2))
        return
    if not args.models:
        parser.error("select models explicitly with --models (the full catalog is about 14 GB)")
    output = args.output_dir.resolve()
    os.environ.setdefault("HF_HOME", str(output.parent / "hf-cache"))
    from huggingface_hub import snapshot_download
    for name in dict.fromkeys(args.models):
        specification = CATALOG[name]
        destination = output / name
        snapshot_download(specification["repo"], revision=specification["revision"],
                          local_dir=destination, allow_patterns=specification["files"],
                          max_workers=4, token=False)
        records = []
        for path in sorted(destination.rglob("*")):
            if not path.is_file() or ".cache" in path.parts or path.name == "download-provenance.json":
                continue
            with path.open("rb") as stream:
                sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append({"path": str(path.relative_to(destination)), "bytes": path.stat().st_size, "sha256": sha256})
        provenance = {**specification, "downloaded_files": records}
        (destination / "download-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps({"model": name, "directory": str(destination), "revision": specification["revision"]}), flush=True)


if __name__ == "__main__":
    main()
