#!/usr/bin/env python3
"""Download the recommended local GGUF chat model for ai-doc-retrieval.

Default target:
    D:\models\llm\qwen3-14b\
    (mac path: /Users/gsmmgsm/gsm/ai-ml/ai-orginal/ai-models/llm/qwen3-14b/)

Default model:
    Qwen/Qwen3-14B-GGUF / Qwen3-14B-Q4_K_M.gguf

The downloaded file is intended for a local llama.cpp / LM Studio /
compatible OpenAI-style server, then selected in `.env` with:

    AI_MODEL_RUNTIME=local
    AI_MODEL_LOCAL_URL=http://127.0.0.1:8081/v1
    AI_MODEL_LOCAL_MODEL=Qwen3-14B-Q4_K_M.gguf
"""
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ROOT = Path(r"D:\models")
# DEFAULT_ROOT = Path("/Users/gsmmgsm/gsm/ai-ml/ai-orginal/ai-models")

DEFAULT_REPO_ID = "Qwen/Qwen3-14B-GGUF"
DEFAULT_FILENAME = "Qwen3-14B-Q4_K_M.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Qwen3 14B Q4_K_M GGUF for local Mac inference."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Model root directory. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help=f"GGUF filename in the repo. Default: {DEFAULT_FILENAME}",
    )
    parser.add_argument(
        "--subdir",
        default="llm/qwen3-14b",
        help="Subdirectory under --root where the file should be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = args.root / args.subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface-hub. Install it with:\n"
            "  python -m pip install huggingface-hub"
        ) from exc

    print("Downloading local LLM model")
    print(f"  repo     : {args.repo_id}")
    print(f"  file     : {args.filename}")
    print(f"  target   : {target_dir}")

    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print("\nDone")
    print(f"  model path: {path}")
    print("\nStart a local OpenAI-compatible server, for example:")
    print(
        "  llama-server "
        f"-m {Path(path)} "
        "--host 127.0.0.1 --port 8081 "
        "-c 8192 -ngl 99"
    )


if __name__ == "__main__":
    main()
