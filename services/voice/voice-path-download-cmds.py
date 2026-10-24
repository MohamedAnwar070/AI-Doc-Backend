#!/usr/bin/env python3
"""
voice-path-download-cmds.py

One-shot, self-contained download script for the voice stack.

Populates ``/opt/models/voice/`` with the three engines
needed by ``voice_api.py``:

    faster-whisper-large-v3/    ~3.0 GB   Systran/faster-whisper-large-v3
    piper/                      ~180 MB   en + ta voices from rhasspy/piper-voices
    silero-vad/                 ~2 MB     snakers4/silero-vad (ONNX)

Usage (just run it):

    python voice-path-download-cmds.py

    # or, after chmod +x:
    ./voice-path-download-cmds.py

The script is idempotent: re-running it skips any engine whose
expected marker file is already present. A failed partial download
gets cleaned up automatically on the next run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable, List, Tuple

# ── Hardcoded target path ────────────────────────────────────────────────────
# Per VOICE_PLAN.md — single source of truth for the model root.
TARGET = Path("/opt/models/voice")

# ── Catalogue ────────────────────────────────────────────────────────────────
HF_REPOS = {
    "faster-whisper-large-v3": {
        "repo": "Systran/faster-whisper-large-v3",
        "size_gb": 3.0,
        "marker": "model.bin",
    },
}

# (file_stem, onnx_url, json_url)
PIPER_VOICES: List[Tuple[str, str, str]] = [
    (
        "en_US-amy-low",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json",
    ),
    (
        "ta_IN-vits-medium",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ta/ta_IN/vits/medium/ta_IN-vits-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ta/ta_IN/vits/medium/ta_IN-vits-medium.onnx.json",
    ),
]

EXTRA_DOWNLOADS = {
    "silero-vad": {
        "url": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        "filename": "silero_vad.onnx",
        "size_mb": 2,
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _hr() -> str:
    return "=" * 72


def _ensure_hf_hub() -> None:
    """Make sure huggingface_hub is importable; pip-install if missing."""
    try:
        import huggingface_hub  # noqa: F401
        return
    except ImportError:
        print("  huggingface_hub not found; installing\u2026")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])


def _check_disk_space(target: Path, needed_gb: float) -> None:
    stat = shutil.disk_usage(target)
    free_gb = stat.free / (1024 ** 3)
    print(f"  Free disk: {free_gb:.1f} GB, need ~{needed_gb:.1f} GB")
    if free_gb < needed_gb + 1.0:
        print(f"  WARNING: less than {needed_gb + 1:.1f} GB free; download may fail")


def _is_already_downloaded(target: Path, marker: str) -> bool:
    if not target.exists():
        return False
    if marker and (target / marker).exists():
        return True
    snapshots = target / "snapshots"
    if snapshots.is_dir():
        for snap in snapshots.iterdir():
            if (snap / marker).exists():
                return True
    return False


def _download_url(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET  {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            read = 0
            chunk_size = 64 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if total:
                    pct = 100 * read / total
                    mb_read = read / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    print(f"\r      {mb_read:6.1f} / {mb_total:6.1f} MB ({pct:5.1f}%)", end="", flush=True)
            print()
        return True
    except Exception as exc:
        print(f"  FAIL {exc}")
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return False


def _download_hf_repo(repo_id: str, local_dir: Path) -> bool:
    _ensure_hf_hub()
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            max_workers=4,
        )
        return True
    except Exception as exc:
        print(f"  FAIL {exc}")
        return False


# ── Per-engine downloaders ──────────────────────────────────────────────────

def download_whisper(target: Path) -> bool:
    cfg = HF_REPOS["faster-whisper-large-v3"]
    dest = target / "faster-whisper-large-v3"
    if _is_already_downloaded(dest, cfg["marker"]):
        print(f"  [skip] {cfg['repo']} (already in {dest})")
        return True
    print(f"  [get ] {cfg['repo']}  (~{cfg['size_gb']} GB)  -> {dest}")
    return _download_hf_repo(cfg["repo"], dest)


def download_piper(target: Path) -> bool:
    dest = target / "piper"
    have_onnx = list(dest.glob("*.onnx")) if dest.exists() else []
    if have_onnx and any((p.with_suffix(p.suffix + ".json")).exists() for p in have_onnx):
        print(f"  [skip] Piper voices (already in {dest}, {len(have_onnx)} onnx files)")
        return True
    print(f"  [get ] Piper voices  ({len(PIPER_VOICES)} voices, ~{len(PIPER_VOICES) * 60} MB)  -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    for stem, onnx_url, json_url in PIPER_VOICES:
        onnx_path = dest / f"{stem}.onnx"
        json_path = dest / f"{stem}.onnx.json"
        if onnx_path.exists() and json_path.exists():
            print(f"      [skip] {stem}")
            ok += 1
            continue
        if _download_url(onnx_url, onnx_path):
            _download_url(json_url, json_path)
            ok += 1
    return ok == len(PIPER_VOICES)


def download_silero(target: Path) -> bool:
    cfg = EXTRA_DOWNLOADS["silero-vad"]
    dest = target / "silero-vad"
    out_file = dest / cfg["filename"]
    if out_file.exists():
        print(f"  [skip] Silero VAD (already in {dest})")
        return True
    print(f"  [get ] Silero VAD ONNX  (~{cfg['size_mb']} MB)  -> {out_file}")
    return _download_url(cfg["url"], out_file)


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> int:
    print(_hr())
    print(" Voice Engine Downloader")
    print(f" Target: {TARGET}")
    print(_hr())

    TARGET.mkdir(parents=True, exist_ok=True)
    _check_disk_space(TARGET, needed_gb=3.2)

    tasks: List[Tuple[str, Callable[[Path], bool]]] = [
        ("Whisper large-v3", download_whisper),
        ("Piper voices (en + ta)", download_piper),
        ("Silero VAD v4", download_silero),
    ]

    failures: List[str] = []
    for name, fn in tasks:
        print(f"\n[{name}]")
        try:
            ok = fn(TARGET)
            if not ok:
                failures.append(name)
        except KeyboardInterrupt:
            print("\nAborted.")
            return 130
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failures.append(name)

    print()
    print(_hr())
    print(" Final layout:")
    for entry in sorted(TARGET.iterdir()):
        if entry.is_dir():
            sub_count = sum(1 for _ in entry.rglob("*") if _.is_file())
            print(f"   {entry.name:<35} ({sub_count} files)")
    print()

    if failures:
        print(f"  Some engines failed: {failures}")
        print(f"  Re-run this script to retry the failed engines.")
        return 1

    print(" All 3 engines downloaded successfully.")
    print()
    print(" Next step:")
    print("   cd /opt/ai-doc-retrieval-backend")
    print("   .venv/bin/python voice_api.py")
    print()
    print(" Then verify:")
    print("   curl -s http://localhost:8003/health | python3 -m json.tool")
    print(_hr())
    return 0


if __name__ == "__main__":
    sys.exit(main())
