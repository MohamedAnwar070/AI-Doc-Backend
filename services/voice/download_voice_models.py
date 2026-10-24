#!/usr/bin/env python3
"""
Download the 3 voice engines to a local directory.

Linux-native. Populates ``/opt/models/voice/`` with:

    faster-whisper-large-v3/    ~3.0 GB   Systran/faster-whisper-large-v3
    piper/                      ~60 MB    rhasspy/piper-voices (en + ta voices)
    silero-vad/                 ~2 MB     snakers4/silero-vad (ONNX)

Usage:
    python download_voice_models.py                                # uses default target
    python download_voice_models.py /custom/target/dir             # custom target

Idempotent: re-running skips directories that already contain the
expected marker file.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Callable, List

# ── Defaults ────────────────────────────────────────────────────────────────

MODEL_ROOTS_BY_OS = {
    "Darwin": "/Users/gsmmgsm/gsm/ai-ml/ai-orginal/ai-models/voice-models",
    "Linux": "/opt/models/voice-models",
    "Windows": r"D:\models\voice-models",
}
VOICE_MODEL_ROOT = os.environ.get(
    "VOICE_MODEL_ROOT",
    MODEL_ROOTS_BY_OS.get(platform.system(), "/models"),
)

# ── Catalogue (4 engines for VOICE_PLAN.md) ─────────────────────────────────

# HuggingFace repos (full repo snapshot)
HF_REPOS = {
    "faster-whisper-large-v3": {
        "repo": "Systran/faster-whisper-large-v3",
        "size_gb": 3.0,
        "marker": "model.bin",
    },
    "xtts-v2": {
        "repo": "coqui/XTTS-v2",
        "size_gb": 2.0,
        "marker": "model.pth",
    },
}

# Piper voices — pulled directly from the rhasspy/piper-voices HF mirror.
# Each entry is (file_stem, onnx_url, json_url).
PIPER_VOICES = [
    # English
    (
        "en_US-amy-low",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json",
    ),
    (
        "en_US-lessac-low",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx.json",
    ),
    # Tamil (VITS medium quality)
    (
        "ta_IN-vits-medium",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ta/ta_IN/vits/medium/ta_IN-vits-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ta/ta_IN/vits/medium/ta_IN-vits-medium.onnx.json",
    ),
]

# Tiny direct downloads (no HF repo)
EXTRA_DOWNLOADS = {
    "silero-vad": {
        "url": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        "filename": "silero_vad.onnx",
        "size_mb": 2,
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────

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


def _hf_snapshot_download(repo: str, local_dir: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        snapshot_download(
            repo_id=repo,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            max_workers=4,
        )
        return True
    except ImportError:
        print("  huggingface_hub not installed; attempting pip install\u2026")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
        from huggingface_hub import snapshot_download  # type: ignore
        snapshot_download(
            repo_id=repo,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            max_workers=4,
        )
        return True


def _download_url(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    try:
        with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
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
                    print(f"\r  {read // 1024} / {total // 1024} KB ({pct:.1f}%)", end="", flush=True)
            print()
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        if dest.exists():
            dest.unlink()
        return False


# ── Per-engine downloaders ──────────────────────────────────────────────────

def download_whisper(target: Path) -> None:
    cfg = HF_REPOS["faster-whisper-large-v3"]
    dest = target / "faster-whisper-large-v3"
    if _is_already_downloaded(dest, cfg["marker"]):
        print(f"[skip] {cfg['repo']} (already in {dest})")
        return
    print(f"[get ] {cfg['repo']}  (~{cfg['size_gb']} GB)")
    _hf_snapshot_download(cfg["repo"], dest)
    print(f"[done] Whisper Large v3 -> {dest}")


def download_piper_voices(target: Path) -> None:
    dest = target / "piper"
    have_onnx = list(dest.glob("*.onnx")) if dest.exists() else []
    if have_onnx and any((p.with_suffix(p.suffix + ".json")).exists() for p in have_onnx):
        print(f"[skip] Piper voices (already in {dest}, {len(have_onnx)} onnx files)")
        return
    print(f"[get ] Piper voices  ({len(PIPER_VOICES)} voices, ~{len(PIPER_VOICES) * 30} MB)")
    dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    for stem, onnx_url, json_url in PIPER_VOICES:
        onnx_path = dest / f"{stem}.onnx"
        json_path = dest / f"{stem}.onnx.json"
        if onnx_path.exists() and json_path.exists():
            print(f"  [skip] {stem}")
            ok += 1
            continue
        if _download_url(onnx_url, onnx_path):
            _download_url(json_url, json_path)
            ok += 1
    print(f"[done] Piper voices -> {dest} ({ok}/{len(PIPER_VOICES)} ok)")


def download_silero_vad(target: Path) -> None:
    cfg = EXTRA_DOWNLOADS["silero-vad"]
    dest = target / "silero-vad"
    out_file = dest / cfg["filename"]
    if out_file.exists():
        print(f"[skip] Silero VAD (already in {dest})")
        return
    print(f"[get ] Silero VAD ONNX  (~{cfg['size_mb']} MB)")
    _download_url(cfg["url"], out_file)
    print(f"[done] Silero VAD -> {out_file}")


def download_xtts(target: Path) -> None:
    cfg = HF_REPOS["xtts-v2"]
    dest = target / "xtts-v2"
    if _is_already_downloaded(dest, cfg["marker"]):
        print(f"[skip] {cfg['repo']} (already in {dest})")
        return
    print(f"[get ] {cfg['repo']}  (~{cfg['size_gb']} GB)")
    _hf_snapshot_download(cfg["repo"], dest)
    print(f"[done] XTTS-v2 -> {dest}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the 4 voice engines (Whisper + Piper + Silero VAD + XTTS-v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=VOICE_MODEL_ROOT,
        help=f"Target directory (default: {VOICE_MODEL_ROOT}).",
    )
    parser.add_argument(
        "--only",
        choices=["whisper", "piper", "silero_vad", "xtts", "all"],
        default="all",
        help="Download just one engine instead of all (default: all).",
    )
    args = parser.parse_args()

    target = Path(args.target_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Voice Engine Downloader (Linux-native, 2026-06)")
    print(" Engines: Whisper large-v3 + Piper (en + ta voices) + Silero VAD + XTTS-v2")
    print("=" * 70)
    print(f" Target: {target}")

    # Whisper 3.0 GB + Piper ~120 MB + Silero 2 MB + XTTS-v2 2.0 GB ~ 5.1 GB
    total_gb = 5.2
    _check_disk_space(target, total_gb)

    tasks: List[Callable[[Path], None]] = []
    if args.only in ("all", "whisper"):
        tasks.append(download_whisper)
    if args.only in ("all", "piper"):
        tasks.append(download_piper_voices)
    if args.only in ("all", "silero_vad"):
        tasks.append(download_silero_vad)
    if args.only in ("all", "xtts"):
        tasks.append(download_xtts)

    print(f" Running {len(tasks)} download task(s)\n")
    for task in tasks:
        try:
            task(target)
        except KeyboardInterrupt:
            print("\nAborted.")
            return 130
        except Exception as exc:
            print(f"  ERROR: {exc}")
            print("  (continuing with next task)\n")

    print("\n" + "=" * 70)
    print(" Voice engine download complete.")
    print(" Final layout:")
    for entry in sorted(target.iterdir()):
        if entry.is_dir():
            sub_count = sum(1 for _ in entry.rglob("*") if _.is_file())
            print(f"   {entry.name:<35} ({sub_count} files)")
    print()
    print(" Next steps:")
    print("   1. Start the voice microservice:")
    print(f"        python voice_api.py")
    print("   2. Verify /health returns 200:")
    print("        curl http://localhost:8003/health")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
