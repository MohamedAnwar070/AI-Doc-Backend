"""
Download the olmOCR model (allenai/olmOCR-2-7B-1025) to a local directory.

Usage:
    python download_olmocr_model.py <target_directory>

Example:
    python download_olmocr_model.py E:\models\olmocr
    python download_olmocr_model.py C:\AI-Models\olmocr

The model is ~10 GB. Make sure the target drive has enough free space.
After download, set in .env:
    OLM_OCR_PATH=E:\models\olmocr   (or wherever you downloaded)
"""
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_olmocr_model.py <target_directory>")
        print("Example: python download_olmocr_model.py E:\\models\\olmocr")
        sys.exit(1)

    target_dir = sys.argv[1]
    os.makedirs(target_dir, exist_ok=True)

    # Check free disk space
    import shutil
    total, used, free = shutil.disk_usage(os.path.splitdrive(target_dir)[0] or target_dir)
    free_gb = free / (1024 ** 3)
    print(f"Target: {target_dir}")
    print(f"Free disk space: {free_gb:.1f} GB")

    if free_gb < 12:
        print(f"WARNING: You need at least 12 GB free. You have {free_gb:.1f} GB.")
        answer = input("Continue anyway? (y/n): ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(1)

    model_id = "allenai/olmOCR-2-7B-1025"

    print(f"\nDownloading model: {model_id}")
    print(f"Saving to: {target_dir}")
    print("This will take a while (~10 GB download)...\n")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        local_dir=os.path.join(target_dir, "olmOCR-2-7B-1025"),
        local_dir_use_symlinks=False,
    )

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Model saved at: {os.path.join(target_dir, 'olmOCR-2-7B-1025')}")
    print(f"\nNow add this to your .env file:")
    print(f"  OLM_OCR_PATH={target_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
