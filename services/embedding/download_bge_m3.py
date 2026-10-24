import os
import argparse
from huggingface_hub import snapshot_download

# Model-VM layout (same home tree as voice/paddle models)
DEFAULT_TARGET_DIR = "/opt/models/bge-m3"

def download_model(target_dir: str):
    model_id = "BAAI/bge-m3"
    
    print(f"Starting download of {model_id}...")
    print(f"Target directory: {target_dir}")
    
    # Ensure the directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Download the snapshot
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=target_dir
        )
        
        print("\nDownload complete!")
        print(f"Model successfully saved to: {downloaded_path}")
        print("\nNext steps:")
        print(f"1. Make sure your .env file or LOCAL_EMBEDDING_MODEL_PATH points to: {downloaded_path}")
        print("2. Restart your embedding API service.")
        
    except Exception as e:
        print(f"\nAn error occurred during download: {e}")
        print("Please ensure you have an active internet connection and sufficient disk space (~2.5GB).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BAAI/bge-m3 embedding model.")
    parser.add_argument(
        "--target_dir", 
        type=str, 
        default=DEFAULT_TARGET_DIR,
        help=f"Local directory to save the model (default: {DEFAULT_TARGET_DIR})"
    )
    args = parser.parse_args()
    
    download_model(args.target_dir)
