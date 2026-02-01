#!/usr/bin/env python3
"""
Download LingBot-World model weights from HuggingFace.

This script provides options for downloading different model variants
and handles authentication if needed.
"""

import argparse
import os
import sys
from pathlib import Path


def download_from_huggingface(model_id: str, local_dir: Path):
    """Download model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download
    
    print(f"Downloading {model_id} to {local_dir}...")
    print("This may take a while depending on your connection speed.")
    
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    print(f"✓ Downloaded to {local_dir}")


def download_from_modelscope(model_id: str, local_dir: Path):
    """Download model from ModelScope (alternative for China)."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("Installing modelscope...")
        os.system(f"{sys.executable} -m pip install modelscope")
        from modelscope import snapshot_download
    
    print(f"Downloading {model_id} from ModelScope to {local_dir}...")
    
    snapshot_download(
        model_id=model_id,
        local_dir=str(local_dir)
    )
    
    print(f"✓ Downloaded to {local_dir}")


MODELS = {
    "base-cam": {
        "hf": "robbyant/lingbot-world-base-cam",
        "ms": "Robbyant/lingbot-world-base-cam",
        "description": "Base model with camera pose control (480P & 720P)"
    },
    # Future models can be added here
    # "base-act": {...},
    # "fast": {...},
}


def main():
    parser = argparse.ArgumentParser(
        description="Download LingBot-World model weights"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="base-cam",
        help="Model variant to download (default: base-cam)"
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Download source (default: huggingface)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./lingbot-world-{model})"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable models:")
        print("-" * 60)
        for name, info in MODELS.items():
            print(f"  {name}: {info['description']}")
        print("-" * 60)
        return
    
    model_info = MODELS[args.model]
    
    if args.output_dir:
        local_dir = args.output_dir
    else:
        local_dir = Path(f"./lingbot-world-{args.model}")
    
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"Model directory {local_dir} already exists and is not empty.")
        response = input("Do you want to re-download? [y/N]: ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    local_dir.mkdir(parents=True, exist_ok=True)
    
    if args.source == "huggingface":
        download_from_huggingface(model_info["hf"], local_dir)
    else:
        download_from_modelscope(model_info["ms"], local_dir)
    
    print("\n✓ Model download complete!")
    print(f"  Model path: {local_dir.absolute()}")
    print("\nTo use this model, set MODEL_PATH in your .env file:")
    print(f"  MODEL_PATH={local_dir.absolute()}")


if __name__ == "__main__":
    main()
