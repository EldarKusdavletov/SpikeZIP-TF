#!/usr/bin/env python3
"""
Script to download all pre-trained models and weights for SpikeZIP-TF.
Downloads ANN models, QANN models, and verifies checksums.
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

# Model configurations
ANN_MODELS = {
    'vit-small-relu': {
        'url': 'https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-small-patch16-relu/resolve/main/vit-small-patch16-relu-82.34.pth',
        'filename': 'vit-small-patch16-relu-82.34.pth',
        'md5': '929f93b',
        'description': 'ViT-Small-ReLU pre-trained checkpoint'
    },
    'vit-base-relu': {
        'url': 'https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-base-patch16-relu/resolve/main/vit_base_patch16_ReLU_83.458.pth',
        'filename': 'vit_base_patch16_ReLU_83.458.pth',
        'md5': '8d49104',
        'description': 'ViT-Base-ReLU pre-trained checkpoint'
    },
    'vit-large-relu': {
        'url': 'https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-large-patch16-relu/resolve/main/vit-large-imagenet-relu-85.41.pth',
        'filename': 'vit-large-imagenet-relu-85.41.pth',
        'md5': '91bded0',
        'description': 'ViT-Large-ReLU pre-trained checkpoint'
    }
}

QANN_MODELS = {
    'vit-small-relu-q32': {
        'url': 'https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-small-patch16-relu-q32/resolve/main/vit-small-imagenet-relu-q32-81.59.pth',
        'filename': 'vit-small-imagenet-relu-q32-81.59.pth',
        'md5': '8207d3e',
        'description': 'ViT-Small-ReLU-Q32 pre-trained QANN checkpoint'
    },
    'vit-base-relu-q32': {
        'url': 'https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-base-patch16-relu-q32/resolve/main/vit-base-imagenet-relu-q32-82.83.pth',
        'filename': 'vit-base-imagenet-relu-q32-82.83.pth',
        'md5': '7edba1d',
        'description': 'ViT-Base-ReLU-Q32 pre-trained QANN checkpoint'
    },
    'vit-large-relu-q32': {
        'url': 'https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-large-patch16-relu-q32/resolve/main/vit-large-imagenet-relu-q32-83.86.pth',
        'filename': 'vit-large-imagenet-relu-q32-83.86.pth',
        'md5': 'd83936c',
        'description': 'ViT-Large-ReLU-Q32 pre-trained QANN checkpoint'
    }
}


def calculate_md5(filepath: str, chunk_size: int = 8192) -> str:
    """Calculate MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5_hash.update(chunk)
    # Return first 7 characters to match expected format from documentation
    # Note: This matches the partial MD5 provided in the model documentation
    return md5_hash.hexdigest()[:7]


def download_file(url: str, destination: str, description: str = "") -> bool:
    """Download a file from URL to destination with progress reporting."""
    try:
        print(f"\nDownloading: {description}")
        print(f"  URL: {url}")
        print(f"  Destination: {destination}")
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print("\n  ✓ Download complete")
        return True
    except Exception as e:
        print(f"\n  ✗ Error downloading: {e}")
        return False


def verify_checksum(filepath: str, expected_md5: str) -> bool:
    """Verify file checksum."""
    actual_md5 = calculate_md5(filepath)
    if actual_md5 == expected_md5:
        print(f"  ✓ Checksum verified: {actual_md5}")
        return True
    else:
        print(f"  ✗ Checksum mismatch: expected {expected_md5}, got {actual_md5}")
        return False


def download_models(models: Dict, output_dir: str, skip_existing: bool = True) -> Tuple[int, int]:
    """Download all models in the dictionary."""
    success_count = 0
    total_count = len(models)
    
    for model_name, model_info in models.items():
        filepath = os.path.join(output_dir, model_info['filename'])
        
        # Check if file already exists
        if skip_existing and os.path.exists(filepath):
            print(f"\n{model_name}: File already exists")
            if verify_checksum(filepath, model_info['md5']):
                success_count += 1
                continue
            else:
                print("  Checksum verification failed. Re-downloading...")
                os.remove(filepath)
        
        # Download the file
        if download_file(model_info['url'], filepath, model_info['description']):
            # Verify checksum
            if verify_checksum(filepath, model_info['md5']):
                success_count += 1
            else:
                print("  Warning: File downloaded but checksum verification failed")
    
    return success_count, total_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download SpikeZIP-TF pre-trained models')
    parser.add_argument('--output-dir', type=str, default='./pretrained_models',
                        help='Directory to save downloaded models (default: ./pretrained_models)')
    parser.add_argument('--models', type=str, choices=['ann', 'qann', 'all'], default='all',
                        help='Which models to download: ann, qann, or all (default: all)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if files exist')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SpikeZIP-TF Model Downloader")
    print("=" * 70)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    total_success = 0
    total_count = 0
    
    # Download ANN models
    if args.models in ['ann', 'all']:
        print("\n" + "=" * 70)
        print("Downloading ANN (Pre-trained) Models")
        print("=" * 70)
        success, count = download_models(ANN_MODELS, output_dir, skip_existing=not args.force)
        total_success += success
        total_count += count
    
    # Download QANN models
    if args.models in ['qann', 'all']:
        print("\n" + "=" * 70)
        print("Downloading QANN (Quantized) Models")
        print("=" * 70)
        success, count = download_models(QANN_MODELS, output_dir, skip_existing=not args.force)
        total_success += success
        total_count += count
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Successfully downloaded and verified: {total_success}/{total_count} models")
    print(f"Models saved to: {os.path.abspath(output_dir)}")
    
    if total_success == total_count:
        print("\n✓ All models downloaded successfully!")
        return 0
    else:
        print(f"\n✗ {total_count - total_success} model(s) failed to download or verify")
        return 1


if __name__ == '__main__':
    sys.exit(main())
