#!/bin/bash
# Setup script for SpikeZIP-TF environment
# This script installs dependencies and patches timm 0.3.2 to work with modern PyTorch

set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Patching timm 0.3.2 to work with modern PyTorch..."
# Find the timm helpers.py file
TIMM_HELPERS=$(python -c "import timm; import os; print(os.path.join(os.path.dirname(timm.__file__), 'models/layers/helpers.py'))")

# Patch the torch._six import to use collections.abc instead
sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/g' "$TIMM_HELPERS"

echo "Setup complete! You can now run the training scripts."
echo ""
echo "Example usage:"
echo "  python main_finetune.py --data_path /path/to/imagenet --model vit_small_patch16 ..."
