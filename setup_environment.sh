#!/bin/bash
# Setup script for SpikeZIP-TF environment
# This script installs dependencies and patches timm 0.3.2 to work with modern PyTorch

set -e

# Virtual environment name
VENV_NAME=".venv"

# Check if virtual environment exists
if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' already exists. Activating..."
    source "$VENV_NAME/bin/activate"
else
    echo "Creating virtual environment '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
    echo "Activating virtual environment..."
    source "$VENV_NAME/bin/activate"
fi

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Patching timm 0.3.2 to work with modern PyTorch..."

# Use Python to safely patch the timm library
python << 'PYTHON_SCRIPT'
import sys
import os

try:
    import timm
except ImportError:
    print("Error: timm is not installed. Please run 'pip install -r requirements.txt' first.")
    sys.exit(1)

# Find the timm helpers.py file
helpers_path = os.path.join(os.path.dirname(timm.__file__), 'models/layers/helpers.py')

if not os.path.exists(helpers_path):
    print(f"Error: Could not find timm helpers.py at {helpers_path}")
    sys.exit(1)

# Read the file
with open(helpers_path, 'r') as f:
    content = f.read()

# Check if patch is already applied
if 'import collections.abc as container_abcs' in content:
    print("Patch already applied to timm.")
    sys.exit(0)

# Apply the patch
old_import = 'from torch._six import container_abcs'
new_import = 'import collections.abc as container_abcs'

if old_import not in content:
    print("Warning: Expected import not found in timm helpers.py")
    print("The library may already be compatible or may have a different version.")
    sys.exit(0)

content = content.replace(old_import, new_import)

# Write back the patched content
with open(helpers_path, 'w') as f:
    f.write(content)

print(f"Successfully patched {helpers_path}")
print("Note: This modifies the installed timm package.")
PYTHON_SCRIPT

echo ""
echo "Setup complete! You can now run the training scripts."
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Example usage:"
echo "  python main_finetune.py --data_path /path/to/imagenet --model vit_small_patch16 ..."
