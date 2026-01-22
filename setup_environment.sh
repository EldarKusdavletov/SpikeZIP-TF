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

# Patch timm immediately after installation, before trying to import it
TIMM_PATH=$("$VENV_NAME/bin/python" -c "import sys; from pathlib import Path; print(Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages' / 'timm' / 'models' / 'layers' / 'helpers.py')")

if [ -f "$TIMM_PATH" ]; then
    echo "Found timm helpers.py at: $TIMM_PATH"
    
    # Check if patch is already applied
    if grep -q "import collections.abc as container_abcs" "$TIMM_PATH"; then
        echo "Patch already applied to timm."
    elif grep -q "from torch._six import container_abcs" "$TIMM_PATH"; then
        echo "Applying patch..."
        sed -i.bak 's/from torch\._six import container_abcs/import collections.abc as container_abcs/g' "$TIMM_PATH"
        echo "Successfully patched $TIMM_PATH"
        echo "Note: This modifies the installed timm package."
    else
        echo "Warning: Expected import not found in timm helpers.py"
        echo "The library may already be compatible or may have a different version."
    fi
else
    echo "Error: Could not find timm helpers.py at $TIMM_PATH"
    exit 1
fi

# Verify the patch worked by trying to import timm
"$VENV_NAME/bin/python" << 'PYTHON_SCRIPT'
import sys
try:
    import timm
    print("✓ Successfully verified timm import after patching")
except Exception as e:
    print(f"✗ Failed to import timm after patching: {e}")
    sys.exit(1)
PYTHON_SCRIPT

echo ""
echo "Setup complete! You can now run the training scripts."
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Example usage:"
echo "  python main_finetune.py --data_path /path/to/imagenet --model vit_small_patch16 ..."
