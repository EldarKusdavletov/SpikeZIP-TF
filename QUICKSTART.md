# Quick Start Guide for SpikeZIP-TF

This guide will help you get started with SpikeZIP-TF.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- ImageNet dataset (for training/evaluation)

**Important Security Note**: This setup uses PyTorch >= 2.6.0 to address critical security vulnerabilities in earlier versions.

## Installation

### Option 1: Automated Setup (Recommended)

```bash
./setup_environment.sh
```

This will:
- Create a virtual environment (venv) if it doesn't exist
- Activate the virtual environment
- Install all required Python packages
- Apply compatibility patches automatically
- Verify the installation

**To activate the environment later:**
```bash
source venv/bin/activate
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Apply compatibility patch for timm
python -c "
import timm, os
helpers_path = os.path.join(os.path.dirname(timm.__file__), 'models/layers/helpers.py')
with open(helpers_path, 'r') as f:
    content = f.read()
content = content.replace('from torch._six import container_abcs', 'import collections.abc as container_abcs')
with open(helpers_path, 'w') as f:
    f.write(content)
print('Patched timm successfully')
"
```

## Verification

Run this command to verify your installation:

```bash
python -c "
import torch, timm, models_vit, models_mae
print('✅ SpikeZIP-TF is ready!')
print(f'PyTorch: {torch.__version__}')
print(f'timm: {timm.__version__}')
"
```

## Usage Examples

### 1. Pre-training

```bash
python main_pretrain.py \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --epochs 400 \
    --data_path /path/to/imagenet \
    --output_dir ./output_pretrain
```

### 2. Fine-tuning with Distillation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=29500 \
    main_finetune_distill.py \
    --batch_size 64 \
    --model vit_small_patch16 \
    --model_teacher vit_small_patch16 \
    --finetune /path/to/checkpoint.pth \
    --pretrain_teacher /path/to/teacher_checkpoint.pth \
    --data_path /path/to/imagenet \
    --output_dir ./output_finetune
```

### 3. SNN Conversion and Evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=29501 \
    main_finetune.py \
    --batch_size 32 \
    --model vit_small_patch16 \
    --finetune /path/to/qann_checkpoint.pth \
    --resume /path/to/qann_checkpoint.pth \
    --mode "SNN" \
    --eval \
    --time_step 64 \
    --data_path /path/to/imagenet \
    --output_dir ./output_snn
```

## Pre-trained Models

Download pre-trained checkpoints from:
- [ViT-Small-ReLU](https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-small-patch16-relu/resolve/main/vit-small-patch16-relu-82.34.pth)
- [ViT-Base-ReLU](https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-base-patch16-relu/resolve/main/vit_base_patch16_ReLU_83.458.pth)
- [ViT-Large-ReLU](https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-large-patch16-relu/resolve/main/vit-large-imagenet-relu-85.41.pth)

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch._six'"

**Solution**: Run `./setup_environment.sh` to apply the compatibility patch.

### Issue: NumPy version conflict

**Solution**: Ensure you have NumPy < 2.0.0 installed:
```bash
pip install "numpy<2.0.0"
```

### Issue: DVS dataset scripts not working

**Solution**: DVS/neuromorphic dataset support requires additional dependencies:
```bash
pip install spikingjelly>=0.0.0.0.14
```

## Support

For more information, see the full README.md or visit the paper:
- Conference: ICML 2024
- Title: "SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN"

## License

This project is licensed under the MuLan PSL 2.0 License.
