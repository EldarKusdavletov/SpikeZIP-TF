<p align="center">
<img src="spikezip_logo.png" alt="spikezip_logo" width="220" align="center">
</p>

<div align="center"><h1>&nbsp;SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN</h1></div>


<p align="center">
| <a href="http://arxiv.org/"><b>Paper</b></a> | <a href="http://arxiv.org/"><b>Blog</b></a> |
</p>


<p align="center">
  <a href="https://opensource.org/license/mulanpsl-2-0">
    <img src="https://img.shields.io/badge/License-MuLan_PSL_2.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/">
    <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintenance">
  </a>
  <a href="https://github.com/">
    <img src="https://img.shields.io/badge/Contributions-welcome-brightgreen.svg?style=flat" alt="Contributions welcome">
  </a>
</p>


## Contents
- [News](#news)
- [Introduction](#introduction)
- [Conversion Framework](#conversion-framework) :fire: **NEW**
- [Usage](#Usage)
  - [Train](#Train)
  - [Conversion](#Conversion)
  - [Evaluation](#Evaluation) 

## News

- [2024/6] Code of SpikeZip-TF is released!
- [2024/12] Code of SpikeZip-TF in NLU tasks is released! You can view the code by switching to the NLU_tasks branch !!
- :fire: **[NEW]** Comprehensive conversion framework added! Supports ANN→QANN→SNN conversion in all variations, plus MLIR and ONNX export! 

## Introduction

![image](https://github.com/Intelligent-Computing-Research-Group/SpikeZIP_transformer/assets/74498528/91609adb-56f2-49e1-92fe-9596b38cb9f4)


This is the official project repository for the following paper. If you find this repository helpful, Please kindly cite:
```
@inproceedings{
spikeziptf2024,
title={SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN},
author={You, Kang= and Xu, Zekai= and Nie, Chen and Deng, Zhijie and Wang, Xiang and Guo, Qinghai and He, Zhezhi},
booktitle={Forty-first International Conference on Machine Learning (ICML)},
year={2024}
}
```
:fire: =: indicates equal contribution.

## Usage

### Setup Environment

Before running any training scripts, you need to set up the Python environment with the required dependencies:

```bash
# Option 1: Automated setup with virtual environment (recommended)
./setup_environment.sh

# This creates/activates a virtual environment and installs all dependencies
# To activate the environment later: source .venv/bin/activate

# Option 2: Manual setup
pip install -r requirements.txt
# Then patch timm 0.3.2 to work with modern PyTorch
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

**Dependencies:**
- Python 3.8+
- PyTorch >= 2.6.0 (for security - fixes critical vulnerabilities)
- torchvision >= 0.20.0
- timm 0.3.2 (with compatibility patch)
- numpy < 2.0.0
- Other dependencies listed in requirements.txt

### Train

Train the Quantized-ANN with pretrain model.

The following table provides the pre-trained checkpoints used in the paper:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Small-ReLU</th>
<th valign="bottom">ViT-Base-ReLU</th>
<th valign="bottom">ViT-Large-ReLU</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-small-patch16-relu/resolve/main/vit-small-patch16-relu-82.34.pth">download</a></td>
<td align="center"><a href="https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-base-patch16-relu/resolve/main/vit_base_patch16_ReLU_83.458.pth">download</a></td>
<td align="center"><a href="https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-large-patch16-relu/resolve/main/vit-large-imagenet-relu-85.41.pth">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>929f93b</tt></td>
<td align="center"><tt>8d49104</tt></td>
<td align="center"><tt>91bded0</tt></td>
</tr>
</tbody></table>

Prepare the ImageNet dataset and run the scripts below:
```
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port='29500' main_finetune_distill.py \
    --accum_iter 4 \
    --batch_size 64 \
    --model vit_small_patch16 \
    --model_teacher vit_small_patch16 \
    --finetune checkpoint-path \
    --pretrain_teacher checkpoint-path \
    --epochs 100 \
    --blr 1.5e-4 --layer_decay 0.65 --warmup_epochs 0 \
    --weight_decay 0.05 --drop_path 0.1 --drop_rate 0.0 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path dataset-path --output_dir output_path --log_dir log_path \
    --mode "QANN_QAT" --level 16 --act_layer relu --act_layer_teacher relu --temp 2.0 --wandb --print_freq 200 --define_params --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5
```

### Conversion

Convert your QANN model to SNN.

The following table provides the pre-trained QANN used in the paper:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Small-ReLU-Q32</th>
<th valign="bottom">ViT-Base-ReLU-Q32</th>
<th valign="bottom">ViT-Large-ReLU-Q32</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-small-patch16-relu-q32/resolve/main/vit-small-imagenet-relu-q32-81.59.pth">download</a></td>
<td align="center"><a href="https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-base-patch16-relu-q32/resolve/main/vit-base-imagenet-relu-q32-82.83.pth">download</a></td>
<td align="center"><a href="https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-large-patch16-relu-q32/resolve/main/vit-large-imagenet-relu-q32-83.86.pth">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>8207d3e</tt></td>
<td align="center"><tt>7edba1d</tt></td>
<td align="center"><tt>d83936c</tt></td>
</tr>
</tbody></table>


Prepare the ImageNet dataset and run the scripts below:
```
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' main_finetune.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_small_patch16 \
    --finetune QANN-checkpoint-path \
    --resume QANN-checkpoint-path \
    --epochs 100 \
    --blr 3.536e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --drop_rate 0.0 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path dataset-path --output_dir output_path --log_dir log_path \
    --mode "SNN" --act_layer relu --eval --ratio 0.5 --time_step 64 --encoding_type analog --level 16 --weight_quantization_bit 32 --define_params --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5
```

### Evaluation
Evaluate the energy of your SNN model

By using QANN trained by yourself or provided in pretrained QANN table above, run the the scripts below to evaluate the energy of your SNN model:
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port='29501' main_finetune.py \
    --accum_iter 4 \
    --batch_size 4 \
    --model vit_small_patch16 \
    --finetune /data1/vit-small-imagenet-relu-q16-80.45.pth \
    --resume /data1/vit-small-imagenet-relu-q16-80.45.pth \
    --epochs 100 \
    --blr 3.536e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --drop_rate 0.0 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data1/ImageNet/ --output_dir /home/kang_you/SpikeZIP_transformer/output/ --log_dir /home/kang_you/SpikeZIP_transformer/output \
    --mode "SNN" --act_layer relu --eval --energy_eval --time_step 32 --encoding_type rate --level 16 --weight_quantization_bit 32 --define_params --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5
```

## Conversion Framework

We now provide a comprehensive conversion framework for easy model conversion and export! :fire:

### Quick Start with Conversion Framework

```bash
# 1. Download all pre-trained models
python download_models.py --output-dir ./pretrained_models --models all

# 2. Convert ANN to all QANN and SNN variations
python convert_models.py \
    --input ./pretrained_models/vit-small-patch16-relu-82.34.pth \
    --output-dir ./converted_models \
    --model-name vit_small_patch16 \
    --all-variations

# 3. Export SNN to MLIR
python snn_to_mlir.py \
    --input ./converted_models/vit_small_patch16_snn_q32_analog_t64.pth \
    --output ./mlir_exports/vit_small_snn.mlir

# 4. Export SNN to ONNX
python snn_to_onnx.py \
    --input ./converted_models/vit_small_patch16_snn_q32_analog_t64.pth \
    --output ./onnx_exports/vit_small_snn.onnx

# OR run the complete pipeline in one command:
python run_conversion_pipeline.py --model-type vit-small
```

### Features

The conversion framework provides:

- **Automated Model Download**: Download all pre-trained ANN and QANN models with checksum verification
- **Multi-Variation Conversion**: Convert to all quantization levels (Q8, Q16, Q32, Q64) and SNN configurations
- **MLIR Export**: Translate SNN models to MLIR linalg dialect for compiler optimization
- **ONNX Export**: Export SNN models to ONNX format with custom SNN operators
- **Complete Pipeline**: One-command execution of the entire conversion workflow

### Conversion Variations

- **QANN**: 4 quantization levels (8, 16, 32, 64 bits)
- **SNN**: 6 variations per QANN
  - Time steps: 32, 64, 128
  - Encodings: analog, rate
- **Total**: 24 SNN models per ANN model (4 QANN × 6 SNN variations)

### Documentation

For complete documentation, see [CONVERSION_FRAMEWORK.md](CONVERSION_FRAMEWORK.md)

For usage examples, run:
```bash
python examples_conversion_framework.py
```

For verification, run:
```bash
python verify_conversion_framework.py
```


