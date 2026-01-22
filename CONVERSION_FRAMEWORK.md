# SpikeZIP-TF Conversion Framework

This directory contains a comprehensive framework for converting neural network models between different formats:
- **ANN** (Artificial Neural Networks) → **QANN** (Quantized ANN) → **SNN** (Spiking Neural Networks)
- **SNN** → **MLIR** (Multi-Level Intermediate Representation)
- **SNN** → **ONNX** (Open Neural Network Exchange)

## Overview

The conversion framework provides tools to:

1. **Download** all necessary pre-trained models and weights
2. **Convert** ANN to QANN and QANN to SNN in all possible variations
3. **Export** SNN models to MLIR (linalg dialect) and ONNX formats
4. **Validate** all conversions and exports

## Quick Start

### Installation

First, ensure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

### Complete Pipeline

Run the complete conversion pipeline with a single command:

```bash
python run_conversion_pipeline.py --model-type vit-small
```

This will:
1. Download all pre-trained models (ANN and QANN)
2. Convert ANN to all QANN variations (Q8, Q16, Q32, Q64)
3. Convert each QANN to all SNN variations (different time steps and encodings)
4. Export all SNN models to MLIR format
5. Export all SNN models to ONNX format

### Step-by-Step Usage

#### 1. Download Models

Download all pre-trained models:

```bash
python download_models.py --output-dir ./pretrained_models --models all
```

Options:
- `--models`: Choose `ann`, `qann`, or `all`
- `--force`: Force re-download even if files exist
- `--output-dir`: Directory to save models (default: `./pretrained_models`)

#### 2. Convert ANN to QANN and SNN

Convert a single model with specific parameters:

```bash
python convert_models.py \
    --input ./pretrained_models/vit-small-patch16-relu-82.34.pth \
    --output-dir ./converted_models \
    --model-name vit_small_patch16 \
    --level 32 \
    --time-step 64 \
    --encoding analog
```

Convert to all variations:

```bash
python convert_models.py \
    --input ./pretrained_models/vit-small-patch16-relu-82.34.pth \
    --output-dir ./converted_models \
    --model-name vit_small_patch16 \
    --all-variations
```

#### 3. Export to MLIR

Export an SNN model to MLIR linalg dialect:

```bash
python snn_to_mlir.py \
    --input ./converted_models/vit_small_patch16_snn_q32_analog_t64.pth \
    --output ./mlir_exports/vit_small_snn.mlir \
    --model-name vit_small_patch16
```

#### 4. Export to ONNX

Export an SNN model to ONNX format:

```bash
python snn_to_onnx.py \
    --input ./converted_models/vit_small_patch16_snn_q32_analog_t64.pth \
    --output ./onnx_exports/vit_small_snn.onnx \
    --model-name vit_small_patch16 \
    --time-steps 64 \
    --encoding analog
```

## Conversion Variations

### QANN Variations

The framework supports multiple quantization levels:

- **Q8**: 8-bit quantization
- **Q16**: 16-bit quantization
- **Q32**: 32-bit quantization
- **Q64**: 64-bit quantization

### SNN Variations

For each QANN model, the framework generates SNN models with:

**Time Steps:**
- 32 time steps
- 64 time steps
- 128 time steps

**Encoding Types:**
- `analog`: Analog encoding (continuous values)
- `rate`: Rate-based encoding (spike rates)

This results in 6 SNN variations per QANN model (3 time steps × 2 encodings).

### Total Variations

For a single ANN model:
- **4 QANN models** (Q8, Q16, Q32, Q64)
- **24 SNN models** (4 QANN × 6 SNN variations)
- **24 MLIR exports**
- **24 ONNX exports**

## Available Models

### Pre-trained ANN Models

| Model | Accuracy | Checkpoint |
|-------|----------|------------|
| ViT-Small-ReLU | 82.34% | [Download](https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-small-patch16-relu/resolve/main/vit-small-patch16-relu-82.34.pth) |
| ViT-Base-ReLU | 83.46% | [Download](https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-base-patch16-relu/resolve/main/vit_base_patch16_ReLU_83.458.pth) |
| ViT-Large-ReLU | 85.41% | [Download](https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-large-patch16-relu/resolve/main/vit-large-imagenet-relu-85.41.pth) |

### Pre-trained QANN Models

| Model | Quantization | Accuracy | Checkpoint |
|-------|-------------|----------|------------|
| ViT-Small-Q32 | 32-bit | 81.59% | [Download](https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-small-patch16-relu-q32/resolve/main/vit-small-imagenet-relu-q32-81.59.pth) |
| ViT-Base-Q32 | 32-bit | 82.83% | [Download](https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-base-patch16-relu-q32/resolve/main/vit-base-imagenet-relu-q32-82.83.pth) |
| ViT-Large-Q32 | 32-bit | 83.86% | [Download](https://huggingface.co/XianYiyk/SpikeZIP-TF-vit-large-patch16-relu-q32/resolve/main/vit-large-imagenet-relu-q32-83.86.pth) |

## File Structure

```
conversion_workspace/
├── pretrained_models/        # Downloaded pre-trained models
│   ├── vit-small-patch16-relu-82.34.pth
│   ├── vit-base-patch16-ReLU-83.458.pth
│   └── ...
├── converted_models/          # Converted QANN and SNN models
│   ├── vit_small_patch16_qann_q8.pth
│   ├── vit_small_patch16_qann_q16.pth
│   ├── vit_small_patch16_snn_q32_analog_t64.pth
│   └── ...
├── mlir_exports/             # MLIR format exports
│   ├── vit_small_snn_q32_analog_t64.mlir
│   └── ...
├── onnx_exports/             # ONNX format exports
│   ├── vit_small_snn_q32_analog_t64.onnx
│   ├── vit_small_snn_q32_analog_t64_metadata.json
│   └── ...
├── pipeline_results.json     # Conversion results summary
└── pipeline_log.txt          # Detailed conversion log
```

## MLIR Export Format

The MLIR export uses the linalg dialect with custom SNN operators:

- `snn.if`: Integrate-and-Fire neuron
- `snn.lif`: Leaky Integrate-and-Fire neuron
- `linalg.matmul`: Linear layers
- `linalg.conv_2d`: Convolutional layers
- `linalg.layer_norm`: Layer normalization

Example MLIR output:
```mlir
module {
  func.func @snn_model(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
    %0 = arith.constant dense<...> : tensor<768x384xf32>
    %1 = linalg.matmul ins(%arg0, %0 : tensor<?x?xf32>, tensor<768x384xf32>) ...
    %2 = "snn.if"(%1, %threshold) : (tensor<?x?xf32>, f32) -> tensor<?x?xf32>
    func.return %2 : tensor<?x?xf32>
  }
}
```

## ONNX Export Format

The ONNX export includes custom operators for SNN-specific operations:

- `snn::IFNeuron`: Integrate-and-Fire neuron
- `snn::LIFNeuron`: Leaky Integrate-and-Fire neuron
- `snn::SpikeEncoding`: Spike encoding layer

Metadata is saved alongside each ONNX file with:
- Time steps
- Encoding type
- Input shape
- Custom operators used

## Advanced Usage

### Custom Conversion Parameters

```python
from convert_models import ModelConverter

converter = ModelConverter()

# Load ANN
ann_model = converter.load_ann_model(
    'path/to/ann.pth',
    model_name='vit_small_patch16'
)

# Convert to QANN with custom parameters
qann_model = converter.convert_ann_to_qann(
    ann_model,
    level=16,
    weight_quantization_bit=16
)

# Convert to SNN with custom parameters
snn_model = converter.convert_qann_to_snn(
    qann_model,
    time_step=128,
    encoding_type='rate',
    level=16
)
```

### Batch Processing

Process multiple models:

```bash
for model in vit-small vit-base vit-large; do
    python run_conversion_pipeline.py \
        --model-type $model \
        --work-dir ./workspace_$model
done
```

## Validation

Each conversion step includes validation:

1. **Download**: MD5 checksum verification
2. **QANN**: Parameter quantization checks
3. **SNN**: Spike generation validation
4. **MLIR**: Syntax validation
5. **ONNX**: Model checker validation

## Troubleshooting

### Issue: Out of memory during conversion

**Solution**: Process models one at a time or reduce batch size:
```bash
python convert_models.py --input model.pth --level 32 --qann-only
```

### Issue: ONNX export fails

**Solution**: Ensure ONNX is installed:
```bash
pip install onnx onnxruntime
```

### Issue: Model download fails

**Solution**: Check internet connection and retry with:
```bash
python download_models.py --force
```

## Performance

Approximate conversion times (on GPU):

| Step | ViT-Small | ViT-Base | ViT-Large |
|------|-----------|----------|-----------|
| ANN → QANN | ~10s | ~20s | ~40s |
| QANN → SNN | ~15s | ~30s | ~60s |
| SNN → MLIR | ~5s | ~10s | ~20s |
| SNN → ONNX | ~10s | ~20s | ~40s |

## Citation

If you use this conversion framework, please cite:

```bibtex
@inproceedings{spikeziptf2024,
    title={SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN},
    author={You, Kang and Xu, Zekai and Nie, Chen and Deng, Zhijie and Wang, Xiang and Guo, Qinghai and He, Zhezhi},
    booktitle={Forty-first International Conference on Machine Learning (ICML)},
    year={2024}
}
```

## License

This project is licensed under the MuLan PSL 2.0 License.

## Support

For issues and questions:
- Open an issue on GitHub
- See the main [README.md](../README.md) for more information
- Check the [QUICKSTART.md](../QUICKSTART.md) guide
