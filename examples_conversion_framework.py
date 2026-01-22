#!/usr/bin/env python3
"""
Example usage script demonstrating the conversion framework.
This script shows how to use each component of the framework.
"""

import os
import sys

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def example_download_models():
    """Example: Download models."""
    print_section("Example 1: Download Models")
    
    print("To download all pre-trained models:")
    print("  python download_models.py --output-dir ./pretrained_models --models all")
    print()
    print("To download only ANN models:")
    print("  python download_models.py --output-dir ./pretrained_models --models ann")
    print()
    print("To download only QANN models:")
    print("  python download_models.py --output-dir ./pretrained_models --models qann")
    print()
    print("To force re-download:")
    print("  python download_models.py --output-dir ./pretrained_models --models all --force")


def example_convert_single():
    """Example: Convert single model."""
    print_section("Example 2: Convert Single Model (ANN → QANN → SNN)")
    
    print("Convert with specific parameters:")
    print("""
python convert_models.py \\
    --input ./pretrained_models/vit-small-patch16-relu-82.34.pth \\
    --output-dir ./converted_models \\
    --model-name vit_small_patch16 \\
    --level 32 \\
    --time-step 64 \\
    --encoding analog
""")
    
    print("This will create:")
    print("  - QANN model: vit_small_patch16_qann_q32.pth")
    print("  - SNN model: vit_small_patch16_snn_q32_t64_analog.pth")


def example_convert_all_variations():
    """Example: Convert to all variations."""
    print_section("Example 3: Convert to All Variations")
    
    print("Convert ANN to all QANN and SNN variations:")
    print("""
python convert_models.py \\
    --input ./pretrained_models/vit-small-patch16-relu-82.34.pth \\
    --output-dir ./converted_models \\
    --model-name vit_small_patch16 \\
    --all-variations
""")
    
    print("This will create:")
    print("  - 4 QANN models (Q8, Q16, Q32, Q64)")
    print("  - 24 SNN models (4 QANN × 6 variations)")
    print("    - 3 time steps: 32, 64, 128")
    print("    - 2 encodings: analog, rate")


def example_export_mlir():
    """Example: Export to MLIR."""
    print_section("Example 4: Export SNN to MLIR")
    
    print("Export SNN model to MLIR linalg dialect:")
    print("""
python snn_to_mlir.py \\
    --input ./converted_models/vit_small_patch16_snn_q32_analog_t64.pth \\
    --output ./mlir_exports/vit_small_snn.mlir \\
    --model-name vit_small_patch16
""")
    
    print("The MLIR file will contain:")
    print("  - Custom SNN operators (snn.if, snn.lif)")
    print("  - Linalg operations (matmul, conv_2d, layer_norm)")
    print("  - Full model graph representation")


def example_export_onnx():
    """Example: Export to ONNX."""
    print_section("Example 5: Export SNN to ONNX")
    
    print("Export SNN model to ONNX format:")
    print("""
python snn_to_onnx.py \\
    --input ./converted_models/vit_small_patch16_snn_q32_analog_t64.pth \\
    --output ./onnx_exports/vit_small_snn.onnx \\
    --model-name vit_small_patch16 \\
    --time-steps 64 \\
    --encoding analog
""")
    
    print("This will create:")
    print("  - ONNX model: vit_small_snn.onnx")
    print("  - Metadata: vit_small_snn_metadata.json")
    print()
    print("The ONNX model includes:")
    print("  - Custom SNN operators (snn::IFNeuron, snn::LIFNeuron, snn::SpikeEncoding)")
    print("  - Standard operators for linear, conv, etc.")


def example_complete_pipeline():
    """Example: Complete pipeline."""
    print_section("Example 6: Run Complete Pipeline")
    
    print("Run the entire conversion pipeline:")
    print("""
python run_conversion_pipeline.py \\
    --work-dir ./conversion_workspace \\
    --model-type vit-small
""")
    
    print("This will:")
    print("  1. Download all pre-trained models")
    print("  2. Convert ANN to all QANN variations")
    print("  3. Convert all QANN to all SNN variations")
    print("  4. Export all SNN models to MLIR")
    print("  5. Export all SNN models to ONNX")
    print()
    print("Results will be saved to:")
    print("  - ./conversion_workspace/pretrained_models/")
    print("  - ./conversion_workspace/converted_models/")
    print("  - ./conversion_workspace/mlir_exports/")
    print("  - ./conversion_workspace/onnx_exports/")
    print("  - ./conversion_workspace/pipeline_results.json")
    print("  - ./conversion_workspace/pipeline_log.txt")


def example_python_api():
    """Example: Using Python API."""
    print_section("Example 7: Using Python API")
    
    print("Use the framework from Python code:")
    print("""
from convert_models import ModelConverter

# Create converter
converter = ModelConverter()

# Load ANN model
ann_model = converter.load_ann_model(
    'pretrained_models/vit-small-patch16-relu-82.34.pth',
    model_name='vit_small_patch16'
)

# Convert to QANN
qann_model = converter.convert_ann_to_qann(
    ann_model,
    level=32,
    weight_quantization_bit=32
)

# Save QANN
converter.save_model(
    qann_model,
    'converted_models/custom_qann.pth',
    metadata={'level': 32}
)

# Convert to SNN
snn_model = converter.convert_qann_to_snn(
    qann_model,
    time_step=64,
    encoding_type='analog',
    level=32
)

# Save SNN
converter.save_model(
    snn_model,
    'converted_models/custom_snn.pth',
    metadata={'time_step': 64, 'encoding': 'analog'}
)
""")


def example_batch_processing():
    """Example: Batch processing."""
    print_section("Example 8: Batch Processing Multiple Models")
    
    print("Process all model types with a bash script:")
    print("""
#!/bin/bash

for model in vit-small vit-base vit-large; do
    echo "Processing $model..."
    
    python run_conversion_pipeline.py \\
        --work-dir ./workspace_$model \\
        --model-type $model
    
    echo "$model complete!"
done

echo "All models processed!"
""")


def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("  SpikeZIP-TF Conversion Framework - Usage Examples")
    print("=" * 70)
    
    print("""
This script demonstrates various ways to use the conversion framework.
Choose the example that best fits your use case.

For complete documentation, see: CONVERSION_FRAMEWORK.md
""")
    
    # Show all examples
    example_download_models()
    example_convert_single()
    example_convert_all_variations()
    example_export_mlir()
    example_export_onnx()
    example_complete_pipeline()
    example_python_api()
    example_batch_processing()
    
    # Summary
    print_section("Quick Reference")
    print("Download:     python download_models.py --models all")
    print("Convert:      python convert_models.py --input model.pth --all-variations")
    print("MLIR Export:  python snn_to_mlir.py --input snn_model.pth --output model.mlir")
    print("ONNX Export:  python snn_to_onnx.py --input snn_model.pth --output model.onnx")
    print("Full Pipeline: python run_conversion_pipeline.py --model-type vit-small")
    print("Verify:       python verify_conversion_framework.py")
    
    print("\n" + "=" * 70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
