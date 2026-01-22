#!/usr/bin/env python3
"""
Verification script to test the conversion framework components.
Tests basic functionality without requiring full model downloads.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 70)
    print("Testing Module Imports")
    print("=" * 70)
    
    tests = {
        'download_models': False,
        'convert_models': False,
        'snn_to_mlir': False,
        'snn_to_onnx': False,
        'run_conversion_pipeline': False,
    }
    
    # Test download_models (no torch required)
    try:
        import download_models
        tests['download_models'] = True
        print("✓ download_models.py - OK")
    except Exception as e:
        print(f"✗ download_models.py - FAILED: {e}")
    
    # Test other modules (require torch)
    try:
        import torch
        torch_available = True
        print(f"✓ PyTorch {torch.__version__} available")
    except ImportError:
        torch_available = False
        print("✗ PyTorch not available (required for conversion modules)")
    
    if torch_available:
        for module_name in ['convert_models', 'snn_to_mlir', 'snn_to_onnx', 'run_conversion_pipeline']:
            try:
                __import__(module_name)
                tests[module_name] = True
                print(f"✓ {module_name}.py - OK")
            except Exception as e:
                print(f"✗ {module_name}.py - FAILED: {e}")
    
    return tests


def test_file_structure():
    """Test that all files exist."""
    print("\n" + "=" * 70)
    print("Testing File Structure")
    print("=" * 70)
    
    required_files = [
        'download_models.py',
        'convert_models.py',
        'snn_to_mlir.py',
        'snn_to_onnx.py',
        'run_conversion_pipeline.py',
        'CONVERSION_FRAMEWORK.md',
    ]
    
    all_exist = True
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✓ {filename} exists")
        else:
            print(f"✗ {filename} missing")
            all_exist = False
    
    return all_exist


def test_executability():
    """Test that scripts are executable."""
    print("\n" + "=" * 70)
    print("Testing Script Executability")
    print("=" * 70)
    
    scripts = [
        'download_models.py',
        'convert_models.py',
        'snn_to_mlir.py',
        'snn_to_onnx.py',
        'run_conversion_pipeline.py',
    ]
    
    all_executable = True
    for script in scripts:
        if os.path.exists(script):
            is_executable = os.access(script, os.X_OK)
            if is_executable:
                print(f"✓ {script} is executable")
            else:
                print(f"✗ {script} is not executable")
                all_executable = False
    
    return all_executable


def print_usage_examples():
    """Print usage examples."""
    print("\n" + "=" * 70)
    print("Usage Examples")
    print("=" * 70)
    
    examples = """
1. Download all models:
   python download_models.py --output-dir ./pretrained_models --models all

2. Convert ANN to all variations:
   python convert_models.py \\
       --input ./pretrained_models/vit-small-patch16-relu-82.34.pth \\
       --output-dir ./converted_models \\
       --model-name vit_small_patch16 \\
       --all-variations

3. Export SNN to MLIR:
   python snn_to_mlir.py \\
       --input ./converted_models/vit_small_patch16_snn_q32_analog_t64.pth \\
       --output ./mlir_exports/vit_small_snn.mlir

4. Export SNN to ONNX:
   python snn_to_onnx.py \\
       --input ./converted_models/vit_small_patch16_snn_q32_analog_t64.pth \\
       --output ./onnx_exports/vit_small_snn.onnx

5. Run complete pipeline:
   python run_conversion_pipeline.py --model-type vit-small
"""
    print(examples)


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("SpikeZIP-TF Conversion Framework - Verification")
    print("=" * 70 + "\n")
    
    # Run tests
    files_ok = test_file_structure()
    exec_ok = test_executability()
    imports_ok = test_imports()
    
    # Print usage examples
    print_usage_examples()
    
    # Summary
    print("=" * 70)
    print("Verification Summary")
    print("=" * 70)
    print(f"File structure: {'✓ PASS' if files_ok else '✗ FAIL'}")
    print(f"Executability: {'✓ PASS' if exec_ok else '✗ FAIL'}")
    print(f"Imports: {'✓ PASS' if all(imports_ok.values()) else '⚠ PARTIAL (PyTorch required for full functionality)'}")
    
    if not all(imports_ok.values()):
        print("\nNote: Install dependencies to enable all features:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
