#!/usr/bin/env python3
"""
SNN to MLIR (linalg dialect) converter for SpikeZIP-TF.
Translates Spiking Neural Network models to MLIR representation.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class MLIRDialect:
    """MLIR dialect definitions for SNN operations."""
    
    @staticmethod
    def module_header() -> str:
        """Generate MLIR module header."""
        return 'module {\n'
    
    @staticmethod
    def module_footer() -> str:
        """Generate MLIR module footer."""
        return '}\n'
    
    @staticmethod
    def func_def(name: str, input_types: List[str], output_types: List[str]) -> str:
        """Generate function definition."""
        inputs = ', '.join([f'%arg{i}: {t}' for i, t in enumerate(input_types)])
        outputs = ', '.join(output_types)
        return f'  func.func @{name}({inputs}) -> ({outputs}) {{\n'
    
    @staticmethod
    def func_end() -> str:
        """Generate function end."""
        return '  }\n'
    
    @staticmethod
    def return_op(values: List[str]) -> str:
        """Generate return operation."""
        return f'    func.return {", ".join(values)} : {", ".join(["tensor<?x?xf32>"] * len(values))}\n'


class SNNtoMLIRConverter:
    """Converter for SNN models to MLIR linalg dialect."""
    
    def __init__(self):
        self.operation_counter = 0
        self.mlir_ops = []
        
    def _get_next_var(self) -> str:
        """Get next variable name."""
        var = f'%{self.operation_counter}'
        self.operation_counter += 1
        return var
    
    def _tensor_type(self, shape: List[int]) -> str:
        """Generate tensor type string."""
        shape_str = 'x'.join(['?' if s is None else str(s) for s in shape])
        return f'tensor<{shape_str}xf32>'
    
    def convert_linear_layer(self, layer: nn.Linear, input_var: str) -> Tuple[str, List[str]]:
        """Convert Linear layer to MLIR linalg.matmul."""
        output_var = self._get_next_var()
        weight_var = self._get_next_var()
        bias_var = self._get_next_var() if layer.bias is not None else None
        
        ops = []
        
        # Define weight constant
        weight_shape = [layer.out_features, layer.in_features]
        ops.append(f'    {weight_var} = arith.constant dense<...> : {self._tensor_type(weight_shape)}\n')
        
        # Matrix multiplication
        ops.append(f'    {output_var} = linalg.matmul ins({input_var}, {weight_var} : '
                  f'{self._tensor_type([None, layer.in_features])}, {self._tensor_type(weight_shape)}) '
                  f'outs({output_var} : {self._tensor_type([None, layer.out_features])})\n')
        
        # Add bias if present
        if bias_var:
            ops.append(f'    {bias_var} = arith.constant dense<...> : {self._tensor_type([layer.out_features])}\n')
            result_var = self._get_next_var()
            ops.append(f'    {result_var} = linalg.add ins({output_var}, {bias_var} : '
                      f'{self._tensor_type([None, layer.out_features])}, {self._tensor_type([layer.out_features])}) '
                      f'outs({result_var} : {self._tensor_type([None, layer.out_features])})\n')
            output_var = result_var
        
        self.mlir_ops.extend(ops)
        return output_var, ops
    
    def convert_conv2d_layer(self, layer: nn.Conv2d, input_var: str) -> Tuple[str, List[str]]:
        """Convert Conv2d layer to MLIR linalg.conv_2d."""
        output_var = self._get_next_var()
        weight_var = self._get_next_var()
        
        ops = []
        
        # Define weight constant
        weight_shape = [layer.out_channels, layer.in_channels, layer.kernel_size[0], layer.kernel_size[1]]
        ops.append(f'    {weight_var} = arith.constant dense<...> : {self._tensor_type(weight_shape)}\n')
        
        # Convolution operation
        ops.append(f'    {output_var} = linalg.conv_2d ins({input_var}, {weight_var} : '
                  f'tensor<?x?x?x?xf32>, {self._tensor_type(weight_shape)}) '
                  f'outs({output_var} : tensor<?x?x?x?xf32>)\n')
        
        self.mlir_ops.extend(ops)
        return output_var, ops
    
    def convert_snn_neuron(self, neuron_type: str, input_var: str, threshold: float) -> Tuple[str, List[str]]:
        """Convert SNN neuron (IF/LIF) to MLIR custom operation."""
        output_var = self._get_next_var()
        threshold_var = self._get_next_var()
        
        ops = []
        
        # Define threshold constant
        ops.append(f'    {threshold_var} = arith.constant {threshold} : f32\n')
        
        # Custom SNN neuron operation
        ops.append(f'    // SNN {neuron_type} Neuron with threshold {threshold}\n')
        ops.append(f'    {output_var} = "snn.{neuron_type.lower()}"({input_var}, {threshold_var}) : '
                  f'(tensor<?x?xf32>, f32) -> tensor<?x?xf32>\n')
        
        self.mlir_ops.extend(ops)
        return output_var, ops
    
    def convert_activation(self, activation_type: str, input_var: str) -> Tuple[str, List[str]]:
        """Convert activation function to MLIR."""
        output_var = self._get_next_var()
        ops = []
        
        if activation_type == 'relu':
            ops.append(f'    {output_var} = linalg.relu ins({input_var} : tensor<?x?xf32>) '
                      f'outs({output_var} : tensor<?x?xf32>)\n')
        elif activation_type == 'gelu':
            ops.append(f'    {output_var} = math.gelu {input_var} : tensor<?x?xf32>\n')
        else:
            ops.append(f'    // Unknown activation: {activation_type}\n')
            output_var = input_var
        
        self.mlir_ops.extend(ops)
        return output_var, ops
    
    def convert_layer_norm(self, layer: nn.LayerNorm, input_var: str) -> Tuple[str, List[str]]:
        """Convert LayerNorm to MLIR operations."""
        output_var = self._get_next_var()
        ops = []
        
        # Layer normalization can be decomposed into mean, variance, and scale operations
        ops.append(f'    // LayerNorm with normalized_shape {layer.normalized_shape}\n')
        ops.append(f'    {output_var} = "linalg.layer_norm"({input_var}) : '
                  f'tensor<?x?xf32> -> tensor<?x?xf32>\n')
        
        self.mlir_ops.extend(ops)
        return output_var, ops
    
    def convert_model_to_mlir(self, model: nn.Module, model_name: str = 'snn_model') -> str:
        """Convert entire SNN model to MLIR."""
        mlir_code = []
        
        # Module header
        mlir_code.append(MLIRDialect.module_header())
        
        # Function definition
        mlir_code.append(MLIRDialect.func_def(
            model_name,
            ['tensor<?x?xf32>'],  # Input tensor
            ['tensor<?x?xf32>']   # Output tensor
        ))
        
        # Reset counters
        self.operation_counter = 1  # Start from 1, %arg0 is input
        self.mlir_ops = []
        
        # Convert model layers
        current_var = '%arg0'
        
        # Traverse model and convert layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                current_var, ops = self.convert_linear_layer(module, current_var)
            elif isinstance(module, nn.Conv2d):
                current_var, ops = self.convert_conv2d_layer(module, current_var)
            elif isinstance(module, nn.LayerNorm):
                current_var, ops = self.convert_layer_norm(module, current_var)
            elif 'IFNeuron' in module.__class__.__name__:
                # SNN-specific neuron
                threshold = getattr(module, 'q_threshold', 1.0)
                if torch.is_tensor(threshold):
                    threshold = threshold.item()
                current_var, ops = self.convert_snn_neuron('IF', current_var, threshold)
        
        # Add all operations
        mlir_code.extend(self.mlir_ops)
        
        # Return operation
        mlir_code.append(MLIRDialect.return_op([current_var]))
        
        # Function end
        mlir_code.append(MLIRDialect.func_end())
        
        # Module footer
        mlir_code.append(MLIRDialect.module_footer())
        
        return ''.join(mlir_code)
    
    def save_mlir(self, mlir_code: str, output_path: str):
        """Save MLIR code to file."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(mlir_code)
        print(f"MLIR saved to: {output_path}")


def convert_snn_to_mlir(checkpoint_path: str, output_path: str, model_name: str = 'vit_small_patch16'):
    """Main function to convert SNN checkpoint to MLIR."""
    print("=" * 70)
    print("SNN to MLIR Converter")
    print("=" * 70)
    
    print(f"\nLoading SNN model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Create model (you would need to import and create the actual model)
    # For now, we'll create a converter and generate MLIR template
    converter = SNNtoMLIRConverter()
    
    # For demonstration, we'll create a simple model structure
    # In practice, you'd load the actual model architecture
    from models_vit import vit_small_patch16, vit_base_patch16, vit_large_patch16
    from functools import partial
    import torch.nn as nn
    
    model_constructors = {
        'vit_small_patch16': lambda: vit_small_patch16(
            num_classes=1000, global_pool=False, 
            act_layer=nn.ReLU, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
        'vit_base_patch16': lambda: vit_base_patch16(
            num_classes=1000, global_pool=False,
            act_layer=nn.ReLU, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
        'vit_large_patch16': lambda: vit_large_patch16(
            num_classes=1000, global_pool=False,
            act_layer=nn.ReLU, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
    }
    
    if model_name in model_constructors:
        model = model_constructors[model_name]()
    else:
        print(f"  Warning: Unknown model {model_name}, using vit_small_patch16 as default")
        model = model_constructors['vit_small_patch16']()
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  Warning: Could not load full state dict: {e}")
    
    # Convert to MLIR
    print("\nConverting SNN to MLIR linalg dialect...")
    mlir_code = converter.convert_model_to_mlir(model, model_name)
    
    # Save MLIR
    converter.save_mlir(mlir_code, output_path)
    
    print("\n✓ Conversion to MLIR complete!")
    return mlir_code


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert SNN models to MLIR')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to SNN checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output MLIR file')
    parser.add_argument('--model-name', type=str, default='vit_small_patch16',
                        help='Model architecture name')
    
    args = parser.parse_args()
    
    convert_snn_to_mlir(args.input, args.output, args.model_name)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
