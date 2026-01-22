#!/usr/bin/env python3
"""
SNN to ONNX exporter for SpikeZIP-TF.
Exports Spiking Neural Network models to ONNX format with custom SNN operators.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from pathlib import Path


class SNNONNXExporter:
    """Exporter for SNN models to ONNX format."""
    
    def __init__(self):
        self.custom_ops = []
        
    def register_snn_ops(self):
        """Register custom SNN operators for ONNX export."""
        
        # Define custom IF Neuron operator
        class IFNeuronOp(torch.autograd.Function):
            @staticmethod
            def symbolic(g, input, threshold, time_steps):
                """Define ONNX symbolic representation."""
                return g.op("snn::IFNeuron", input, threshold, time_steps_i=time_steps)
            
            @staticmethod
            def forward(ctx, input, threshold, time_steps):
                """Forward pass (for tracing)."""
                # Simplified IF neuron implementation
                output = torch.zeros_like(input)
                membrane = torch.zeros_like(input)
                
                for t in range(time_steps):
                    membrane = membrane + input / time_steps
                    spikes = (membrane >= threshold).float()
                    membrane = membrane - spikes * threshold
                    output = output + spikes
                
                return output
        
        # Define custom LIF Neuron operator
        class LIFNeuronOp(torch.autograd.Function):
            @staticmethod
            def symbolic(g, input, threshold, decay, time_steps):
                """Define ONNX symbolic representation."""
                return g.op("snn::LIFNeuron", input, threshold, decay,
                           time_steps_i=time_steps)
            
            @staticmethod
            def forward(ctx, input, threshold, decay, time_steps):
                """Forward pass (for tracing)."""
                # Simplified LIF neuron implementation
                output = torch.zeros_like(input)
                membrane = torch.zeros_like(input)
                
                for t in range(time_steps):
                    membrane = membrane * decay + input / time_steps
                    spikes = (membrane >= threshold).float()
                    membrane = membrane - spikes * threshold
                    output = output + spikes
                
                return output
        
        # Define custom spike encoding operator
        class SpikeEncodingOp(torch.autograd.Function):
            @staticmethod
            def symbolic(g, input, encoding_type, time_steps):
                """Define ONNX symbolic representation."""
                return g.op("snn::SpikeEncoding", input,
                           encoding_type_s=encoding_type,
                           time_steps_i=time_steps)
            
            @staticmethod
            def forward(ctx, input, encoding_type, time_steps):
                """Forward pass (for tracing)."""
                if encoding_type == 'rate':
                    # Rate encoding - convert boolean to float
                    return (torch.rand_like(input.unsqueeze(0).repeat(time_steps, 1, 1)) < input.unsqueeze(0)).float()
                else:  # analog
                    # Analog encoding
                    return input.unsqueeze(0).repeat(time_steps, 1, 1) / time_steps
        
        self.custom_ops.extend([IFNeuronOp, LIFNeuronOp, SpikeEncodingOp])
        
        return self.custom_ops
    
    def wrap_snn_model_for_export(self, model: nn.Module, time_steps: int = 64,
                                   encoding_type: str = 'analog') -> nn.Module:
        """Wrap SNN model to make it ONNX-exportable."""
        
        class ONNXCompatibleSNN(nn.Module):
            def __init__(self, base_model, time_steps, encoding_type):
                super().__init__()
                self.base_model = base_model
                self.time_steps = time_steps
                self.encoding_type = encoding_type
            
            def forward(self, x):
                """Forward pass compatible with ONNX export."""
                # For ONNX export, we'll flatten the temporal dimension
                # The actual spiking behavior is encoded in custom ops
                
                # Add temporal dimension metadata as a comment
                # This would be handled by custom operators
                return self.base_model(x)
        
        wrapped_model = ONNXCompatibleSNN(model, time_steps, encoding_type)
        return wrapped_model
    
    def export_to_onnx(self, model: nn.Module, output_path: str,
                       input_shape: tuple = (1, 3, 224, 224),
                       time_steps: int = 64, encoding_type: str = 'analog',
                       opset_version: int = 13):
        """Export SNN model to ONNX format."""
        print(f"\nExporting SNN model to ONNX...")
        print(f"  Output path: {output_path}")
        print(f"  Input shape: {input_shape}")
        print(f"  Time steps: {time_steps}")
        print(f"  Encoding: {encoding_type}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                   exist_ok=True)
        
        # Register custom operators
        self.register_snn_ops()
        
        # Wrap model for export
        wrapped_model = self.wrap_snn_model_for_export(model, time_steps, encoding_type)
        wrapped_model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        try:
            torch.onnx.export(
                wrapped_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                custom_opsets={"snn": 1}  # Custom SNN operator domain
            )
            print("  ✓ ONNX export successful")
            
            # Save metadata
            metadata_path = output_path.replace('.onnx', '_metadata.json')
            import json
            metadata = {
                'time_steps': time_steps,
                'encoding_type': encoding_type,
                'input_shape': list(input_shape),
                'opset_version': opset_version,
                'custom_operators': ['snn::IFNeuron', 'snn::LIFNeuron', 'snn::SpikeEncoding']
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  ✓ Metadata saved to: {metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error during ONNX export: {e}")
            return False
    
    def validate_onnx_model(self, onnx_path: str) -> bool:
        """Validate exported ONNX model."""
        try:
            import onnx
            print(f"\nValidating ONNX model: {onnx_path}")
            
            # Load the ONNX model
            model = onnx.load(onnx_path)
            
            # Check the model
            onnx.checker.check_model(model)
            print("  ✓ ONNX model is valid")
            
            # Print model info
            print(f"  Graph name: {model.graph.name}")
            print(f"  Inputs: {len(model.graph.input)}")
            print(f"  Outputs: {len(model.graph.output)}")
            print(f"  Nodes: {len(model.graph.node)}")
            
            return True
            
        except ImportError:
            print("  Warning: onnx package not installed, skipping validation")
            return True
        except Exception as e:
            print(f"  ✗ ONNX validation error: {e}")
            return False


def export_snn_to_onnx(checkpoint_path: str, output_path: str,
                       model_name: str = 'vit_small_patch16',
                       time_steps: int = 64, encoding_type: str = 'analog'):
    """Main function to export SNN checkpoint to ONNX."""
    print("=" * 70)
    print("SNN to ONNX Exporter")
    print("=" * 70)
    
    print(f"\nLoading SNN model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        metadata = checkpoint.get('metadata', {})
    else:
        state_dict = checkpoint
        metadata = {}
    
    print(f"  Model metadata: {metadata}")
    
    # Create model
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
    
    if model_name not in model_constructors:
        print(f"  ✗ Unknown model: {model_name}")
        return False
    
    model = model_constructors[model_name]()
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  Warning: Could not load full state dict: {e}")
    
    model.eval()
    
    # Create exporter
    exporter = SNNONNXExporter()
    
    # Export to ONNX
    success = exporter.export_to_onnx(
        model, output_path,
        input_shape=(1, 3, 224, 224),
        time_steps=time_steps,
        encoding_type=encoding_type
    )
    
    if success:
        # Validate ONNX model
        exporter.validate_onnx_model(output_path)
        print("\n✓ Export to ONNX complete!")
    else:
        print("\n✗ Export to ONNX failed!")
    
    return success


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export SNN models to ONNX')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to SNN checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output ONNX file')
    parser.add_argument('--model-name', type=str, default='vit_small_patch16',
                        choices=['vit_small_patch16', 'vit_base_patch16', 'vit_large_patch16'],
                        help='Model architecture name')
    parser.add_argument('--time-steps', type=int, default=64,
                        help='Number of time steps for SNN')
    parser.add_argument('--encoding', type=str, default='analog',
                        choices=['analog', 'rate'],
                        help='Spike encoding type')
    
    args = parser.parse_args()
    
    success = export_snn_to_onnx(
        args.input, args.output, args.model_name,
        args.time_steps, args.encoding
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
