#!/usr/bin/env python3
"""
Comprehensive model conversion script for SpikeZIP-TF.
Supports:
- ANN to QANN conversion (multiple quantization levels)
- QANN to SNN conversion (multiple time steps and encoding types)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import SpikeZIP-TF modules
import models_vit
from spike_quan_wrapper import myquan_replace, SNNWrapper


# Conversion configurations
QUANTIZATION_LEVELS = [8, 16, 32, 64]
TIME_STEPS = [32, 64, 128]  # Commonly used time steps
ENCODING_TYPES = ['analog', 'rate']
WEIGHT_QUANTIZATION_BITS = [8, 16, 32]

CONVERSION_VARIATIONS = {
    'ann_to_qann': {
        'q8': {'level': 8, 'weight_bits': 8},
        'q16': {'level': 16, 'weight_bits': 16},
        'q32': {'level': 32, 'weight_bits': 32},
        'q64': {'level': 64, 'weight_bits': 32},
    },
    'qann_to_snn': {
        'analog_t32': {'time_step': 32, 'encoding': 'analog'},
        'analog_t64': {'time_step': 64, 'encoding': 'analog'},
        'analog_t128': {'time_step': 128, 'encoding': 'analog'},
        'rate_t32': {'time_step': 32, 'encoding': 'rate'},
        'rate_t64': {'time_step': 64, 'encoding': 'rate'},
        'rate_t128': {'time_step': 128, 'encoding': 'rate'},
    }
}


class ModelConverter:
    """Main conversion class for ANN -> QANN -> SNN pipeline."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
    
    def load_ann_model(self, checkpoint_path: str, model_name: str = 'vit_small_patch16',
                       act_layer: str = 'relu', num_classes: int = 1000) -> nn.Module:
        """Load a pre-trained ANN model."""
        print(f"\nLoading ANN model from: {checkpoint_path}")
        
        # Create model
        model = models_vit.__dict__[model_name](
            num_classes=num_classes,
            global_pool=False,
            act_layer=act_layer
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load state dict
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded with message: {msg}")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def convert_ann_to_qann(self, ann_model: nn.Module, level: int = 32,
                            weight_quantization_bit: int = 32,
                            define_params: bool = True) -> nn.Module:
        """Convert ANN model to Quantized ANN (QANN)."""
        print(f"\nConverting ANN to QANN (level={level}, weight_bits={weight_quantization_bit})")
        
        # Clone the model to avoid modifying the original
        import copy
        qann_model = copy.deepcopy(ann_model)
        
        # Apply quantization
        qann_model = myquan_replace(
            qann_model,
            mode="QANN",
            level=level,
            weight_quantization_bit=weight_quantization_bit,
            define_params=define_params
        )
        
        qann_model = qann_model.to(self.device)
        qann_model.eval()
        
        print("  ✓ Conversion to QANN complete")
        return qann_model
    
    def convert_qann_to_snn(self, qann_model: nn.Module, time_step: int = 64,
                            encoding_type: str = 'analog', level: int = 32,
                            weight_quantization_bit: int = 32) -> nn.Module:
        """Convert QANN model to Spiking Neural Network (SNN)."""
        print(f"\nConverting QANN to SNN (time_step={time_step}, encoding={encoding_type})")
        
        # Clone the model
        import copy
        snn_model = copy.deepcopy(qann_model)
        
        # Apply SNN conversion
        snn_model = myquan_replace(
            snn_model,
            mode="SNN",
            level=level,
            weight_quantization_bit=weight_quantization_bit,
            define_params=True
        )
        
        snn_model = snn_model.to(self.device)
        snn_model.eval()
        
        print("  ✓ Conversion to SNN complete")
        return snn_model
    
    def save_model(self, model: nn.Module, save_path: str, 
                   metadata: Optional[Dict] = None):
        """Save model checkpoint with metadata."""
        print(f"\nSaving model to: {save_path}")
        
        # Create parent directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'metadata': metadata or {}
        }
        
        # Save
        torch.save(checkpoint, save_path)
        print("  ✓ Model saved successfully")
    
    def convert_all_variations(self, ann_checkpoint: str, output_dir: str,
                               model_name: str = 'vit_small_patch16'):
        """Convert ANN to all QANN and SNN variations."""
        print("\n" + "=" * 70)
        print("Converting ANN to all QANN and SNN variations")
        print("=" * 70)
        
        # Load base ANN model
        ann_model = self.load_ann_model(ann_checkpoint, model_name)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        # Convert to all QANN variations
        for qann_name, qann_config in CONVERSION_VARIATIONS['ann_to_qann'].items():
            try:
                print(f"\n--- Processing QANN variant: {qann_name} ---")
                
                qann_model = self.convert_ann_to_qann(
                    ann_model,
                    level=qann_config['level'],
                    weight_quantization_bit=qann_config['weight_bits']
                )
                
                # Save QANN model
                qann_path = os.path.join(output_dir, f'{model_name}_qann_{qann_name}.pth')
                self.save_model(qann_model, qann_path, metadata={
                    'conversion': 'ann_to_qann',
                    'quantization_level': qann_config['level'],
                    'weight_bits': qann_config['weight_bits']
                })
                results.append(('qann', qann_name, qann_path, True))
                
                # Convert each QANN to all SNN variations
                for snn_name, snn_config in CONVERSION_VARIATIONS['qann_to_snn'].items():
                    try:
                        print(f"\n  --- Processing SNN variant: {snn_name} ---")
                        
                        snn_model = self.convert_qann_to_snn(
                            qann_model,
                            time_step=snn_config['time_step'],
                            encoding_type=snn_config['encoding'],
                            level=qann_config['level'],
                            weight_quantization_bit=qann_config['weight_bits']
                        )
                        
                        # Save SNN model
                        snn_path = os.path.join(
                            output_dir,
                            f'{model_name}_snn_{qann_name}_{snn_name}.pth'
                        )
                        self.save_model(snn_model, snn_path, metadata={
                            'conversion': 'qann_to_snn',
                            'base_qann': qann_name,
                            'time_step': snn_config['time_step'],
                            'encoding': snn_config['encoding'],
                            'quantization_level': qann_config['level'],
                            'weight_bits': qann_config['weight_bits']
                        })
                        results.append(('snn', f"{qann_name}_{snn_name}", snn_path, True))
                        
                    except Exception as e:
                        print(f"  ✗ Error converting to SNN {snn_name}: {e}")
                        results.append(('snn', f"{qann_name}_{snn_name}", None, False))
                
            except Exception as e:
                print(f"✗ Error converting to QANN {qann_name}: {e}")
                results.append(('qann', qann_name, None, False))
        
        # Print summary
        print("\n" + "=" * 70)
        print("Conversion Summary")
        print("=" * 70)
        
        qann_success = sum(1 for r in results if r[0] == 'qann' and r[3])
        qann_total = sum(1 for r in results if r[0] == 'qann')
        snn_success = sum(1 for r in results if r[0] == 'snn' and r[3])
        snn_total = sum(1 for r in results if r[0] == 'snn')
        
        print(f"QANN conversions: {qann_success}/{qann_total} successful")
        print(f"SNN conversions: {snn_success}/{snn_total} successful")
        print(f"\nAll models saved to: {os.path.abspath(output_dir)}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Convert ANN models to QANN and SNN variations'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input ANN checkpoint')
    parser.add_argument('--output-dir', type=str, default='./converted_models',
                        help='Directory to save converted models')
    parser.add_argument('--model-name', type=str, default='vit_small_patch16',
                        choices=['vit_small_patch16', 'vit_base_patch16', 'vit_large_patch16'],
                        help='Model architecture name')
    parser.add_argument('--all-variations', action='store_true',
                        help='Convert to all QANN and SNN variations')
    parser.add_argument('--qann-only', action='store_true',
                        help='Only convert to QANN (skip SNN)')
    parser.add_argument('--level', type=int, default=32,
                        help='Quantization level (default: 32)')
    parser.add_argument('--time-step', type=int, default=64,
                        help='Time step for SNN (default: 64)')
    parser.add_argument('--encoding', type=str, default='analog',
                        choices=['analog', 'rate'],
                        help='Encoding type for SNN')
    
    args = parser.parse_args()
    
    # Create converter
    converter = ModelConverter()
    
    if args.all_variations:
        # Convert to all variations
        converter.convert_all_variations(
            args.input,
            args.output_dir,
            args.model_name
        )
    else:
        # Single conversion pipeline
        print("=" * 70)
        print("Single Conversion Pipeline")
        print("=" * 70)
        
        # Load ANN
        ann_model = converter.load_ann_model(args.input, args.model_name)
        
        # Convert to QANN
        qann_model = converter.convert_ann_to_qann(ann_model, level=args.level)
        qann_path = os.path.join(args.output_dir, f'{args.model_name}_qann_q{args.level}.pth')
        converter.save_model(qann_model, qann_path)
        
        if not args.qann_only:
            # Convert to SNN
            snn_model = converter.convert_qann_to_snn(
                qann_model,
                time_step=args.time_step,
                encoding_type=args.encoding,
                level=args.level
            )
            snn_path = os.path.join(
                args.output_dir,
                f'{args.model_name}_snn_q{args.level}_t{args.time_step}_{args.encoding}.pth'
            )
            converter.save_model(snn_model, snn_path)
        
        print("\n✓ Conversion complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
