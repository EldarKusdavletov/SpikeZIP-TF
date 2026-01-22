#!/usr/bin/env python3
"""
Main orchestration script for SpikeZIP-TF conversion pipeline.
Handles:
1. Downloading models and data
2. Converting ANN to QANN to SNN (all variations)
3. Exporting to MLIR and ONNX
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List

# Import our conversion modules
try:
    from download_models import main as download_main
    from convert_models import ModelConverter
    from snn_to_mlir import convert_snn_to_mlir
    from snn_to_onnx import export_snn_to_onnx
except ImportError as e:
    print(f"Warning: Could not import conversion modules: {e}")
    print("Make sure all conversion scripts are in the same directory.")


class ConversionPipeline:
    """Main pipeline for complete model conversion workflow."""
    
    def __init__(self, work_dir: str = './conversion_workspace'):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        self.models_dir = self.work_dir / 'pretrained_models'
        self.converted_dir = self.work_dir / 'converted_models'
        self.mlir_dir = self.work_dir / 'mlir_exports'
        self.onnx_dir = self.work_dir / 'onnx_exports'
        
        # Create directories
        for dir_path in [self.models_dir, self.converted_dir, self.mlir_dir, self.onnx_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.pipeline_log = []
    
    def log(self, message: str, level: str = 'INFO'):
        """Log a message."""
        log_entry = f"[{level}] {message}"
        print(log_entry)
        self.pipeline_log.append(log_entry)
    
    def step_1_download_models(self, models_type: str = 'all') -> bool:
        """Step 1: Download pre-trained models."""
        self.log("=" * 70)
        self.log("STEP 1: Downloading Models")
        self.log("=" * 70)
        
        try:
            # Use download_models.py
            import download_models
            
            # Simulate command line args
            sys.argv = [
                'download_models.py',
                '--output-dir', str(self.models_dir),
                '--models', models_type
            ]
            
            result = download_models.main()
            
            if result == 0:
                self.log("✓ Models downloaded successfully")
                return True
            else:
                self.log("✗ Some models failed to download", 'WARNING')
                return False
                
        except Exception as e:
            self.log(f"✗ Error downloading models: {e}", 'ERROR')
            return False
    
    def step_2_convert_all_variations(self, model_type: str = 'vit-small') -> Dict:
        """Step 2: Convert ANN to QANN to SNN (all variations)."""
        self.log("=" * 70)
        self.log("STEP 2: Converting ANN -> QANN -> SNN (All Variations)")
        self.log("=" * 70)
        
        model_mapping = {
            'vit-small': {
                'ann_file': 'vit-small-patch16-relu-82.34.pth',
                'model_name': 'vit_small_patch16'
            },
            'vit-base': {
                'ann_file': 'vit_base_patch16_ReLU_83.458.pth',
                'model_name': 'vit_base_patch16'
            },
            'vit-large': {
                'ann_file': 'vit-large-imagenet-relu-85.41.pth',
                'model_name': 'vit_large_patch16'
            }
        }
        
        if model_type not in model_mapping:
            self.log(f"✗ Unknown model type: {model_type}", 'ERROR')
            return {}
        
        config = model_mapping[model_type]
        ann_path = self.models_dir / config['ann_file']
        
        if not ann_path.exists():
            self.log(f"✗ ANN model not found: {ann_path}", 'ERROR')
            return {}
        
        try:
            converter = ModelConverter()
            results = converter.convert_all_variations(
                str(ann_path),
                str(self.converted_dir),
                config['model_name']
            )
            
            self.log(f"✓ Converted {len(results)} model variations")
            return {
                'model_type': model_type,
                'conversions': results
            }
            
        except Exception as e:
            self.log(f"✗ Error during conversion: {e}", 'ERROR')
            return {}
    
    def step_3_export_to_mlir(self, snn_models: List[str]) -> Dict:
        """Step 3: Export SNN models to MLIR."""
        self.log("=" * 70)
        self.log("STEP 3: Exporting SNN Models to MLIR")
        self.log("=" * 70)
        
        mlir_exports = []
        
        for snn_path in snn_models:
            if not os.path.exists(snn_path):
                self.log(f"  ✗ SNN model not found: {snn_path}", 'WARNING')
                continue
            
            # Generate output path
            model_name = Path(snn_path).stem
            mlir_path = self.mlir_dir / f"{model_name}.mlir"
            
            try:
                self.log(f"  Converting {model_name} to MLIR...")
                convert_snn_to_mlir(snn_path, str(mlir_path))
                mlir_exports.append(str(mlir_path))
                self.log(f"  ✓ Exported to {mlir_path}")
            except Exception as e:
                self.log(f"  ✗ Error exporting {model_name}: {e}", 'ERROR')
        
        return {
            'total': len(snn_models),
            'successful': len(mlir_exports),
            'exports': mlir_exports
        }
    
    def step_4_export_to_onnx(self, snn_models: List[str]) -> Dict:
        """Step 4: Export SNN models to ONNX."""
        self.log("=" * 70)
        self.log("STEP 4: Exporting SNN Models to ONNX")
        self.log("=" * 70)
        
        onnx_exports = []
        
        for snn_path in snn_models:
            if not os.path.exists(snn_path):
                self.log(f"  ✗ SNN model not found: {snn_path}", 'WARNING')
                continue
            
            # Generate output path
            model_name = Path(snn_path).stem
            onnx_path = self.onnx_dir / f"{model_name}.onnx"
            
            # Extract model info from filename
            # Format: modelname_snn_qXX_tYY_encoding
            parts = model_name.split('_')
            base_model = '_'.join(parts[:3])  # e.g., vit_small_patch16
            
            # Default values
            time_steps = 64
            encoding = 'analog'
            
            # Try to extract from metadata or filename
            try:
                if 't32' in model_name:
                    time_steps = 32
                elif 't64' in model_name:
                    time_steps = 64
                elif 't128' in model_name:
                    time_steps = 128
                
                if 'rate' in model_name:
                    encoding = 'rate'
            except:
                pass
            
            try:
                self.log(f"  Converting {model_name} to ONNX...")
                success = export_snn_to_onnx(
                    snn_path, str(onnx_path),
                    model_name=base_model,
                    time_steps=time_steps,
                    encoding_type=encoding
                )
                
                if success:
                    onnx_exports.append(str(onnx_path))
                    self.log(f"  ✓ Exported to {onnx_path}")
                else:
                    self.log(f"  ✗ Export failed for {model_name}", 'WARNING')
                    
            except Exception as e:
                self.log(f"  ✗ Error exporting {model_name}: {e}", 'ERROR')
        
        return {
            'total': len(snn_models),
            'successful': len(onnx_exports),
            'exports': onnx_exports
        }
    
    def run_complete_pipeline(self, model_type: str = 'vit-small') -> Dict:
        """Run the complete conversion pipeline."""
        self.log("\n" + "=" * 70)
        self.log("COMPLETE CONVERSION PIPELINE")
        self.log("=" * 70)
        
        results = {
            'success': True,
            'steps': {}
        }
        
        # Step 1: Download models
        if self.step_1_download_models('all'):
            results['steps']['download'] = {'status': 'success'}
        else:
            results['steps']['download'] = {'status': 'partial'}
        
        # Step 2: Convert to all variations
        conversion_results = self.step_2_convert_all_variations(model_type)
        results['steps']['conversion'] = conversion_results
        
        # Collect all SNN models for export
        snn_models = []
        if conversion_results and 'conversions' in conversion_results:
            for conv_type, name, path, success in conversion_results['conversions']:
                if conv_type == 'snn' and success and path:
                    snn_models.append(path)
        
        # Step 3: Export to MLIR
        if snn_models:
            mlir_results = self.step_3_export_to_mlir(snn_models)
            results['steps']['mlir_export'] = mlir_results
        
        # Step 4: Export to ONNX
        if snn_models:
            onnx_results = self.step_4_export_to_onnx(snn_models)
            results['steps']['onnx_export'] = onnx_results
        
        # Save results
        results_file = self.work_dir / 'pipeline_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.log(f"\n✓ Pipeline results saved to: {results_file}")
        
        # Save log
        log_file = self.work_dir / 'pipeline_log.txt'
        with open(log_file, 'w') as f:
            f.write('\n'.join(self.pipeline_log))
        
        self.log(f"✓ Pipeline log saved to: {log_file}")
        
        return results
    
    def print_summary(self, results: Dict):
        """Print pipeline summary."""
        self.log("\n" + "=" * 70)
        self.log("PIPELINE SUMMARY")
        self.log("=" * 70)
        
        if 'steps' in results:
            for step_name, step_results in results['steps'].items():
                self.log(f"\n{step_name.upper().replace('_', ' ')}:")
                if isinstance(step_results, dict):
                    for key, value in step_results.items():
                        self.log(f"  {key}: {value}")
        
        self.log("\n" + "=" * 70)
        self.log(f"Work directory: {self.work_dir.absolute()}")
        self.log("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Complete conversion pipeline for SpikeZIP-TF'
    )
    parser.add_argument('--work-dir', type=str, default='./conversion_workspace',
                        help='Working directory for all outputs')
    parser.add_argument('--model-type', type=str, default='vit-small',
                        choices=['vit-small', 'vit-base', 'vit-large'],
                        help='Model type to convert')
    parser.add_argument('--download-only', action='store_true',
                        help='Only download models, skip conversion')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading, use existing models')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ConversionPipeline(args.work_dir)
    
    if args.download_only:
        # Only download models
        pipeline.step_1_download_models('all')
    elif args.skip_download:
        # Skip download and run conversion
        conversion_results = pipeline.step_2_convert_all_variations(args.model_type)
        
        # Get SNN models for export
        snn_models = []
        if conversion_results and 'conversions' in conversion_results:
            for conv_type, name, path, success in conversion_results['conversions']:
                if conv_type == 'snn' and success and path:
                    snn_models.append(path)
        
        if snn_models:
            pipeline.step_3_export_to_mlir(snn_models)
            pipeline.step_4_export_to_onnx(snn_models)
    else:
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(args.model_type)
        pipeline.print_summary(results)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
