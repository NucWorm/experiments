#!/usr/bin/env python3
"""
Evaluation script for Cellpose3 denoising results.

This script integrates with the NucWorm evaluation framework to evaluate
the denoising results against ground truth.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_evaluation(output_dir, config):
    """Run evaluation using the NucWorm evaluation framework."""
    output_path = Path(output_dir)
    center_point_dir = output_path / 'center_point'
    
    if not center_point_dir.exists():
        logger.error(f"Center point directory not found: {center_point_dir}")
        return False
    
    # Find the evaluation script
    eval_script = Path(__file__).parent.parent.parent.parent.parent / 'utils' / 'evaluation' / 'evaluate_centroids.py'
    
    if not eval_script.exists():
        logger.error(f"Evaluation script not found: {eval_script}")
        return False
    
    logger.info(f"Running evaluation script: {eval_script}")
    
    # Run evaluation for each dataset
    datasets = ['nejatbakhsh20', 'wen20', 'yemini21']
    
    for dataset in datasets:
        dataset_dir = center_point_dir / dataset
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            continue
        
        logger.info(f"Evaluating dataset: {dataset}")
        
        # Run evaluation
        cmd = [
            'python', str(eval_script),
            '--predictions', str(dataset_dir),
            '--ground_truth', f'/projects/weilab/gohaina/nucworm/outputs/data/ground_truth/{dataset}',
            '--output', str(output_path / 'evaluation_results' / dataset),
            '--dataset', dataset
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"✅ Evaluation completed for {dataset}")
            logger.info(f"Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Evaluation failed for {dataset}: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate Cellpose3 denoising results")
    parser.add_argument("--config", type=str, default="config_denoising.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str,
                       help="Override output directory from config")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override output directory if provided
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    
    logger.info("Starting Cellpose3 denoising evaluation...")
    logger.info(f"Output directory: {config['data']['output_dir']}")
    
    # Run evaluation
    success = run_evaluation(
        config['data']['output_dir'],
        config
    )
    
    if success:
        logger.info("✅ Cellpose3 denoising evaluation completed successfully!")
    else:
        logger.error("❌ Cellpose3 denoising evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
