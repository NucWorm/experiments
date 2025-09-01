#!/usr/bin/env python3
"""
Cellpose3 evaluation script for NucWorm benchmark.

This script evaluates Cellpose3 results using the NucWorm evaluation framework.
"""

import os
import sys
import argparse
import yaml
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_evaluation(gt_dir, pred_dir, output_dir, config):
    """Run evaluation using the NucWorm evaluation framework."""
    
    # Path to the evaluation script
    eval_script = Path(__file__).parent.parent.parent.parent.parent / 'utils' / 'evaluation' / 'evaluate_centroids.py'
    
    if not eval_script.exists():
        logger.error(f"Evaluation script not found at {eval_script}")
        return False
    
    # Standard NucWorm evaluation parameters
    cmd = [
        'python', str(eval_script),
        '--gt_dir', str(gt_dir),
        '--pred_dir', str(pred_dir),
        '--dataset_thresholds', 'nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4',
        '--output_dir', str(output_dir),
        '--output_prefix', 'cellpose3_evaluation',
        '--save_csv',
        '--save_json',
        '--save_summary',
        '--verbose'
    ]
    
    logger.info(f"Running evaluation command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Evaluation completed successfully!")
        logger.info(f"STDOUT: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed with return code {e.returncode}")
        logger.error(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Cellpose3 evaluation for NucWorm benchmark")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Ground truth directory')
    parser.add_argument('--pred_dir', type=str,
                       help='Predictions directory (overrides config)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for evaluation results (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup paths
    if args.pred_dir:
        pred_dir = Path(args.pred_dir)
    else:
        pred_dir = Path(config['data']['output_dir']) / 'center_point'
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config['data']['output_dir']) / 'evaluation_results'
    
    gt_dir = Path(args.gt_dir)
    
    # Verify directories exist
    if not gt_dir.exists():
        logger.error(f"Ground truth directory not found: {gt_dir}")
        sys.exit(1)
    
    if not pred_dir.exists():
        logger.error(f"Predictions directory not found: {pred_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Ground truth directory: {gt_dir}")
    logger.info(f"Predictions directory: {pred_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Run evaluation
    success = run_evaluation(gt_dir, pred_dir, output_dir, config)
    
    if success:
        logger.info("Cellpose3 evaluation completed successfully!")
    else:
        logger.error("Cellpose3 evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
