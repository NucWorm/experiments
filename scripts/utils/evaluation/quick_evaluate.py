#!/usr/bin/env python3
"""
Quick evaluation wrapper for common centroid evaluation scenarios.

This script provides convenient shortcuts for typical evaluation tasks,
making it easier to run evaluations without remembering all the command-line arguments.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_evaluation(args):
    """Run the evaluation with the given arguments."""
    cmd = [sys.executable, "evaluate_centroids.py"] + args
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Quick evaluation wrapper for common scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard NucWorm evaluation (anisotropic thresholds)
  python quick_evaluate.py standard --gt_dir ground_truth --pred_dir predictions

  # Isotropic evaluation
  python quick_evaluate.py isotropic --gt_dir ground_truth --pred_dir predictions

  # Cross-validation evaluation
  python quick_evaluate.py cv --gt_dir ground_truth --pred_dir predictions

  # Custom threshold
  python quick_evaluate.py custom --gt_dir ground_truth --pred_dir predictions --threshold 20
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Evaluation type')
    
    # Standard evaluation (anisotropic thresholds from NucWorm benchmark)
    standard_parser = subparsers.add_parser('standard', help='Standard NucWorm evaluation with anisotropic thresholds')
    standard_parser.add_argument('--gt_dir', required=True, help='Ground truth directory')
    standard_parser.add_argument('--pred_dir', required=True, help='Predictions directory')
    standard_parser.add_argument('--output_dir', help='Output directory')
    standard_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Isotropic evaluation
    isotropic_parser = subparsers.add_parser('isotropic', help='Isotropic threshold evaluation')
    isotropic_parser.add_argument('--gt_dir', required=True, help='Ground truth directory')
    isotropic_parser.add_argument('--pred_dir', required=True, help='Predictions directory')
    isotropic_parser.add_argument('--threshold', type=float, default=15, help='Distance threshold (default: 15)')
    isotropic_parser.add_argument('--output_dir', help='Output directory')
    isotropic_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Cross-validation evaluation
    cv_parser = subparsers.add_parser('cv', help='Cross-validation evaluation')
    cv_parser.add_argument('--gt_dir', required=True, help='Ground truth directory')
    cv_parser.add_argument('--pred_dir', required=True, help='Predictions directory')
    cv_parser.add_argument('--folds', nargs='+', default=['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4'],
                          help='Fold names (default: fold_0 fold_1 fold_2 fold_3 fold_4)')
    cv_parser.add_argument('--threshold_type', choices=['isotropic', 'anisotropic'], default='anisotropic',
                          help='Threshold type (default: anisotropic)')
    cv_parser.add_argument('--threshold', type=float, default=15, help='Isotropic threshold (default: 15)')
    cv_parser.add_argument('--output_dir', help='Output directory')
    cv_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Custom evaluation
    custom_parser = subparsers.add_parser('custom', help='Custom evaluation with user-specified parameters')
    custom_parser.add_argument('--gt_dir', required=True, help='Ground truth directory')
    custom_parser.add_argument('--pred_dir', required=True, help='Predictions directory')
    custom_parser.add_argument('--threshold', type=float, help='Isotropic threshold')
    custom_parser.add_argument('--threshold_xy', type=float, help='X,Y threshold')
    custom_parser.add_argument('--threshold_z', type=float, help='Z threshold')
    custom_parser.add_argument('--dataset_thresholds', help='Dataset-specific thresholds')
    custom_parser.add_argument('--folds', nargs='+', help='Fold names')
    custom_parser.add_argument('--pred_format', choices=['npy', 'csv'], default='npy', help='Prediction format')
    custom_parser.add_argument('--output_dir', help='Output directory')
    custom_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Build evaluation arguments
    eval_args = [
        '--gt_dir', args.gt_dir,
        '--pred_dir', args.pred_dir,
        '--save_csv',
        '--save_summary'
    ]
    
    if args.output_dir:
        eval_args.extend(['--output_dir', args.output_dir])
    
    if args.verbose:
        eval_args.append('--verbose')
    
    # Add command-specific arguments
    if args.command == 'standard':
        # Standard NucWorm anisotropic thresholds
        eval_args.extend([
            '--dataset_thresholds', 'nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4',
            '--output_prefix', 'nucworm_anisotropic'
        ])
        print("Running standard NucWorm evaluation with anisotropic thresholds:")
        print("  NEJATBAKHSH20: 15px (X,Y) | 2px (Z)")
        print("  WEN20: 10px (X,Y) | 2px (Z)")
        print("  YEMINI21: 15px (X,Y) | 4px (Z)")
        
    elif args.command == 'isotropic':
        eval_args.extend([
            '--threshold', str(args.threshold),
            '--output_prefix', 'isotropic_evaluation'
        ])
        print(f"Running isotropic evaluation with threshold: {args.threshold}px")
        
    elif args.command == 'cv':
        eval_args.extend(['--folds'] + args.folds)
        
        if args.threshold_type == 'anisotropic':
            eval_args.extend([
                '--dataset_thresholds', 'nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4',
                '--output_prefix', 'cv_anisotropic'
            ])
            print("Running cross-validation evaluation with anisotropic thresholds")
        else:
            eval_args.extend([
                '--threshold', str(args.threshold),
                '--output_prefix', 'cv_isotropic'
            ])
            print(f"Running cross-validation evaluation with isotropic threshold: {args.threshold}px")
        
        print(f"Folds: {', '.join(args.folds)}")
        
    elif args.command == 'custom':
        if args.threshold:
            eval_args.extend(['--threshold', str(args.threshold)])
        elif args.threshold_xy and args.threshold_z:
            eval_args.extend(['--threshold_xy', str(args.threshold_xy), '--threshold_z', str(args.threshold_z)])
        elif args.dataset_thresholds:
            eval_args.extend(['--dataset_thresholds', args.dataset_thresholds])
        else:
            print("Error: Must specify either --threshold, --threshold_xy/--threshold_z, or --dataset_thresholds")
            return
        
        if args.folds:
            eval_args.extend(['--folds'] + args.folds)
        
        eval_args.extend(['--pred_format', args.pred_format])
        eval_args.extend(['--output_prefix', 'custom_evaluation'])
        
        print("Running custom evaluation with specified parameters")
    
    # Run the evaluation
    result = run_evaluation(eval_args)
    
    if result.returncode == 0:
        print("\n✅ Evaluation completed successfully!")
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
