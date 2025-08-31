#!/usr/bin/env python3
"""
Comprehensive centroid evaluation script for nucleus detection methods.

This script provides flexible evaluation of predicted centroids against ground truth
with support for both isotropic and anisotropic distance thresholds. It can handle
multiple datasets, cross-validation folds, and various output formats.

Features:
- Isotropic and anisotropic threshold support
- Cross-validation fold evaluation
- Multiple dataset support
- Flexible file format handling (CSV, NPY)
- Comprehensive metrics and visualizations
- Volume-wise and aggregate reporting

Example usage:
    # Basic evaluation with isotropic threshold
    python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions --threshold 15

    # Anisotropic thresholds
    python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions \
        --threshold_xy 15 --threshold_z 2

    # Cross-validation evaluation
    python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions \
        --folds fold_0 fold_1 fold_2 --threshold 15

    # Dataset-specific thresholds
    python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions \
        --dataset_thresholds nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


def load_ground_truth_csv(csv_path: str) -> np.ndarray:
    """
    Load ground truth centroids from CSV file.
    
    Expected format: space-separated values with [Z, X, Y] coordinates.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        numpy array of shape (N, 3) with [Z, X, Y] coordinates
    """
    try:
        df = pd.read_csv(csv_path, sep=' ', header=None)
        if df.shape[1] != 3:
            raise ValueError(f"Expected 3 columns, got {df.shape[1]}")
        return df.values.astype(np.float32)
    except Exception as e:
        print(f"Error loading ground truth {csv_path}: {e}")
        return np.array([])


def load_predictions_npy(npy_path: str) -> np.ndarray:
    """
    Load predicted centroids from NPY file.
    
    Expected format: numpy array of shape (N, 3) with [X, Y, Z] coordinates.
    Converts to [Z, X, Y] format to match ground truth.
    
    Args:
        npy_path: Path to NPY file
        
    Returns:
        numpy array of shape (N, 3) with [Z, X, Y] coordinates
    """
    try:
        pred = np.load(npy_path)
        if pred.ndim != 2 or pred.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {pred.shape}")
        
        # Convert from [X, Y, Z] to [Z, X, Y] to match ground truth format
        pred_reordered = pred[:, [2, 0, 1]]
        return pred_reordered.astype(np.float32)
    except Exception as e:
        print(f"Error loading predictions {npy_path}: {e}")
        return np.array([])


def load_predictions_csv(csv_path: str) -> np.ndarray:
    """
    Load predicted centroids from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        numpy array of shape (N, 3) with [Z, X, Y] coordinates
    """
    try:
        df = pd.read_csv(csv_path, sep=' ', header=None)
        if df.shape[1] != 3:
            raise ValueError(f"Expected 3 columns, got {df.shape[1]}")
        return df.values.astype(np.float32)
    except Exception as e:
        print(f"Error loading predictions {csv_path}: {e}")
        return np.array([])


def evaluate_isotropic(predicted_coords: np.ndarray, gt_coords: np.ndarray, 
                      threshold: float) -> Tuple[float, float, float, float, float]:
    """
    Evaluate predictions using isotropic distance threshold.
    
    Args:
        predicted_coords: (N, 3) array of predicted coordinates [Z, X, Y]
        gt_coords: (M, 3) array of ground truth coordinates [Z, X, Y]
        threshold: Distance threshold for matching (pixels)
        
    Returns:
        precision, recall, f1_score, mean_distance, median_distance
    """
    if len(predicted_coords) == 0 and len(gt_coords) == 0:
        return 1.0, 1.0, 1.0, 0.0, 0.0
    elif len(predicted_coords) == 0:
        return 0.0, 0.0, 0.0, np.inf, np.inf
    elif len(gt_coords) == 0:
        return 0.0, 1.0, 0.0, np.inf, np.inf
    
    # Calculate pairwise distances
    distances = cdist(predicted_coords, gt_coords, metric='euclidean')
    
    # Find matches within threshold
    matches = distances <= threshold
    
    # Count true positives using greedy matching
    tp = 0
    matched_gt = set()
    matched_distances = []
    
    for pred_idx in range(len(predicted_coords)):
        # Find closest unmatched ground truth within threshold
        gt_matches = np.where(matches[pred_idx])[0]
        unmatched_gt = [gt_idx for gt_idx in gt_matches if gt_idx not in matched_gt]
        
        if unmatched_gt:
            # Find closest unmatched GT
            closest_gt = min(unmatched_gt, key=lambda gt_idx: distances[pred_idx, gt_idx])
            tp += 1
            matched_gt.add(closest_gt)
            matched_distances.append(distances[pred_idx, closest_gt])
    
    fp = len(predicted_coords) - tp
    fn = len(gt_coords) - tp
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate distance metrics
    if matched_distances:
        mean_distance = np.mean(matched_distances)
        median_distance = np.median(matched_distances)
    else:
        mean_distance = np.inf
        median_distance = np.inf
    
    return precision, recall, f1, mean_distance, median_distance


def evaluate_anisotropic(predicted_coords: np.ndarray, gt_coords: np.ndarray,
                        threshold_xy: float, threshold_z: float) -> Tuple[float, float, float, float, float]:
    """
    Evaluate predictions using anisotropic distance thresholds.
    
    Args:
        predicted_coords: (N, 3) array of predicted coordinates [Z, X, Y]
        gt_coords: (M, 3) array of ground truth coordinates [Z, X, Y]
        threshold_xy: Distance threshold for X,Y directions (pixels)
        threshold_z: Distance threshold for Z direction (pixels)
        
    Returns:
        precision, recall, f1_score, mean_distance, median_distance
    """
    if len(predicted_coords) == 0 and len(gt_coords) == 0:
        return 1.0, 1.0, 1.0, 0.0, 0.0
    elif len(predicted_coords) == 0:
        return 0.0, 0.0, 0.0, np.inf, np.inf
    elif len(gt_coords) == 0:
        return 0.0, 1.0, 0.0, np.inf, np.inf
    
    # Calculate anisotropic distances
    z_dist = np.abs(predicted_coords[:, 0:1] - gt_coords[:, 0:1].T)  # Z distance
    x_dist = np.abs(predicted_coords[:, 1:2] - gt_coords[:, 1:2].T)  # X distance  
    y_dist = np.abs(predicted_coords[:, 2:3] - gt_coords[:, 2:3].T)  # Y distance
    
    # Check if within thresholds for each dimension
    z_within = z_dist <= threshold_z
    xy_within = (x_dist <= threshold_xy) & (y_dist <= threshold_xy)
    
    # A match requires both Z and XY to be within thresholds
    bool_mask = z_within & xy_within
    
    tp = 0
    matched_gt = set()
    matched_distances = []
    
    for i in range(len(predicted_coords)):
        neighbors = bool_mask[i].nonzero()[0]
        
        if len(neighbors) == 0:
            continue  # No valid matches
        else:
            # Find closest match among valid neighbors using Euclidean distance
            dist = np.sqrt(z_dist[i, neighbors]**2 + x_dist[i, neighbors]**2 + y_dist[i, neighbors]**2)
            closest_idx = neighbors[np.argmin(dist)]
            if closest_idx not in matched_gt:
                tp += 1
                matched_gt.add(closest_idx)
                matched_distances.append(np.min(dist))
    
    fp = len(predicted_coords) - tp
    fn = len(gt_coords) - tp
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate distance metrics
    if matched_distances:
        mean_distance = np.mean(matched_distances)
        median_distance = np.median(matched_distances)
    else:
        mean_distance = np.inf
        median_distance = np.inf
    
    return precision, recall, f1, mean_distance, median_distance


def find_matching_files(gt_dir: str, pred_dir: str, pred_format: str = 'npy') -> List[Tuple[str, str, str]]:
    """
    Find matching pairs of ground truth and prediction files.
    
    Args:
        gt_dir: Directory containing ground truth CSV files
        pred_dir: Directory containing prediction files
        pred_format: Format of prediction files ('npy' or 'csv')
        
    Returns:
        List of tuples (gt_file, pred_file, base_id)
    """
    matches = []
    
    # Get all ground truth CSV files
    gt_files = glob.glob(os.path.join(gt_dir, "**/*.csv"), recursive=True)
    
    for gt_file in gt_files:
        # Extract base identifier
        gt_name = os.path.basename(gt_file)
        
        # Remove common suffixes to get base identifier
        base_id = gt_name
        for suffix in ["_center.csv", "_ophys_center.csv", ".csv"]:
            if base_id.endswith(suffix):
                base_id = base_id[:-len(suffix)]
                break
        
        # Look for corresponding prediction file
        if pred_format == 'npy':
            pred_pattern = os.path.join(pred_dir, "**", f"{base_id}_im_points.npy")
        else:  # csv
            pred_pattern = os.path.join(pred_dir, "**", f"{base_id}.csv")
        
        pred_files = glob.glob(pred_pattern, recursive=True)
        
        if pred_files:
            matches.append((gt_file, pred_files[0], base_id))
        else:
            # Try alternative naming patterns
            alt_patterns = [
                os.path.join(pred_dir, "**", f"{base_id}.{pred_format}"),
                os.path.join(pred_dir, "**", f"{base_id}_points.{pred_format}"),
            ]
            
            for pattern in alt_patterns:
                alt_files = glob.glob(pattern, recursive=True)
                if alt_files:
                    matches.append((gt_file, alt_files[0], base_id))
                    break
    
    return matches


def parse_dataset_thresholds(threshold_str: str) -> Dict[str, Tuple[float, float]]:
    """
    Parse dataset-specific threshold string.
    
    Args:
        threshold_str: String like "dataset1:15,2 dataset2:10,2"
        
    Returns:
        Dictionary mapping dataset names to (threshold_xy, threshold_z) tuples
    """
    thresholds = {}
    if not threshold_str:
        return thresholds
    
    for item in threshold_str.split():
        if ':' in item:
            dataset, thresh_vals = item.split(':', 1)
            if ',' in thresh_vals:
                xy, z = map(float, thresh_vals.split(','))
                thresholds[dataset] = (xy, z)
            else:
                # Single value for isotropic threshold
                val = float(thresh_vals)
                thresholds[dataset] = (val, val)
    
    return thresholds


def evaluate_single_volume(gt_file: str, pred_file: str, base_id: str,
                          threshold_xy: float, threshold_z: float,
                          pred_format: str = 'npy') -> Dict:
    """
    Evaluate a single volume.
    
    Args:
        gt_file: Path to ground truth file
        pred_file: Path to prediction file
        base_id: Base identifier for the volume
        threshold_xy: X,Y threshold
        threshold_z: Z threshold
        pred_format: Format of prediction file
        
    Returns:
        Dictionary with evaluation results
    """
    # Load data
    gt_coords = load_ground_truth_csv(gt_file)
    
    if pred_format == 'npy':
        pred_coords = load_predictions_npy(pred_file)
    else:
        pred_coords = load_predictions_csv(pred_file)
    
    # Determine if using isotropic or anisotropic evaluation
    if threshold_xy == threshold_z:
        precision, recall, f1, mean_dist, median_dist = evaluate_isotropic(
            pred_coords, gt_coords, threshold_xy
        )
        threshold_type = "isotropic"
        threshold_used = threshold_xy
    else:
        precision, recall, f1, mean_dist, median_dist = evaluate_anisotropic(
            pred_coords, gt_coords, threshold_xy, threshold_z
        )
        threshold_type = "anisotropic"
        threshold_used = f"{threshold_xy},{threshold_z}"
    
    return {
        'volume_id': base_id,
        'gt_file': gt_file,
        'pred_file': pred_file,
        'gt_count': len(gt_coords),
        'pred_count': len(pred_coords),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_distance': mean_dist,
        'median_distance': median_dist,
        'threshold_type': threshold_type,
        'threshold_xy': threshold_xy,
        'threshold_z': threshold_z,
        'threshold_used': threshold_used
    }


def print_summary(results: List[Dict], output_file: Optional[str] = None):
    """
    Print comprehensive evaluation summary.
    
    Args:
        results: List of evaluation results
        output_file: Optional file to save summary to
    """
    if not results:
        print("No results to summarize!")
        return
    
    df = pd.DataFrame(results)
    
    # Overall summary
    print("\n" + "="*80)
    print("📊 CENTROID EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n🎯 OVERALL PERFORMANCE ({len(df)} volumes):")
    print(f"   Precision: {df['precision'].mean():.1%} | Recall: {df['recall'].mean():.1%} | F1: {df['f1_score'].mean():.1%}")
    print(f"   Mean Distance: {df['mean_distance'].mean():.2f} pixels")
    print(f"   Total GT: {df['gt_count'].sum():,} | Total Pred: {df['pred_count'].sum():,}")
    
    # Dataset breakdown if available
    if 'dataset' in df.columns:
        print(f"\n📁 DATASET BREAKDOWN:")
        for dataset in sorted(df['dataset'].unique()):
            dataset_results = df[df['dataset'] == dataset]
            print(f"   {dataset.upper()}: P={dataset_results['precision'].mean():.1%} | R={dataset_results['recall'].mean():.1%} | F1={dataset_results['f1_score'].mean():.1%}")
    
    # Fold breakdown if available
    if 'fold' in df.columns:
        print(f"\n🔢 FOLD BREAKDOWN:")
        for fold in sorted(df['fold'].unique()):
            fold_results = df[df['fold'] == fold]
            print(f"   {fold}: P={fold_results['precision'].mean():.1%} | R={fold_results['recall'].mean():.1%} | F1={fold_results['f1_score'].mean():.1%}")
    
    # Best/worst performers
    best_f1 = df.loc[df['f1_score'].idxmax()]
    worst_f1 = df.loc[df['f1_score'].idxmin()]
    
    print(f"\n🏆 BEST PERFORMER: {best_f1['volume_id']} - F1: {best_f1['f1_score']:.1%}")
    print(f"📉 WORST PERFORMER: {worst_f1['volume_id']} - F1: {worst_f1['f1_score']:.1%}")
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("📊 CENTROID EVALUATION RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"🎯 OVERALL PERFORMANCE ({len(df)} volumes):\n")
            f.write(f"   Precision: {df['precision'].mean():.1%} | Recall: {df['recall'].mean():.1%} | F1: {df['f1_score'].mean():.1%}\n")
            f.write(f"   Mean Distance: {df['mean_distance'].mean():.2f} pixels\n")
            f.write(f"   Total GT: {df['gt_count'].sum():,} | Total Pred: {df['pred_count'].sum():,}\n")
        print(f"\n💾 Summary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive centroid evaluation for nucleus detection methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with isotropic threshold
  python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions --threshold 15

  # Anisotropic thresholds
  python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions \\
      --threshold_xy 15 --threshold_z 2

  # Cross-validation evaluation
  python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions \\
      --folds fold_0 fold_1 fold_2 --threshold 15

  # Dataset-specific thresholds (example from NucWorm benchmark)
  python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions \\
      --dataset_thresholds "nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4"

  # CSV prediction format
  python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions \\
      --threshold 15 --pred_format csv
        """
    )
    
    # Required arguments
    parser.add_argument("--gt_dir", required=True, help="Directory containing ground truth CSV files")
    parser.add_argument("--pred_dir", required=True, help="Directory containing prediction files")
    
    # Threshold arguments
    threshold_group = parser.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument("--threshold", type=float, help="Isotropic distance threshold (pixels)")
    threshold_group.add_argument("--threshold_xy", type=float, help="X,Y distance threshold (pixels)")
    threshold_group.add_argument("--dataset_thresholds", type=str, 
                                help="Dataset-specific thresholds: 'dataset1:xy,z dataset2:xy,z'")
    
    parser.add_argument("--threshold_z", type=float, help="Z distance threshold (pixels, required with --threshold_xy)")
    
    # Optional arguments
    parser.add_argument("--folds", nargs='+', help="Fold names to evaluate (e.g., fold_0 fold_1)")
    parser.add_argument("--datasets", nargs='+', help="Dataset names to evaluate")
    parser.add_argument("--pred_format", choices=['npy', 'csv'], default='npy', 
                       help="Format of prediction files (default: npy)")
    parser.add_argument("--output_dir", help="Directory to save results (default: current directory)")
    parser.add_argument("--output_prefix", default="evaluation", 
                       help="Prefix for output files (default: evaluation)")
    parser.add_argument("--save_summary", action='store_true', 
                       help="Save readable summary to text file")
    parser.add_argument("--save_csv", action='store_true', 
                       help="Save detailed results to CSV file")
    parser.add_argument("--save_json", action='store_true', 
                       help="Save detailed results to JSON file")
    parser.add_argument("--verbose", action='store_true', 
                       help="Print detailed progress information")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.threshold_xy is not None and args.threshold_z is None:
        parser.error("--threshold_z is required when using --threshold_xy")
    
    if args.dataset_thresholds and (args.threshold_xy is not None or args.threshold is not None):
        parser.error("--dataset_thresholds cannot be used with --threshold or --threshold_xy")
    
    # Set up output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = args.output_dir
    else:
        output_dir = "."
    
    # Parse dataset-specific thresholds
    dataset_thresholds = {}
    if args.dataset_thresholds:
        dataset_thresholds = parse_dataset_thresholds(args.dataset_thresholds)
        if args.verbose:
            print("Dataset-specific thresholds:")
            for dataset, (xy, z) in dataset_thresholds.items():
                print(f"  {dataset}: XY={xy}, Z={z}")
    
    # Determine evaluation directories
    if args.folds:
        # Cross-validation evaluation
        eval_dirs = []
        for fold in args.folds:
            fold_pred_dir = os.path.join(args.pred_dir, fold, "center_point")
            if os.path.exists(fold_pred_dir):
                eval_dirs.append((fold, fold_pred_dir))
            else:
                print(f"Warning: Fold directory {fold_pred_dir} not found")
    else:
        # Single directory evaluation
        eval_dirs = [("", args.pred_dir)]
    
    all_results = []
    
    # Evaluate each directory
    for fold_name, pred_dir in eval_dirs:
        if args.verbose:
            print(f"\n{'='*60}")
            if fold_name:
                print(f"Processing FOLD: {fold_name}")
            else:
                print("Processing predictions")
            print(f"{'='*60}")
        
        # Get datasets to evaluate
        if args.datasets:
            datasets_to_eval = args.datasets
        else:
            # Auto-detect datasets from ground truth directory
            datasets_to_eval = [d for d in os.listdir(args.gt_dir) 
                              if os.path.isdir(os.path.join(args.gt_dir, d))]
        
        for dataset in datasets_to_eval:
            if args.verbose:
                print(f"\nProcessing dataset: {dataset}")
            
            # Set thresholds for this dataset
            if dataset_thresholds and dataset in dataset_thresholds:
                threshold_xy, threshold_z = dataset_thresholds[dataset]
                if args.verbose:
                    print(f"  Using dataset-specific thresholds: XY={threshold_xy}, Z={threshold_z}")
            elif args.threshold_xy is not None:
                threshold_xy, threshold_z = args.threshold_xy, args.threshold_z
                if args.verbose:
                    print(f"  Using anisotropic thresholds: XY={threshold_xy}, Z={threshold_z}")
            else:
                threshold_xy = threshold_z = args.threshold
                if args.verbose:
                    print(f"  Using isotropic threshold: {threshold_xy}")
            
            # Set up directories
            gt_dataset_dir = os.path.join(args.gt_dir, dataset)
            pred_dataset_dir = os.path.join(pred_dir, dataset)
            
            if not os.path.exists(gt_dataset_dir):
                print(f"Warning: Ground truth directory {gt_dataset_dir} not found")
                continue
            
            if not os.path.exists(pred_dataset_dir):
                print(f"Warning: Prediction directory {pred_dataset_dir} not found")
                continue
            
            # Find matching files
            matches = find_matching_files(gt_dataset_dir, pred_dataset_dir, args.pred_format)
            
            if not matches:
                print(f"  No matching files found for dataset {dataset}")
                continue
            
            if args.verbose:
                print(f"  Found {len(matches)} matching file pairs")
            
            # Evaluate each volume
            for gt_file, pred_file, base_id in tqdm(matches, desc=f"  {dataset}", disable=not args.verbose):
                result = evaluate_single_volume(
                    gt_file, pred_file, base_id, threshold_xy, threshold_z, args.pred_format
                )
                result['dataset'] = dataset
                if fold_name:
                    result['fold'] = fold_name
                all_results.append(result)
                
                if args.verbose:
                    print(f"    {base_id}: P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1_score']:.3f}")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Save CSV
        if args.save_csv or not any([args.save_json, args.save_summary]):
            csv_file = os.path.join(output_dir, f"{args.output_prefix}_results.csv")
            df.to_csv(csv_file, index=False)
            print(f"\n💾 Results saved to: {csv_file}")
        
        # Save JSON
        if args.save_json:
            json_file = os.path.join(output_dir, f"{args.output_prefix}_results.json")
            df.to_dict('records')
            with open(json_file, 'w') as f:
                json.dump({
                    'metadata': {
                        'total_volumes': len(df),
                        'datasets': sorted(df['dataset'].unique()) if 'dataset' in df.columns else [],
                        'folds': sorted(df['fold'].unique()) if 'fold' in df.columns else [],
                        'threshold_type': df['threshold_type'].iloc[0] if len(df) > 0 else 'unknown'
                    },
                    'overall_metrics': {
                        'precision': float(df['precision'].mean()),
                        'recall': float(df['recall'].mean()),
                        'f1_score': float(df['f1_score'].mean()),
                        'mean_distance': float(df['mean_distance'].mean())
                    },
                    'volume_results': df.to_dict('records')
                }, f, indent=2)
            print(f"💾 JSON results saved to: {json_file}")
        
        # Print and save summary
        summary_file = None
        if args.save_summary:
            summary_file = os.path.join(output_dir, f"{args.output_prefix}_summary.txt")
        
        print_summary(all_results, summary_file)
        
    else:
        print("No results generated!")


if __name__ == "__main__":
    main()
