#!/usr/bin/env python3
"""
Evaluation script to compare ground truth cell centroids with predicted centroids.
This script evaluates the WormID nnUNet model performance on a volume-by-volume basis.
"""

import os
import glob
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import json


def load_ground_truth(csv_path):
    """
    Load ground truth centroids from CSV file.
    Expected format: CSV with implicit z, y, x coordinate ordering (space-separated).
    """
    try:
        # Try to read CSV with different possible column names
        df = pd.read_csv(csv_path, header=None, sep='\s+')  # No header, space-separated
        
        # Check if we have exactly 3 columns with numeric data
        if len(df.columns) == 3:
            # Assume implicit z, y, x ordering (depth, height, width)
            coords = df.values.astype(np.float32)
            print(f"  Loaded {len(coords)} ground truth centroids from implicit z,y,x columns")
            return coords
        else:
            # Fallback: try with header detection
            df_with_header = pd.read_csv(csv_path)
            
            # Look for coordinate columns (common variations)
            coord_cols = []
            for col in df_with_header.columns:
                col_lower = col.lower()
                if any(coord in col_lower for coord in ['x', 'y', 'z', 'coord']):
                    coord_cols.append(col)
            
            if len(coord_cols) >= 3:
                # Use first 3 coordinate columns found
                coords = df_with_header[coord_cols[:3]].values.astype(np.float32)
                print(f"  Loaded {len(coords)} ground truth centroids from columns: {coord_cols[:3]}")
                return coords
            else:
                # If no obvious coordinate columns, assume first 3 numeric columns
                numeric_cols = df_with_header.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 3:
                    coords = df_with_header[numeric_cols[:3]].values.astype(np.float32)
                    print(f"  Loaded {len(coords)} ground truth centroids from numeric columns: {numeric_cols[:3]}")
                    return coords
                else:
                    print(f"  Warning: Could not identify coordinate columns in {csv_path}")
                    return np.array([])
                
    except Exception as e:
        print(f"  Error loading {csv_path}: {e}")
        return np.array([])


def load_predictions(npy_path):
    """
    Load predicted centroids from NPY file.
    """
    try:
        coords = np.load(npy_path)
        print(f"  Loaded {len(coords)} predicted centroids")
        return coords
    except Exception as e:
        print(f"  Error loading {npy_path}: {e}")
        return np.array([])


def evaluate_predictions(predicted_coords, gt_coords, threshold=15):
    """
    Evaluate predicted cell centroids against ground truth.
    
    Args:
        predicted_coords: n×3 array of predicted coordinates
        gt_coords: m×3 array of ground truth coordinates
        threshold: Distance threshold for matching (pixels)
    
    Returns:
        precision, recall, f1_score
    """
    if len(predicted_coords) == 0 and len(gt_coords) == 0:
        return 1.0, 1.0, 1.0  # Perfect if both are empty
    elif len(predicted_coords) == 0:
        return 0.0, 0.0, 0.0  # No predictions, all GT are FN
    elif len(gt_coords) == 0:
        return 0.0, 1.0, 0.0  # No GT, all predictions are FP
    
    # Calculate pairwise distances
    from scipy.spatial.distance import cdist
    distances = cdist(predicted_coords, gt_coords, metric='euclidean')
    
    # Find matches within threshold
    matches = distances <= threshold
    
    # Count true positives, false positives, false negatives
    tp = 0
    matched_gt = set()
    
    for pred_idx in range(len(predicted_coords)):
        # Find closest unmatched ground truth within threshold
        gt_matches = np.where(matches[pred_idx])[0]
        unmatched_gt = [gt_idx for gt_idx in gt_matches if gt_idx not in matched_gt]
        
        if unmatched_gt:
            # Find closest unmatched GT
            closest_gt = min(unmatched_gt, key=lambda gt_idx: distances[pred_idx, gt_idx])
            tp += 1
            matched_gt.add(closest_gt)
    
    fp = len(predicted_coords) - tp  # Unmatched predictions
    fn = len(gt_coords) - tp         # Unmatched ground truth
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def find_matching_files(gt_dir, pred_dir):
    """
    Find matching pairs of ground truth CSV and prediction NPY files.
    """
    matches = []
    
    # Get all ground truth CSV files
    gt_files = glob.glob(os.path.join(gt_dir, "**/*.csv"), recursive=True)
    
    for gt_file in gt_files:
        # Extract the base identifier (e.g., "000541_sub-20190928-11_ses-20190928_ophys")
        gt_name = os.path.basename(gt_file)
        
        # Remove "_center.csv" suffix to get base identifier
        if gt_name.endswith("_center.csv"):
            base_id = gt_name.replace("_center.csv", "")  # Remove "_center.csv" more robustly
        else:
            base_id = os.path.splitext(gt_name)[0]
        
        # Look for corresponding prediction file (search recursively)
        pred_pattern = os.path.join(pred_dir, "**", f"{base_id}_im_points.npy")
        print(f"  Looking for pattern: {pred_pattern}")
        
        # Try both recursive glob and direct search
        pred_files = glob.glob(pred_pattern, recursive=True)
        if not pred_files:
            # Fallback: search all subdirectories manually
            for root, dirs, files in os.walk(pred_dir):
                for file in files:
                    if file == f"{base_id}_im_points.npy":
                        pred_files = [os.path.join(root, file)]
                        break
                if pred_files:
                    break
        
        print(f"  Found {len(pred_files)} prediction files: {pred_files}")
        if pred_files:
            # Use the first match found
            pred_file = pred_files[0]
            matches.append((gt_file, pred_file, base_id))
            print(f"  Found prediction: {os.path.basename(pred_file)}")
        else:
            print(f"  No prediction found for: {base_id}")
    
    return matches


def main():
    parser = argparse.ArgumentParser(description="Evaluate WormID nnUNet predictions against ground truth")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth CSV files")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing predicted NPY files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Distance threshold for matching (pixels, default: 15)")
    parser.add_argument("--dataset", type=str, default="unknown",
                        help="Dataset name for output files")
    
    args = parser.parse_args()
    
    print(f"=== WormID nnUNet Evaluation ===")
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Prediction directory: {args.pred_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Distance threshold: {args.threshold} pixels")
    print(f"Dataset: {args.dataset}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find matching files
    print("Finding matching ground truth and prediction files...")
    matches = find_matching_files(args.gt_dir, args.pred_dir)
    
    if not matches:
        print("No matching files found!")
        return
    
    print(f"Found {len(matches)} matching file pairs")
    print()
    
    # Evaluate each pair
    results = []
    
    print("Evaluating predictions...")
    for gt_file, pred_file, base_id in tqdm(matches, desc="Processing volumes"):
        print(f"\n--- Evaluating: {base_id} ---")
        
        # Load files
        gt_coords = load_ground_truth(gt_file)
        pred_coords = load_predictions(pred_file)
        
        # Convert ground truth from z,y,x to x,y,z order to match predictions
        if len(gt_coords) > 0 and gt_coords.shape[1] == 3:
            # Transpose from (z, y, x) to (x, y, z)
            gt_coords = gt_coords[:, [2, 1, 0]]  # [z, y, x] -> [x, y, z]
            print(f"  Converted ground truth coordinates from z,y,x to x,y,z order")
        
        # Evaluate
        precision, recall, f1 = evaluate_predictions(pred_coords, gt_coords, args.threshold)
        
        # Store results
        result = {
            'volume_id': base_id,
            'gt_file': gt_file,
            'pred_file': pred_file,
            'gt_count': len(gt_coords),
            'pred_count': len(pred_coords),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'threshold': args.threshold
        }
        results.append(result)
        
        print(f"  Results: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    # Calculate overall metrics
    if results:
        overall_precision = np.mean([r['precision'] for r in results])
        overall_recall = np.mean([r['recall'] for r in results])
        overall_f1 = np.mean([r['f1_score'] for r in results])
        
        print(f"\n=== Overall Results ===")
        print(f"Total volumes evaluated: {len(results)}")
        print(f"Average Precision: {overall_precision:.3f}")
        print(f"Average Recall: {overall_recall:.3f}")
        print(f"Average F1 Score: {overall_f1:.3f}")
        
        # Save detailed results
        results_file = os.path.join(args.output_dir, f"{args.dataset}_evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'dataset': args.dataset,
                'threshold': args.threshold,
                'overall_metrics': {
                    'precision': overall_precision,
                    'recall': overall_recall,
                    'f1_score': overall_f1
                },
                'volume_results': results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Save summary CSV
        summary_file = os.path.join(args.output_dir, f"{args.dataset}_evaluation_summary.csv")
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary CSV saved to: {summary_file}")


if __name__ == "__main__":
    main()
