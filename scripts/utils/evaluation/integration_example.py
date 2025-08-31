#!/usr/bin/env python3
"""
Example showing how to integrate the new evaluation tools with existing methods.

This demonstrates how the nnUNet method (or any other method) can use the
comprehensive evaluation tools instead of maintaining its own evaluation code.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the utils directory to the path so we can import the evaluation module
utils_dir = Path(__file__).parent.parent
sys.path.insert(0, str(utils_dir))

from evaluation.evaluate_centroids import evaluate_single_volume, print_summary


def run_evaluation_with_new_tools(gt_dir, pred_dir, output_dir, dataset_name="unknown"):
    """
    Example of how to use the new evaluation tools in a method-specific script.
    
    This replaces the need for method-specific evaluation code while providing
    more comprehensive metrics and flexibility.
    """
    
    # Option 1: Use the command-line interface (recommended for most cases)
    cmd = [
        sys.executable, 
        "utils/evaluation/evaluate_centroids.py",
        "--gt_dir", gt_dir,
        "--pred_dir", pred_dir,
        "--output_dir", output_dir,
        "--threshold", "15",  # or use dataset-specific thresholds
        "--save_csv",
        "--save_summary",
        "--verbose"
    ]
    
    print(f"Running evaluation: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode == 0:
        print("✅ Evaluation completed successfully!")
        return True
    else:
        print("❌ Evaluation failed!")
        return False


def run_evaluation_programmatically(gt_dir, pred_dir, output_dir):
    """
    Example of how to use the evaluation functions directly in Python code.
    
    This is useful when you need more control over the evaluation process
    or want to integrate it into a larger workflow.
    """
    
    # Import the evaluation functions
    from evaluation.evaluate_centroids import (
        find_matching_files, 
        evaluate_single_volume,
        print_summary
    )
    
    results = []
    
    # Find matching files
    matches = find_matching_files(gt_dir, pred_dir, pred_format='npy')
    
    if not matches:
        print("No matching files found!")
        return False
    
    print(f"Found {len(matches)} matching file pairs")
    
    # Evaluate each volume
    for gt_file, pred_file, base_id in matches:
        result = evaluate_single_volume(
            gt_file, pred_file, base_id, 
            threshold_xy=15, threshold_z=15,  # Isotropic threshold
            pred_format='npy'
        )
        results.append(result)
        print(f"  {base_id}: P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1_score']:.3f}")
    
    # Print summary
    print_summary(results)
    
    # Save results (you could add custom saving logic here)
    import pandas as pd
    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "custom_evaluation_results.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    return True


def main():
    """
    Example usage of the integration approaches.
    """
    
    # Example paths (these would be actual paths in a real implementation)
    gt_dir = "/path/to/ground_truth"
    pred_dir = "/path/to/predictions"
    output_dir = "/path/to/output"
    
    print("=== Integration Example ===")
    print("This shows how to integrate the new evaluation tools with existing methods.")
    print()
    
    print("Option 1: Command-line interface (recommended)")
    print("- Use subprocess to call the evaluation script")
    print("- Provides full functionality with minimal code changes")
    print("- Easy to maintain and update")
    print()
    
    print("Option 2: Programmatic interface")
    print("- Import and use evaluation functions directly")
    print("- More control over the evaluation process")
    print("- Better for complex workflows")
    print()
    
    # Uncomment to test with actual data:
    # run_evaluation_with_new_tools(gt_dir, pred_dir, output_dir)
    # run_evaluation_programmatically(gt_dir, pred_dir, output_dir)


if __name__ == "__main__":
    main()
