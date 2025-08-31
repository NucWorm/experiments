# Centroid Evaluation Module

This module provides comprehensive evaluation tools for nucleus detection methods that output centroid coordinates.

## Features

- **Flexible Threshold Support**: Both isotropic and anisotropic distance thresholds
- **Multiple File Formats**: Support for NPY and CSV prediction files
- **Cross-Validation**: Evaluate across multiple folds
- **Dataset-Specific Thresholds**: Different thresholds for different datasets
- **Comprehensive Metrics**: Precision, recall, F1, and distance metrics
- **Multiple Output Formats**: CSV, JSON, and human-readable summaries

## Quick Start

### Basic Evaluation

```bash
# Isotropic threshold (15 pixels in all directions)
python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions --threshold 15

# Anisotropic thresholds (15 pixels X,Y, 2 pixels Z)
python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions \
    --threshold_xy 15 --threshold_z 2
```

### Cross-Validation Evaluation

```bash
# Evaluate multiple folds
python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions \
    --folds fold_0 fold_1 fold_2 fold_3 fold_4 --threshold 15
```

### Dataset-Specific Thresholds

```bash
# Different thresholds for different datasets
python evaluate_centroids.py --gt_dir ground_truth --pred_dir predictions \
    --dataset_thresholds "nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4"
```

## Command Line Arguments

### Required Arguments

- `--gt_dir`: Directory containing ground truth CSV files
- `--pred_dir`: Directory containing prediction files
- **Threshold (choose one)**:
  - `--threshold`: Isotropic distance threshold (pixels)
  - `--threshold_xy`: X,Y distance threshold (pixels) - requires `--threshold_z`
  - `--dataset_thresholds`: Dataset-specific thresholds (see format below)

### Optional Arguments

- `--threshold_z`: Z distance threshold (pixels, required with `--threshold_xy`)
- `--folds`: Fold names to evaluate (e.g., `fold_0 fold_1 fold_2`)
- `--datasets`: Specific datasets to evaluate (default: auto-detect)
- `--pred_format`: Format of prediction files (`npy` or `csv`, default: `npy`)
- `--output_dir`: Directory to save results (default: current directory)
- `--output_prefix`: Prefix for output files (default: `evaluation`)
- `--save_summary`: Save human-readable summary to text file
- `--save_csv`: Save detailed results to CSV file
- `--save_json`: Save detailed results to JSON file
- `--verbose`: Print detailed progress information

## File Format Requirements

### Ground Truth Files
- **Format**: CSV files with space-separated values
- **Coordinates**: [Z, X, Y] format (depth, width, height)
- **Naming**: Should end with `_center.csv` or similar pattern

### Prediction Files
- **NPY Format**: NumPy arrays with shape (N, 3) containing [X, Y, Z] coordinates
- **CSV Format**: Space-separated values with [X, Y, Z] coordinates
- **Naming**: Should match ground truth base names with `_im_points.npy` or similar

## Dataset-Specific Threshold Format

The `--dataset_thresholds` argument accepts a space-separated list of dataset specifications:

```
"dataset1:xy_threshold,z_threshold dataset2:xy_threshold,z_threshold"
```

**Examples**:
- `"nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4"`
- `"dataset1:10,1 dataset2:20,3"`

For isotropic thresholds, use the same value twice:
- `"dataset1:15,15 dataset2:10,10"`

## Output Files

### CSV Results (`--save_csv`)
Detailed results for each volume with columns:
- `volume_id`, `dataset`, `fold` (if applicable)
- `gt_count`, `pred_count`
- `precision`, `recall`, `f1_score`
- `mean_distance`, `median_distance`
- `threshold_type`, `threshold_xy`, `threshold_z`

### JSON Results (`--save_json`)
Structured results including:
- Metadata (total volumes, datasets, folds, threshold type)
- Overall aggregated metrics
- Detailed volume-by-volume results

### Summary Text (`--save_summary`)
Human-readable summary with:
- Overall performance metrics
- Dataset and fold breakdowns
- Best/worst performing volumes
- Key insights

## Evaluation Metrics

### Matching Algorithm
1. **Distance Calculation**: Euclidean distance between predicted and ground truth centroids
2. **Threshold Filtering**: Only consider matches within specified distance threshold(s)
3. **Greedy Matching**: Each prediction is matched to the closest unmatched ground truth
4. **Metric Calculation**: Standard precision, recall, and F1 score

### Anisotropic Thresholds
When using anisotropic thresholds:
- **X,Y Threshold**: Maximum allowed distance in X and Y directions
- **Z Threshold**: Maximum allowed distance in Z direction
- **Match Criteria**: A prediction matches ground truth only if it's within both thresholds

## Integration with Methods

### For nnUNet Method
```bash
# Evaluate nnUNet results
python utils/evaluation/evaluate_centroids.py \
    --gt_dir /path/to/ground_truth \
    --pred_dir /path/to/nnunet/outputs/center_point \
    --threshold 15 \
    --save_csv --save_summary
```

### For Cross-Validation
```bash
# Evaluate all folds
python utils/evaluation/evaluate_centroids.py \
    --gt_dir /path/to/ground_truth \
    --pred_dir /path/to/nnunet/outputs \
    --folds fold_0 fold_1 fold_2 fold_3 fold_4 \
    --dataset_thresholds "nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4" \
    --save_csv --save_json --save_summary
```

## Examples from NucWorm Benchmark

### Standard Evaluation
```bash
# Using the thresholds from our recent evaluation
python evaluate_centroids.py \
    --gt_dir ground_truth \
    --pred_dir nnunet \
    --dataset_thresholds "nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4" \
    --save_csv --save_summary --verbose
```

### Isotropic Comparison
```bash
# Compare with isotropic thresholds
python evaluate_centroids.py \
    --gt_dir ground_truth \
    --pred_dir nnunet \
    --dataset_thresholds "nejatbakhsh20:15,15 wen20:10,10 yemini21:15,15" \
    --save_csv --save_summary
```

## Troubleshooting

### Common Issues

1. **No matching files found**:
   - Check file naming conventions
   - Verify directory structure
   - Use `--verbose` to see search patterns

2. **Coordinate format errors**:
   - Ensure ground truth is [Z, X, Y] format
   - Ensure predictions are [X, Y, Z] format (for NPY) or [X, Y, Z] (for CSV)

3. **Memory issues with large datasets**:
   - Process datasets individually using `--datasets` argument
   - Use smaller batch sizes if implementing batch processing

### Debug Mode
Use `--verbose` flag to see detailed progress information and file matching results.

## Contributing

When adding new features:
1. Maintain backward compatibility
2. Add comprehensive help text
3. Include examples in the README
4. Test with both NPY and CSV formats
5. Ensure cross-validation support
