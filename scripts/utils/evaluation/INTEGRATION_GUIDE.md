# Evaluation Module Integration Guide

This document explains how to integrate the new comprehensive evaluation tools into existing methods and workflows.

## What Was Created

### Core Files
- **`evaluate_centroids.py`**: Main evaluation script with full functionality
- **`quick_evaluate.py`**: Convenient wrapper for common evaluation scenarios
- **`__init__.py`**: Python module for programmatic access
- **`README.md`**: Comprehensive documentation
- **`integration_example.py`**: Examples of how to integrate with existing methods

### Key Features
- **Flexible Thresholds**: Isotropic and anisotropic distance thresholds
- **Multiple Formats**: Support for NPY and CSV prediction files
- **Cross-Validation**: Evaluate across multiple folds
- **Dataset-Specific**: Different thresholds for different datasets
- **Multiple Outputs**: CSV, JSON, and human-readable summaries
- **No Hard-coded Paths**: All paths and thresholds are configurable via command line

## Integration Approaches

### 1. Command-Line Integration (Recommended)

Replace existing evaluation calls with calls to the new evaluation script:

```python
# Old approach (method-specific evaluation)
# python src/evaluate.py --gt_dir gt --pred_dir pred --threshold 15

# New approach (comprehensive evaluation)
subprocess.run([
    "python", "utils/evaluation/evaluate_centroids.py",
    "--gt_dir", gt_dir,
    "--pred_dir", pred_dir,
    "--threshold", "15",
    "--save_csv", "--save_summary"
])
```

### 2. Programmatic Integration

Import and use evaluation functions directly:

```python
from utils.evaluation import evaluate_single_volume, find_matching_files

# Use the functions in your existing code
matches = find_matching_files(gt_dir, pred_dir)
for gt_file, pred_file, base_id in matches:
    result = evaluate_single_volume(gt_file, pred_file, base_id, 15, 15)
    # Process result...
```

### 3. Quick Evaluation Wrapper

Use the quick evaluation wrapper for common scenarios:

```python
# Standard NucWorm evaluation
subprocess.run([
    "python", "utils/evaluation/quick_evaluate.py", "standard",
    "--gt_dir", gt_dir, "--pred_dir", pred_dir
])
```

## Migration from Existing Evaluation

### For nnUNet Method

The existing `methods/nnunet/src/evaluate.py` can be replaced or enhanced:

**Option A: Replace entirely**
```bash
# Old
python methods/nnunet/src/evaluate.py --gt_dir gt --pred_dir pred

# New
python utils/evaluation/evaluate_centroids.py --gt_dir gt --pred_dir pred --threshold 15
```

**Option B: Enhance existing script**
```python
# In methods/nnunet/src/evaluate.py
import subprocess
import sys

def run_comprehensive_evaluation(gt_dir, pred_dir, output_dir):
    """Use the comprehensive evaluation tools."""
    cmd = [
        sys.executable,
        "../../../utils/evaluation/evaluate_centroids.py",
        "--gt_dir", gt_dir,
        "--pred_dir", pred_dir,
        "--output_dir", output_dir,
        "--dataset_thresholds", "nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4",
        "--save_csv", "--save_summary"
    ]
    return subprocess.run(cmd).returncode == 0
```

## Slurm Script Integration

### Minimal Risk Integration

To integrate with existing Slurm scripts with minimal risk:

```bash
# In methods/nnunet/scripts/run_evaluation.slurm

# Add this section before the existing evaluation
echo "Running comprehensive evaluation..."

# Use the new evaluation tools
python ../../../utils/evaluation/quick_evaluate.py standard \
    --gt_dir "$gt_dir" \
    --pred_dir "$pred_dir" \
    --output_dir "$output_dir" \
    --verbose

# Keep existing evaluation as backup (commented out)
# python src/evaluate.py --gt_dir "$gt_dir" --pred_dir "$pred_dir" --output_dir "$output_dir"
```

### Full Integration

Replace the evaluation section entirely:

```bash
# Replace the evaluation loop in run_evaluation.slurm
for dataset in "${datasets[@]}"; do
    echo "=== Evaluating dataset: ${dataset} ==="
    
    gt_dir="/projects/weilab/dataset/NucWorm/neuropal/${dataset}"
    pred_dir="/projects/weilab/gohaina/nucworm/outputs/center_point/${dataset}"
    output_dir="/projects/weilab/gohaina/nucworm/outputs/evaluation_results/${dataset}"
    
    # Use comprehensive evaluation
    python ../../../utils/evaluation/evaluate_centroids.py \
        --gt_dir "$gt_dir" \
        --pred_dir "$pred_dir" \
        --output_dir "$output_dir" \
        --dataset_thresholds "nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4" \
        --save_csv --save_summary --verbose
done
```

## Benefits of Integration

### For Method Developers
- **Less Code to Maintain**: No need to implement evaluation logic
- **Consistent Metrics**: All methods use the same evaluation approach
- **Advanced Features**: Access to anisotropic thresholds, cross-validation, etc.
- **Better Outputs**: Multiple output formats and comprehensive summaries

### For Benchmark Users
- **Standardized Evaluation**: Consistent evaluation across all methods
- **Flexible Thresholds**: Easy to experiment with different thresholds
- **Comprehensive Results**: Detailed metrics and visualizations
- **Easy Comparison**: Standardized output formats for method comparison

## Testing the Integration

### Test with Existing Data
```bash
# Test the new evaluation tools with your existing data
cd /path/to/your/experiments/scripts

# Test basic functionality
python utils/evaluation/evaluate_centroids.py \
    --gt_dir /path/to/ground_truth \
    --pred_dir /path/to/predictions \
    --threshold 15 \
    --save_csv --save_summary --verbose

# Test quick evaluation wrapper
python utils/evaluation/quick_evaluate.py standard \
    --gt_dir /path/to/ground_truth \
    --pred_dir /path/to/predictions
```

### Validate Results
Compare results from the new evaluation tools with your existing evaluation to ensure consistency.

## Future Extensions

The evaluation module is designed to be easily extensible:

- **New Metrics**: Add new evaluation metrics to the core functions
- **New Formats**: Support additional file formats
- **New Thresholds**: Implement different matching algorithms
- **Visualizations**: Add plotting and visualization capabilities

## Support

For questions or issues with the evaluation module:
1. Check the comprehensive README.md
2. Review the integration examples
3. Test with your specific data and requirements
4. Modify the scripts as needed for your use case

The evaluation tools are designed to be generalizable and should work with any centroid-based nucleus detection method.
