# NucWorm Benchmark Scripts

This directory contains the benchmark scripts for nuclei detection methods on worm neuropal data.

## 📁 **Directory Structure**

```
scripts/
├── README.md              # This file
├── methods/               # Individual method implementations
│   ├── nnunet/           # nnUNet-based method
│   └── [future_methods]/ # Additional methods
├── data/                  # Shared data processing utilities
│   └── vol_conversion/   # H5 to TIFF conversion
└── utils/                 # Shared utilities
    ├── evaluation/       # Common evaluation tools
    └── visualization/    # Visualization utilities
```

## 🎯 **Available Methods**

### **nnUNet Method**
- **Location**: `methods/nnunet/`
- **Description**: 3D nnUNet for nuclei detection and centroid extraction
- **Status**: ✅ Ready for use
- **Quick Start**: `cd methods/nnunet && sbatch scripts/run_full_pipeline.slurm`

## 🚀 **Getting Started**

### **1. Choose a Method**
Navigate to the method directory:
```bash
cd methods/nnunet  # or your preferred method
```

### **2. Run the Pipeline**
Each method provides standardized scripts:
```bash
# Complete pipeline
sbatch scripts/run_full_pipeline.slurm

# Individual steps
sbatch scripts/run_training.slurm      # Train model
sbatch scripts/run_inference.slurm     # Generate predictions
sbatch scripts/run_postprocess.slurm   # Extract centroids
sbatch scripts/run_evaluation.slurm    # Evaluate results
```

## 📊 **Data Flow**

```
Raw H5 Files → TIFF Conversion → Method Processing → Centroid CSVs
     ↓              ↓                    ↓              ↓
  neuropal/    vol_conversion/      methods/      outputs/
```

### **Input Data**
- **Source**: Neuropal H5 files
- **Conversion**: H5 → TIFF (via `data/vol_conversion/`)
- **Location**: `/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/`

### **Output Data**
- **Heatmaps**: Intermediate predictions
- **Centroids**: Final CSV files with coordinates
- **Evaluation**: Performance metrics

## 🔧 **Method Standards**

All methods in this benchmark follow these standards:

### **Directory Structure**
```
method_name/
├── README.md              # Method documentation
├── requirements.txt       # Dependencies
├── config.yaml           # Configuration
├── src/                  # Implementation
├── models/               # Pre-trained models
└── scripts/              # Execution scripts
```

### **Script Naming**
- `run_training.slurm` - Train the model
- `run_inference.slurm` - Generate predictions
- `run_postprocess.slurm` - Extract centroids
- `run_evaluation.slurm` - Evaluate performance
- `run_full_pipeline.slurm` - Complete pipeline

### **Output Standards**
- **Heatmaps**: TIFF format in `nnunet_heatmaps/`
- **Centroids**: CSV format in `center_point/`
- **Evaluation**: JSON/CSV metrics in `evaluation_results/`

## 📈 **Evaluation**

### **Standard Metrics**
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Distance Metrics**: Average distance to nearest ground truth

### **Evaluation Scripts**
Each method provides evaluation scripts that compute standardized metrics for fair comparison.

## 🔍 **Monitoring**

### **Check Job Status**
```bash
squeue -u gohaina
```

### **View Logs**
```bash
# Method-specific logs
tail -f /projects/weilab/gohaina/logs/<method>_<step>_<jobid>.out
```

## 🚨 **Troubleshooting**

### **Common Issues**
1. **Environment setup**: Each method manages its own conda environment
2. **GPU availability**: Check partition availability with `sinfo`
3. **Memory requirements**: Adjust `--mem` parameter in Slurm scripts
4. **Path issues**: Ensure all paths are updated for new structure

### **Getting Help**
1. Check method-specific README files
2. Review Slurm job logs
3. Verify input data availability
4. Check resource requirements

## 🔗 **Adding New Methods**

To add a new method to the benchmark:

1. **Create method directory**: `mkdir methods/new_method`
2. **Follow standard structure**: Use existing methods as templates
3. **Implement required scripts**: Training, inference, post-processing, evaluation
4. **Update this README**: Add method to available methods list
5. **Test thoroughly**: Ensure all scripts work correctly

## 📞 **Contributing**

This benchmark is designed to be easily extensible. When adding new methods:
- Follow the established directory structure
- Use standardized script naming
- Implement standard evaluation metrics
- Document thoroughly in method README
- Test on all datasets

## 📝 **Citation**

If you use this benchmark, please cite the original papers for each method and the NucWorm benchmark.
