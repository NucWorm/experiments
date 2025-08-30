# nnUNet Fold-Based Pipeline

This document describes the updated nnUNet pipeline that supports running all 5 folds using array jobs.

## 🎯 **Overview**

The pipeline has been updated to support running all 5 trained models (fold_0 through fold_4) in parallel using Slurm array jobs. Each fold will generate its own organized output structure.

## 📁 **Output Structure**

```
outputs/
└── nnunet/                    # Method name
    ├── fold_0/               # Fold-specific results
    │   ├── center_point/
    │   │   ├── nejatbakhsh20/
    │   │   ├── wen20/
    │   │   └── yemini21/
    │   └── nnunet_heatmaps/
    │       ├── nejatbakhsh20/
    │       ├── wen20/
    │       └── yemini21/
    ├── fold_1/
    ├── fold_2/
    ├── fold_3/
    └── fold_4/
```

## 🚀 **Usage**

### **Run All 5 Folds (Array Job)**
```bash
cd /projects/weilab/gohaina/nucworm/scripts/methods/nnunet
sbatch scripts/run_full_pipeline.slurm
```

This will:
- Submit 5 parallel array jobs (indices 0-4)
- **Only array task 0** runs H5 to TIFF conversion (prevents duplication)
- **Other array tasks** wait for TIFF files to be available
- Each job processes one fold using its corresponding model
- All jobs run simultaneously (no staggering)
- Each fold generates its own complete output structure

### **Run Individual Steps**
```bash
# Step 1: H5 to TIFF conversion (if needed)
cd /projects/weilab/gohaina/nucworm/scripts/data/vol_conversion
sbatch scripts/run_h5_to_tiff_conversion.slurm

# Step 2: nnUNet inference for all folds
cd /projects/weilab/gohaina/nucworm/scripts/methods/nnunet
sbatch scripts/run_inference.slurm

# Step 3: Centroid extraction for all folds
sbatch scripts/run_postprocess.slurm
```

## 🔧 **Technical Details**

### **Array Job Configuration**
- **Array indices**: 0-4 (corresponding to fold_0 through fold_4)
- **Model mapping**: `nnunet3d_fold_{SLURM_ARRAY_TASK_ID}_model.pth`
- **Output paths**: `{OUTPUT_BASE}/{METHOD_NAME}/fold_{SLURM_ARRAY_TASK_ID}/`

### **Environment Variables**
The scripts accept these parameters via environment variables:
- `FOLD_ID`: Array task ID (0-4)
- `FOLD_NAME`: Fold name (fold_0, fold_1, etc.)
- `MODEL_PATH`: Path to the specific model file
- `OUTPUT_BASE`: Base output directory
- `METHOD_NAME`: Method name (nnunet)

### **Resource Requirements**
- **Partition**: `weilab`
- **Memory**: 64GB per fold
- **GPU**: 1 GPU per fold
- **Time**: 24 hours per fold
- **CPUs**: 4 per fold

## 📊 **Data Flow**

```
H5 Files → TIFF Conversion → nnUNet Inference → Centroid Extraction
    ↓              ↓                ↓                    ↓
neuropal/    /scratch/gohaina/   outputs/nnunet/    outputs/nnunet/
             neuropal_as_tiff/   fold_X/heatmaps/   fold_X/center_point/
```

### **Data Locations**
- **Input**: Neuropal H5 files
- **Intermediate**: TIFF files in `/scratch/gohaina/neuropal_as_tiff/`
- **Final Outputs**: Organized by method and fold in `/outputs/nnunet/`

## 📊 **Expected Results**

After completion, you'll have:
- **5 complete fold results** with identical structure
- **Heatmaps** for each fold and dataset
- **Centroid coordinates** (.npy files) for each fold and dataset
- **Organized outputs** ready for cross-validation analysis
- **Clean separation** between intermediate and final data

## 🔍 **Monitoring**

### **Check Job Status**
```bash
squeue -u gohaina
```

### **View Logs**
```bash
# Array job logs (replace %A with job ID, %a with array task ID)
tail -f /projects/weilab/gohaina/logs/nnunet_fold_%A_%a_pipeline_*.out
tail -f /projects/weilab/gohaina/logs/nnunet_fold_%A_%a_inference_*.out
tail -f /projects/weilab/gohaina/logs/nnunet_fold_%A_%a_postprocess_*.out
```

## 🧪 **Testing**

Test scripts are available to verify the setup:
```bash
# Test single fold structure
./scripts/test_fold_structure.sh

# Test all 5 folds structure
./scripts/test_all_folds.sh
```

## 📝 **Key Changes Made**

1. **Updated Slurm scripts** to support array jobs
2. **Added fold parameters** via environment variables
3. **Modified output paths** to include method and fold organization
4. **Maintained Python script flexibility** - no hardcoded paths
5. **Created test scripts** for verification

## 🚨 **Important Notes**

- **No staggering**: All 5 folds run in parallel
- **Resource allocation**: Each fold needs its own GPU and memory
- **Output organization**: Each fold has its own complete directory structure
- **Model files**: All 5 model files must exist in the models/ directory
- **Input data**: TIFF files must be available in the data directory

## 🔗 **Next Steps**

After running all folds:
1. **Verify outputs** in all fold directories
2. **Compare results** across folds
3. **Implement cross-validation evaluation** using the organized outputs
4. **Aggregate results** for final performance metrics

## 📞 **Troubleshooting**

### **Common Issues**
1. **Model file not found**: Check that all 5 model files exist
2. **GPU not available**: Ensure `weilab` partition has enough GPU nodes
3. **Memory issues**: Increase memory allocation if needed
4. **Path issues**: Verify all paths are correct

### **Getting Help**
1. Check job logs for specific errors
2. Verify model files exist
3. Check resource availability
4. Test with individual scripts first
