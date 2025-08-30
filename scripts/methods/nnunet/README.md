# nnUNet Method for Nuclei Detection

This directory contains the nnUNet-based method for nuclei detection and centroid extraction from 3D microscopy volumes.

## 📁 **Directory Structure**

```
nnunet/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── config.yaml           # Training configuration
├── src/                  # Core implementation
│   ├── train.py          # Training script
│   ├── inference.py      # Inference script (generates heatmaps)
│   ├── postprocess.py    # Post-processing (extracts centroids)
│   └── evaluate.py       # Evaluation script
├── models/               # Pre-trained models
│   └── nnunet3d_final_model.pth
└── scripts/              # Slurm execution scripts
    ├── run_training.slurm
    ├── run_inference.slurm
    ├── run_postprocess.slurm
    ├── run_evaluation.slurm
    └── run_full_pipeline.slurm
```

## 🚀 **Quick Start**

### **Option 1: Run Complete Pipeline**
```bash
cd /projects/weilab/gohaina/nucworm/scripts/methods/nnunet
sbatch scripts/run_full_pipeline.slurm
```

### **Option 2: Run Individual Steps**
```bash
# Step 1: nnUNet inference (generates heatmaps)
cd /projects/weilab/gohaina/nucworm/scripts/methods/nnunet
sbatch scripts/run_inference.slurm

# Step 2: Centroid extraction (generates CSVs)
sbatch scripts/run_postprocess.slurm

# Step 3: Evaluation (optional)
sbatch scripts/run_evaluation.slurm
```

## 🔧 **Method Details**

### **Architecture**
- **Model**: 3D nnUNet for nuclei detection
- **Input**: 3D TIFF volumes
- **Output**: Heatmaps → Centroids (CSV files)
- **Patch Size**: 32×96×64 voxels
- **Stride**: 16×48×32 for sliding window inference

### **Datasets Processed**
- `nejatbakhsh20` (21 volumes)
- `yemini21` (10 volumes)
- `wen20` (9 volumes)

### **Resource Requirements**
- **Training**: 64GB RAM, 1 GPU, 48 hours
- **Inference**: 64GB RAM, 1 GPU, 12 hours
- **Post-processing**: 32GB RAM, 4 hours
- **Evaluation**: 32GB RAM, 2 hours

## 📊 **Input/Output Structure**

### **Inputs**
```
/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/
├── nejatbakhsh20/     ← 21 TIFF files
├── yemini21/          ← 10 TIFF files
└── wen20/             ← 9 TIFF files
```

### **Outputs**
```
/projects/weilab/gohaina/nnunet_heatmaps/     ← Heatmaps
├── nejatbakhsh20/
├── yemini21/
└── wen20/

/projects/weilab/gohaina/center_point/        ← Centroid CSVs
├── nejatbakhsh20/
├── yemini21/
└── wen20/
```

## ⚙️ **Configuration**

### **Training Configuration** (`config.yaml`)
- Patch size: [32, 96, 64]
- Stride: [16, 48, 32]
- Batch size: 2
- Learning rate: 0.0001
- Epochs: 1000

### **Environment Setup**
The scripts automatically create and manage a conda environment:
- Environment name: `wormid_nnunet`
- Python version: 3.9
- Dependencies: See `requirements.txt`

## 📈 **Evaluation Metrics**

The evaluation script computes:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Distance-based metrics**: Average distance to nearest ground truth

## 🔍 **Monitoring**

### **Check Job Status**
```bash
squeue -u gohaina
```

### **View Logs**
```bash
# Training logs
tail -f /projects/weilab/gohaina/logs/nnunet_training_<jobid>.out

# Inference logs
tail -f /projects/weilab/gohaina/logs/nnunet_inference_<jobid>.out

# Post-processing logs
tail -f /projects/weilab/gohaina/logs/nnunet_postprocess_<jobid>.out

# Evaluation logs
tail -f /projects/weilab/gohaina/logs/nnunet_evaluation_<jobid>.out
```

## 🚨 **Troubleshooting**

### **Common Issues**
1. **Environment not found**: Scripts will create `wormid_nnunet` conda environment
2. **Model file missing**: Check path to `models/nnunet3d_final_model.pth`
3. **GPU not available**: Ensure `--partition=weilab` has GPU nodes
4. **Memory issues**: Increase `--mem` parameter if needed

### **Check Job Logs**
```bash
# Check error logs for specific issues
cat /projects/weilab/gohaina/logs/nnunet_<step>_<jobid>.err
```

## 📞 **Next Steps**

After successful completion:
1. **Verify outputs** in all target directories
2. **Check CSV formats** for centroid coordinates
3. **Validate results** against expected data ranges
4. **Compare with other methods** in the benchmark

## 🔗 **Related Methods**

This method is part of the NucWorm benchmark. Other methods can be found in:
- `/projects/weilab/gohaina/nucworm/scripts/methods/[other_method]/`

## 📝 **Citation**

If you use this method, please cite the original nnUNet paper and the NucWorm benchmark.