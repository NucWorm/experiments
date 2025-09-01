# Cellpose3 Method for Nuclei Detection

This directory contains the Cellpose3-based method for nuclei detection and centroid extraction from 3D microscopy volumes in the NucWorm benchmark.

## 📁 **Directory Structure**

```
cellpose3/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── config.yaml           # Configuration
├── src/                  # Core implementation
│   ├── inference.py      # Inference script (generates instance masks)
│   ├── postprocess.py    # Post-processing (extracts centroids)
│   └── evaluate.py       # Evaluation script
├── models/               # Pre-trained models (if any)
└── scripts/              # Slurm execution scripts
    ├── run_inference.slurm
    ├── run_postprocess.slurm
    ├── run_evaluation.slurm
    └── run_full_pipeline.slurm
```

## 🚀 **Quick Start**

### **Option 1: Run Complete Pipeline**
```bash
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose3
sbatch scripts/run_full_pipeline.slurm
```

### **Option 2: Run Individual Steps**
```bash
# Step 1: Cellpose3 inference (generates instance masks)
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose3
sbatch scripts/run_inference.slurm

# Step 2: Centroid extraction (generates NPY files)
sbatch scripts/run_postprocess.slurm

# Step 3: Evaluation (optional)
sbatch scripts/run_evaluation.slurm
```

## 🔧 **Method Details**

### **Architecture**
- **Model**: Cellpose3 standard model (cyto3) for cell segmentation
- **Input**: 3D TIFF volumes in (C=3, Z, Y, X) format (RGB neuropal data)
- **Data handling**: Transposes to (Z, C, Y, X) then converts RGB to grayscale using standard weights (0.299*R + 0.587*G + 0.114*B)
- **Output**: Instance masks → Centroids (NPY files)
- **Processing**: True 3D segmentation (2.5D approach with YX, ZY, ZX slice analysis)
- **Preprocessing**: Same as nnUNet (clipping to [0, 60000] + z-score normalization)
- **GPU**: Recommended for faster processing
- **Note**: Uses segmentation only (no denoising) for optimal performance

### **Datasets Processed**
- `nejatbakhsh20` (21 volumes)
- `yemini21` (10 volumes)
- `wen20` (9 volumes)

### **Resource Requirements**
- **Inference**: 32GB RAM, 1 GPU, 12 hours
- **Post-processing**: 16GB RAM, 4 hours
- **Evaluation**: 16GB RAM, 2 hours

## 📊 **Input/Output Structure**

### **Inputs**
```
/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/
├── nejatbakhsh20/     ← 21 TIFF files (Z=21, C=3, Y, X)
├── yemini21/          ← 10 TIFF files (Z=39, C=3, Y, X)
└── wen20/             ← 9 TIFF files (Z=28, C=3, Y, X)
```

### **Outputs**
```
/projects/weilab/gohaina/nucworm/outputs/cellpose3/
├── masks/             ← Instance segmentation masks
│   ├── nejatbakhsh20/
│   ├── yemini21/
│   └── wen20/
├── center_point/      ← Centroid coordinates (NPY files)
│   ├── nejatbakhsh20/
│   ├── yemini21/
│   └── wen20/
└── evaluation_results/ ← Evaluation metrics (if run)
```

## ⚙️ **Configuration**

### **Model Configuration** (`config.yaml`)
- **Model type**: `cyto3` (general cell segmentation)
- **Restore type**: `denoise_cyto3` (Cellpose3 denoise model)
- **GPU**: Enabled for faster processing
- **Channels**: `[0,0]` (grayscale images)

### **Segmentation Parameters**
- **Diameter**: Dataset-specific (based on 3μm neuron size)
  - **YEMINI21**: 15 pixels (X,Y directions)
  - **NEJATBAKHSH20**: 15 pixels (X,Y directions)
  - **WEN20**: 10 pixels (X,Y directions)
- **Flow threshold**: 0.4 (flow error threshold)
- **Cell probability threshold**: 0.0 (cell probability threshold)
- **Minimum size**: 15 pixels (minimum cell size)
- **Anisotropy**: Dataset-specific (based on Z vs XY resolution)
  - **YEMINI21**: 3.57 (0.75μm Z / 0.21μm XY)
  - **NEJATBAKHSH20**: 7.14 (1.5μm Z / 0.21μm XY)
  - **WEN20**: 4.69 (1.5μm Z / 0.32μm XY)

### **Environment Setup**
The scripts automatically create and manage a conda environment:
- Environment name: `cellpose3`
- Python version: 3.9
- Dependencies: See `requirements.txt`

## 📈 **Evaluation Metrics**

The evaluation script computes:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Distance-based metrics**: Average distance to nearest ground truth

### **Standard NucWorm Thresholds**
- **NEJATBAKHSH20**: 15px (X,Y) | 2px (Z)
- **WEN20**: 10px (X,Y) | 2px (Z)
- **YEMINI21**: 15px (X,Y) | 4px (Z)

## 🔍 **Monitoring**

### **Check Job Status**
```bash
squeue -u gohaina
```

### **View Logs**
```bash
# Inference logs
tail -f /projects/weilab/gohaina/logs/cellpose3_inference_<jobid>.out

# Post-processing logs
tail -f /projects/weilab/gohaina/logs/cellpose3_postprocess_<jobid>.out

# Evaluation logs
tail -f /projects/weilab/gohaina/logs/cellpose3_evaluation_<jobid>.out

# Full pipeline logs
tail -f /projects/weilab/gohaina/logs/cellpose3_pipeline_<jobid>.out
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Environment not found**: Scripts will create `cellpose3` conda environment
2. **GPU not available**: Ensure `--partition=weilab` has GPU nodes
3. **Memory issues**: Increase `--mem` parameter if needed
4. **Cellpose3 installation**: Ensure Cellpose3 >= 3.0.0 is installed

### **Check Job Logs**
```bash
# Check error logs for specific issues
cat /projects/weilab/gohaina/logs/cellpose3_<step>_<jobid>.err
```

### **Manual Environment Setup**
```bash
# Create environment manually
conda create -n cellpose3 python=3.9 -y
conda activate cellpose3
pip install -r requirements.txt

# Verify installation
python -c "import cellpose; print(f'Cellpose version: {cellpose.__version__}')"
```

## 🔧 **Customization**

### **Adjusting Segmentation Parameters**
Edit `config.yaml` to modify:
- **Diameter**: Expected cell size in pixels
- **Flow threshold**: Lower values = more sensitive detection
- **Cell probability threshold**: Higher values = more conservative detection
- **Anisotropy**: Adjust based on your data's Z-axis resolution

### **Processing Specific Datasets**
```bash
# Process only one dataset
python src/inference.py --config config.yaml --dataset nejatbakhsh20
python src/postprocess.py --config config.yaml --dataset nejatbakhsh20
```

### **Different Centroid Extraction Methods**
```bash
# Use center of mass instead of regionprops
python src/postprocess.py --config config.yaml --method center_of_mass
```

## 📞 **Next Steps**

After successful completion:
1. **Verify outputs** in all target directories
2. **Check NPY formats** for centroid coordinates
3. **Validate results** against expected data ranges
4. **Compare with nnUNet results** in the benchmark
5. **Run evaluation** if ground truth is available

## 🔗 **Related Methods**

This method is part of the NucWorm benchmark. Other methods can be found in:
- `/projects/weilab/gohaina/nucworm/scripts/methods/nnunet/`

## 📝 **Citation**

If you use this method, please cite:
- **Cellpose3**: Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1), 100-106.
- **NucWorm benchmark**: [Your benchmark paper citation]

## 🔬 **Technical Notes**

### **3D Processing**
- Cellpose3 uses 2.5D segmentation: computes flows on YX, ZY, and ZX slices, then averages and runs dynamics in 3D
- Dataset-specific anisotropy values account for different Z vs XY resolution
- True 3D context is preserved, unlike slice-by-slice approaches
- Handles anisotropic volumes by upsampling Z-axis based on anisotropy parameter

### **Centroid Extraction**
- Uses `scikit-image.regionprops` for accurate centroid calculation
- Alternative center-of-mass method available
- Outputs coordinates in [Z, Y, X] format for NucWorm compatibility

### **Performance Considerations**
- GPU acceleration significantly improves processing speed
- Memory requirements scale with volume size
- Consider processing large volumes in chunks if memory is limited
