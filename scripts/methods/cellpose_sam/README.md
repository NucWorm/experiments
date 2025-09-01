# Cellpose-SAM Method for Nuclei Detection

This directory contains the Cellpose-SAM-based method for nuclei detection and centroid extraction from 3D microscopy volumes in the NucWorm benchmark.

## 📁 **Directory Structure**

```
cellpose_sam/
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
    ├── run_full_pipeline.slurm
    └── run_single_volume_test.slurm
```

## 🚀 **Quick Start**

### **Option 1: Run Complete Pipeline**
```bash
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose_sam
sbatch scripts/run_full_pipeline.slurm
```

### **Option 2: Run Individual Steps**
```bash
# Step 1: Cellpose-SAM inference (generates instance masks)
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose_sam
sbatch scripts/run_inference.slurm

# Step 2: Centroid extraction (generates NPY files)
sbatch scripts/run_postprocess.slurm

# Step 3: Evaluation (optional)
sbatch scripts/run_evaluation.slurm
```

### **Option 3: Test on Single Volume**
```bash
# Test Cellpose-SAM on a single volume first
sbatch scripts/run_single_volume_test.slurm
```

## 🔧 **Method Details**

### **Architecture**
- **Model**: Cellpose-SAM (SAM transformer backbone + Cellpose framework)
- **Input**: 3D TIFF volumes in (C=3, Z, Y, X) format (RGB neuropal data)
- **Data handling**: Transposes to (Z, C, Y, X) and uses RGB channels directly (channel invariant)
- **Output**: Instance masks → Centroids (NPY files)
- **Processing**: True 3D segmentation (2.5D approach with YX, ZY, ZX slice analysis)
- **Preprocessing**: Same as nnUNet (clipping to [0, 60000] + z-score normalization)
- **GPU**: Recommended for faster processing
- **Channel Invariance**: Uses RGB channels directly without grayscale conversion

### **Key Advantages Over Cellpose3**
- **Superior Generalization**: Approaches human-consensus bounds vs inter-human agreement
- **Channel Invariance**: Uses RGB channels directly without conversion
- **Robustness**: Invariant to channel order, cell size variations, noise, blur, downsampling
- **Speed**: Despite 50x more parameters, it's the fastest method tested
- **No Preprocessing**: Runs natively without resizing or channel specification

### **Datasets Processed**
- `nejatbakhsh20` (21 volumes)
- `yemini21` (10 volumes)
- `wen20` (9 volumes)

### **Resource Requirements**
- **Inference**: 32GB RAM, 1 GPU, 12 hours
- **Post-processing**: 16GB RAM, 4 hours
- **Evaluation**: 16GB RAM, 2 hours
- **Full Pipeline**: 320GB RAM, 1 GPU, 18 hours

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
/projects/weilab/gohaina/nucworm/outputs/cellpose_sam/
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
- **Model type**: `cellpose_sam` (SAM transformer backbone + Cellpose framework)
- **GPU**: Enabled for faster processing
- **Channel handling**: `use_rgb_directly: true` (leverages channel invariance)
- **Input size**: 256×256 (vs SAM's 1024×1024)
- **Patch size**: 8×8 (vs SAM's 16×16)

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
- Environment name: `cellpose_sam`
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
tail -f /projects/weilab/gohaina/logs/cellpose_sam_inference_<jobid>.out

# Post-processing logs
tail -f /projects/weilab/gohaina/logs/cellpose_sam_postprocess_<jobid>.out

# Evaluation logs
tail -f /projects/weilab/gohaina/logs/cellpose_sam_evaluation_<jobid>.out

# Full pipeline logs
tail -f /projects/weilab/gohaina/logs/cellpose_sam_pipeline_<jobid>.out

# Test logs
tail -f /projects/weilab/gohaina/logs/cellpose_sam_test_<jobid>.out
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Cellpose-SAM model not available**: 
   - The model may not be available in the current Cellpose release
   - Scripts will automatically fall back to `cyto3` model
   - Check Cellpose version: `python -c "import cellpose; print(cellpose.__version__)"`

2. **Environment not found**: Scripts will create `cellpose_sam` conda environment

3. **GPU not available**: Ensure `--partition=weilab` has GPU nodes

4. **Memory issues**: Increase `--mem` parameter if needed

5. **Channel handling errors**: Scripts include fallback to grayscale conversion

### **Check Job Logs**
```bash
# Check error logs for specific issues
cat /projects/weilab/gohaina/logs/cellpose_sam_<step>_<jobid>.err
```

### **Manual Environment Setup**
```bash
# Create environment manually
conda create -n cellpose_sam python=3.9 -y
conda activate cellpose_sam
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

### **Channel Handling Options**
```yaml
# Use RGB channels directly (recommended for Cellpose-SAM)
data_format:
  use_rgb_directly: true
  convert_to_grayscale: false

# Or fallback to grayscale conversion
data_format:
  use_rgb_directly: false
  convert_to_grayscale: true
```

## 📞 **Next Steps**

After successful completion:
1. **Verify outputs** in all target directories
2. **Check NPY formats** for centroid coordinates
3. **Validate results** against expected data ranges
4. **Compare with Cellpose3 and nnUNet results** in the benchmark
5. **Run evaluation** if ground truth is available

## 🔗 **Related Methods**

This method is part of the NucWorm benchmark. Other methods can be found in:
- `/projects/weilab/gohaina/nucworm/scripts/methods/cellpose3/`
- `/projects/weilab/gohaina/nucworm/scripts/methods/nnunet/`

## 📝 **Citation**

If you use this method, please cite:
- **Cellpose-SAM**: Pachitariu, M., Rariden, M., & Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation. bioRxiv.
- **Cellpose**: Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1), 100-106.
- **NucWorm benchmark**: [Your benchmark paper citation]

## 🔬 **Technical Notes**

### **3D Processing**
- Cellpose-SAM uses 2.5D segmentation: computes flows on YX, ZY, and ZX slices, then averages and runs dynamics in 3D
- Dataset-specific anisotropy values account for different Z vs XY resolution
- True 3D context is preserved, unlike slice-by-slice approaches
- Handles anisotropic volumes by upsampling Z-axis based on anisotropy parameter

### **Channel Invariance**
- Cellpose-SAM is trained to be invariant to channel order
- Uses RGB channels directly without grayscale conversion
- Leverages SAM's transformer backbone for superior feature learning
- Includes fallback to grayscale conversion if needed

### **Centroid Extraction**
- Uses `scipy.ndimage.center_of_mass` for accurate centroid calculation
- Alternative center-of-mass method available
- Outputs coordinates in [Z, Y, X] format for NucWorm compatibility

### **Performance Considerations**
- GPU acceleration significantly improves processing speed
- Memory requirements scale with volume size
- Consider processing large volumes in chunks if memory is limited
- Cellpose-SAM is faster than Cellpose3 despite larger model size

## 🆕 **What's New in Cellpose-SAM**

### **Key Innovations**
1. **SAM Backbone**: Uses pretrained SAM transformer for superior generalization
2. **Channel Invariance**: Handles RGB channels directly without conversion
3. **Size Invariance**: No need to specify cell diameters
4. **Robustness**: Handles image degradations without preprocessing
5. **Speed**: Fastest method despite larger model size

### **Expected Performance Improvements**
- **Error Rate**: ~0.163 vs Cellpose3's ~0.292 (44% improvement)
- **Average Precision**: Significantly higher than Cellpose3
- **Generalization**: Much better out-of-distribution performance
- **Robustness**: Handles image degradations without additional preprocessing
