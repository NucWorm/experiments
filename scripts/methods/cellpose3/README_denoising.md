# Cellpose3 Denoising Method for NucWorm Benchmark

This directory contains the **Cellpose3 denoising variant** for the NucWorm benchmark, which performs both denoising and segmentation on neuropal volumes.

## 🎯 **Overview**

The Cellpose3 denoising method combines denoising and segmentation capabilities to process neuropal volumes and extract neuron centroids for evaluation.

### **Key Features**
- **3D Denoising**: Uses Cellpose3's denoising model to clean up neuropal volumes
- **3D Segmentation**: Performs true 3D segmentation using 2.5D approach
- **Centroid Extraction**: Extracts center points from instance masks
- **GPU Acceleration**: Leverages CUDA for faster processing
- **Dataset-Specific Parameters**: Optimized for each neuropal dataset

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
/projects/weilab/gohaina/nucworm/outputs/cellpose3_denoising/
├── masks/             ← Instance segmentation masks
│   ├── nejatbakhsh20/
│   ├── yemini21/
│   └── wen20/
├── denoised/          ← Denoised images
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

### **Model Configuration** (`config_denoising.yaml`)
- **Model type**: `cyto3` (general cell segmentation)
- **Restore type**: `denoise_cyto3` (enables denoising)
- **GPU**: Recommended for faster processing
- **Input format**: RGB neuropal volumes (C=3, Z, Y, X)
- **Output format**: Grayscale denoised images + segmentation masks

### **Processing Parameters**
- **Preprocessing**: Same as nnUNet (clipping to [0, 60000] + z-score normalization)
- **RGB to Grayscale**: Standard weights (0.299*R + 0.587*G + 0.114*B)
- **3D Segmentation**: True 3D with anisotropy handling
- **Dataset-specific**: Diameter and anisotropy values per dataset

## 🚀 **Quick Start**

### **Run Full Pipeline**
```bash
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose3
sbatch scripts/run_denoising_pipeline.slurm
```

### **Run Individual Steps**
```bash
# Step 1: Denoising and segmentation
python src/inference_denoising.py --config config_denoising.yaml

# Step 2: Post-processing (centroid extraction)
python src/postprocess_denoising.py --config config_denoising.yaml

# Step 3: Evaluation
python src/evaluate_denoising.py --config config_denoising.yaml
```

## 📁 **File Structure**

```
cellpose3/
├── config_denoising.yaml          ← Denoising configuration
├── src/
│   ├── inference_denoising.py     ← Denoising inference script
│   ├── postprocess_denoising.py   ← Post-processing script
│   └── evaluate_denoising.py      ← Evaluation script
├── scripts/
│   └── run_denoising_pipeline.slurm ← Slurm job script
└── README_denoising.md            ← This file
```

## 🔧 **Technical Details**

### **Architecture**
- **Model**: Cellpose3 denoising model (cyto3 + denoise_cyto3)
- **Input**: 3D TIFF volumes in (C=3, Z, Y, X) format (RGB neuropal data)
- **Data handling**: Transposes to (Z, C, Y, X) then converts RGB to grayscale
- **Output**: Denoised images + Instance masks → Centroids (NPY files)
- **Processing**: True 3D denoising and segmentation (2.5D approach)
- **Preprocessing**: Same as nnUNet (clipping to [0, 60000] + z-score normalization)
- **GPU**: Recommended for faster processing

### **Dataset-Specific Parameters**
- **nejatbakhsh20**: diameter=15, anisotropy=7.14
- **wen20**: diameter=10, anisotropy=4.69  
- **yemini21**: diameter=15, anisotropy=3.57

### **Output Format**
- **Denoised images**: TIFF files with float32 precision
- **Segmentation masks**: TIFF files with uint16 labels
- **Centroids**: NPY files with (Z, Y, X) coordinates
- **Naming**: Matches nnUNet format for compatibility

## 📈 **Performance Considerations**

- **GPU acceleration** significantly improves processing speed
- **Memory requirements** scale with volume size
- **Denoising** adds computational overhead but improves segmentation quality
- **3D processing** is more memory-intensive than 2D slice-by-slice

## 🔍 **Quality Control**

- **Logging**: Comprehensive logging for debugging
- **Error handling**: Robust error handling and recovery
- **Validation**: Input/output validation at each step
- **Progress tracking**: Progress bars for long-running operations

## 📊 **Expected Results**

The pipeline should produce:
- **Denoised images** for visual inspection
- **Segmentation masks** with instance labels
- **Centroid coordinates** in (Z, Y, X) format
- **Evaluation metrics** when run with ground truth

## 🆚 **Comparison with Standard Cellpose3**

| Feature | Standard Cellpose3 | Denoising Cellpose3 |
|---------|-------------------|---------------------|
| **Model** | `models.Cellpose` | `denoise.CellposeDenoiseModel` |
| **Output** | Segmentation only | Denoising + Segmentation |
| **Quality** | Good | Enhanced (denoised) |
| **Speed** | Faster | Slower (denoising overhead) |
| **Memory** | Lower | Higher (denoised images) |
| **Use Case** | Clean data | Noisy data |

## 🎯 **When to Use Denoising Variant**

- **Noisy neuropal data** that benefits from denoising
- **Quality over speed** requirements
- **Research purposes** to compare denoised vs. original results
- **Data preprocessing** for downstream analysis

## 🔧 **Troubleshooting**

### **Common Issues**
1. **GPU memory**: Reduce batch size or use CPU
2. **Long processing time**: Enable GPU acceleration
3. **File not found**: Check input directory structure
4. **Permission errors**: Ensure write access to output directory

### **Debug Mode**
```bash
# Run with debug logging
python src/inference_denoising.py --config config_denoising.yaml --debug
```

## 📚 **References**

- [Cellpose3 Documentation](https://cellpose.readthedocs.io/)
- [NucWorm Benchmark](https://github.com/weilab/nucworm)
- [Neuropal Data](https://www.nature.com/articles/nature12354)

---

**Status**: ✅ Ready for use  
**Last Updated**: September 2025  
**Maintainer**: NucWorm Team
