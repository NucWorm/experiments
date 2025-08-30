# H5 to TIFF Conversion

This directory contains scripts to convert neuropal H5 volumes to TIFF stacks for use in the NucWorm benchmark.

## 📁 **Directory Structure**

```
vol_conversion/
├── README.md              # This file
├── environment.yml        # Conda environment specification
├── src/                  # Core conversion scripts
│   ├── convert_h5_to_tiff.py
│   └── check_tiff_shape.py
├── scripts/              # Slurm execution scripts
│   └── run_h5_to_tiff_conversion.slurm
└── docs/                 # Documentation
    ├── PREFERRED_SLURM_PARAMS.md
    └── [legacy README.md]
```

## 🚀 **Quick Start**

### **Run Full Conversion Pipeline**
```bash
cd /projects/weilab/gohaina/nucworm/scripts/data/vol_conversion
sbatch scripts/run_h5_to_tiff_conversion.slurm
```

This will:
- Create `h5_conversion` conda environment
- Run 3 parallel array jobs (one per dataset)
- Convert all H5 files to TIFF format
- Save to `/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/`

## 📊 **Datasets Processed**

- **nejatbakhsh20**: 21 volumes
- **yemini21**: 10 volumes  
- **wen20**: 9 volumes

## 🔧 **Technical Details**

### **Data Format Conversion**
- **Input**: H5 files with shape `(Z, C, Y, X)`
- **Output**: Multi-page TIFF files in `(C, Z, Y, X)` format
- **Format**: uint8, no normalization needed
- **Compatibility**: Optimized for nnUNet processing

### **Resource Requirements**
- **Partition**: `weilab`
- **Memory**: 32GB per array task
- **CPUs**: 2 per array task
- **Time**: 2 hours per dataset

## 📈 **Output Structure**

```
/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/
├── nejatbakhsh20/
│   ├── 000541/                    ← Case groups
│   └── ...
├── yemini21/
│   ├── 000715/                    ← Case groups  
│   └── ...
└── wen20/
    ├── 000692/                    ← Case groups
    └── ...
```

## 🔍 **Monitoring**

### **Check Job Status**
```bash
squeue -u gohaina
```

### **View Logs**
```bash
# Monitor specific array tasks:
tail -f /projects/weilab/gohaina/logs/h5_to_tiff_conversion_<jobid>_0.out  # nejatbakhsh20
tail -f /projects/weilab/gohaina/logs/h5_to_tiff_conversion_<jobid>_1.out  # yemini21  
tail -f /projects/weilab/gohaina/logs/h5_to_tiff_conversion_<jobid>_2.out  # wen20
```

## 🚨 **Troubleshooting**

### **Common Issues**
1. **Environment creation fails**: Check conda availability and environment.yml validity
2. **Memory issues**: Increase memory allocation in slurm script
3. **Time limits**: Increase time allocation if conversion is slow

### **Environment Management**
- **Remove environment**: `conda env remove -n h5_conversion`
- **Recreate environment**: Scripts automatically recreate if needed
- **Check environment**: `conda env list`

## 🔗 **Integration**

This conversion step is automatically called by:
- `methods/nnunet/scripts/run_full_pipeline.slurm`

The converted TIFF files are then used by all benchmark methods for nuclei detection.

## 📝 **Notes**

- This is a shared data processing utility used by all benchmark methods
- Output files are stored in the shared outputs directory
- Environment is automatically managed by the scripts
- Array jobs provide efficient parallel processing


