# Data Processing Utilities

This directory contains shared data processing utilities for the NucWorm benchmark.

## 📁 **Directory Structure**

```
data/
├── README.md              # This file
├── .gitignore            # Git ignore rules for data files
└── vol_conversion/       # H5 to TIFF conversion utilities
    ├── README.md
    ├── environment.yml
    ├── src/              # Core conversion scripts
    ├── scripts/          # Slurm execution scripts
    └── docs/             # Documentations
```

## 🔧 **Available Utilities**

### **H5 to TIFF Conversion** (`vol_conversion/`)
- **Purpose**: Convert neuropal H5 volumes to TIFF stacks
- **Input**: H5 files from neuropal datasets
- **Output**: TIFF files for use by benchmark methods
- **Usage**: `cd vol_conversion && sbatch scripts/run_h5_to_tiff_conversion.slurm`

## 📊 **Data Flow**

```
Raw H5 Files → vol_conversion → TIFF Files → Benchmark Methods
     ↓              ↓              ↓              ↓
  neuropal/    data/vol_conversion/  outputs/    methods/
```

## 🚨 **Important Notes**

### **Git Ignore Rules**
This directory uses `.gitignore` to exclude:
- Large data files (H5, TIFF, model files)
- Processing outputs
- Temporary files
- Log files
- Environment files

### **Output Locations**
All processed data is stored in:
- `/projects/weilab/gohaina/nucworm/outputs/data/`

### **Shared Resources**
These utilities are used by all benchmark methods and should be:
- **Efficient**: Optimized for large-scale processing
- **Reliable**: Robust error handling and logging
- **Documented**: Clear usage instructions
- **Maintainable**: Clean, well-structured code

## 🔗 **Integration**

Data processing utilities are automatically called by:
- `methods/*/scripts/run_full_pipeline.slurm`

Each method can also call these utilities individually as needed.

## 📝 **Adding New Utilities**

When adding new data processing utilities:
1. Create a new subdirectory (e.g., `new_utility/`)
2. Follow the standard structure: `src/`, `scripts/`, `docs/`
3. Update this README
4. Add appropriate `.gitignore` rules
5. Test thoroughly before integration


