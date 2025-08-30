#!/bin/bash
#SBATCH --job-name=h5_to_tiff_conversion_single
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/h5_to_tiff_conversion_single_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/h5_to_tiff_conversion_single_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohaina@bc.edu

# H5 to TIFF Conversion Single Job Script
# This script converts all neuropal H5 volumes to TIFF stacks in a single job

set -e

echo "=== Starting H5 to TIFF Conversion (All Datasets) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Current directory: $(pwd)"
echo "Current time: $(date)"

# Load miniconda module
echo "Loading miniconda module..."
module load miniconda

# Activate existing conda environment
echo "Activating existing h5_conversion environment..."
source activate h5_conversion

# Change to the vol_conversion directory
cd /projects/weilab/gohaina/nucworm/scripts/data/vol_conversion

echo "Working directory: $(pwd)"

# Check if required files exist
if [ ! -f "src/convert_h5_to_tiff.py" ]; then
    echo "Error: src/convert_h5_to_tiff.py not found!"
    exit 1
fi

# Define datasets to process
datasets=("nejatbakhsh20" "yemini21" "wen20")

# Process each dataset
for dataset in "${datasets[@]}"; do
    echo ""
    echo "=== Processing dataset: ${dataset} ==="
    
    # Create output directory for current dataset
    echo "Creating output directory for ${dataset}..."
    mkdir -p "/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/${dataset}"

    # First, do a dry run to see what will be converted
    echo "Performing dry run for ${dataset}..."
    python src/convert_h5_to_tiff.py --dry_run --dataset ${dataset}

    echo ""
    echo "Starting actual conversion for ${dataset}..."
    python src/convert_h5_to_tiff.py \
        --output_dir "/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff" \
        --dataset ${dataset}

    echo "=== Conversion Complete for ${dataset} ==="
    echo "TIFF files saved to: /projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/${dataset}/"
done

echo ""
echo "=== All H5 to TIFF Conversion Complete ==="
echo "All TIFF files saved to: /projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/"
echo "Current time: $(date)"
