#!/bin/bash
#SBATCH --job-name=h5_to_tiff_conversion
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/projects/weilab/gohaina/logs/h5_to_tiff_conversion_%A_%a.out
#SBATCH --error=/projects/weilab/gohaina/logs/h5_to_tiff_conversion_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohaina@bc.edu

# H5 to TIFF Conversion Array Job Script
# This script converts neuropal H5 volumes to TIFF stacks using array jobs
# Array indices: 0=nejatbakhsh20, 1=yemini21, 2=wen20

set -e

# Define dataset names based on array index
datasets=("nejatbakhsh20" "yemini21" "wen20")
current_dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "=== Starting H5 to TIFF Conversion for ${current_dataset} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${current_dataset}"
echo "Node: $SLURM_NODELIST"
echo "Current directory: $(pwd)"
echo "Current time: $(date)"

# Load miniconda module
echo "Loading miniconda module..."
module load miniconda

# Create and activate conda environment from file
echo "Setting up conda environment from environment.yml..."
if conda env list | grep -q "h5_conversion"; then
    echo "Environment h5_conversion already exists, updating packages..."
    conda env update -f environment.yml -n h5_conversion
else
    echo "Creating new environment h5_conversion..."
    conda env create -f environment.yml -n h5_conversion
fi

echo "Activating h5_conversion environment..."
source activate h5_conversion

# Change to the vol_conversion directory
cd /projects/weilab/gohaina/nucworm/scripts/data/vol_conversion

echo "Working directory: $(pwd)"

# Check if required files exist
if [ ! -f "src/convert_h5_to_tiff.py" ]; then
    echo "Error: src/convert_h5_to_tiff.py not found!"
    exit 1
fi

# Create output directory for current dataset
echo "Creating output directory for ${current_dataset}..."
mkdir -p "/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/${current_dataset}"

# First, do a dry run to see what will be converted
echo "Performing dry run for ${current_dataset}..."
python src/convert_h5_to_tiff.py --dry_run --dataset ${current_dataset}

echo ""
echo "Starting actual conversion for ${current_dataset}..."
python src/convert_h5_to_tiff.py \
    --output_dir "/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff" \
    --dataset ${current_dataset}

echo "=== Conversion Complete for ${current_dataset} ==="
echo "TIFF files saved to: /projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/${current_dataset}/"
echo "Current time: $(date)"
