#!/bin/bash
#SBATCH --job-name=nnunet_postprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/centroid_extraction_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/centroid_extraction_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohaina@bc.edu

# Centroid Extraction Slurm Script
# This script extracts centroids from nnUNet heatmaps and saves them as CSVs

set -e

echo "=== Starting Centroid Extraction ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Current directory: $(pwd)"
echo "Current time: $(date)"

# Load miniconda module
echo "Loading miniconda module..."
module load miniconda

# Create and activate conda environment
echo "Setting up conda environment..."
if conda env list | grep -q "wormid_nnunet"; then
    echo "Environment wormid_nnunet already exists, activating..."
    conda activate wormid_nnunet
    echo "Updating packages..."
    pip install -r requirements.txt
else
    echo "Creating new environment wormid_nnunet..."
    conda create -n wormid_nnunet python=3.9 -y
    conda activate wormid_nnunet
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
fi

# Change to the wormid_nnunet directory
cd /projects/weilab/gohaina/nucworm/scripts/methods/nnunet

echo "Working directory: $(pwd)"

# Check if required files exist
if [ ! -f "src/postprocess.py" ]; then
    echo "Error: src/postprocess.py not found!"
    exit 1
fi

# Create output directories
echo "Creating output directories..."
mkdir -p /projects/weilab/gohaina/nucworm/outputs/center_point/{nejatbakhsh20,yemini21,wen20}

# Define datasets to process
datasets=("nejatbakhsh20" "yemini21" "wen20")

# Process each dataset
for dataset in "${datasets[@]}"; do
    echo ""
    echo "=== Processing dataset: ${dataset} ==="
    
    heatmap_dir="/projects/weilab/gohaina/nnunet_heatmaps/${dataset}"
    output_dir="/projects/weilab/gohaina/nucworm/outputs/center_point/${dataset}"
    
    # Check if heatmap directory exists and has files
    if [ ! -d "$heatmap_dir" ]; then
        echo "Warning: Heatmap directory $heatmap_dir does not exist, skipping..."
        continue
    fi
    
    heatmap_count=$(find "$heatmap_dir" -name "*.tiff" -o -name "*.tif" | wc -l)
    if [ "$heatmap_count" -eq 0 ]; then
        echo "Warning: No heatmap files found in $heatmap_dir, skipping..."
        continue
    fi
    
    echo "Found $heatmap_count heatmap files in $heatmap_dir"
    echo "Extracting centroids..."
    
    # Run centroid extraction on the entire folder
    echo "Running centroid extraction on folder: $heatmap_dir"
    python src/postprocess.py \
        --input_folder "$heatmap_dir" \
        --output_folder "$output_dir" \
        --min_distance 5 \
        --threshold_abs 0.1
    
    echo "Completed processing ${dataset}"
    echo "Centroids saved to: $output_dir"
done

echo ""
echo "=== Centroid Extraction Complete ==="
echo "All centroids saved to: /projects/weilab/gohaina/nucworm/outputs/center_point/"
echo "Current time: $(date)"
echo ""
echo "Pipeline complete! Final outputs are CSV files with centroid coordinates."
