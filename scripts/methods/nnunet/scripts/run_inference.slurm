#!/bin/bash
#SBATCH --job-name=nnunet_inference
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/nnunet_inference_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/nnunet_inference_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohaina@bc.edu

# nnUNet Inference Slurm Script
# This script runs the WormID nnUNet model on all converted TIFF files to generate heatmaps

set -e

echo "=== Starting nnUNet Inference ==="
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
if [ ! -f "src/inference.py" ]; then
    echo "Error: src/inference.py not found!"
    exit 1
fi

if [ ! -f "models/nnunet3d_final_model.pth" ]; then
    echo "Error: Model file not found!"
    exit 1
fi

# Create output directories
echo "Creating output directories..."
mkdir -p /projects/weilab/gohaina/nnunet_heatmaps/{nejatbakhsh20,yemini21,wen20}

# Define datasets to process
datasets=("nejatbakhsh20" "yemini21" "wen20")

# Process each dataset
for dataset in "${datasets[@]}"; do
    echo ""
    echo "=== Processing dataset: ${dataset} ==="
    
    input_dir="/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/${dataset}"
    output_dir="/projects/weilab/gohaina/nnunet_heatmaps/${dataset}"
    
    # Check if input directory exists and has TIFF files
    if [ ! -d "$input_dir" ]; then
        echo "Warning: Input directory $input_dir does not exist, skipping..."
        continue
    fi
    
    tiff_count=$(find "$input_dir" -name "*.tiff" | wc -l)
    if [ "$tiff_count" -eq 0 ]; then
        echo "Warning: No TIFF files found in $input_dir, skipping..."
        continue
    fi
    
    echo "Found $tiff_count TIFF files in $input_dir"
    echo "Processing with nnUNet model..."
    
    # Run inference on this dataset
    python src/inference.py \
        --input "$input_dir" \
        --output_dir "$output_dir" \
        --model_path "/projects/weilab/gohaina/nucworm/scripts/methods/nnunet/models/nnunet3d_final_model.pth" \
        --patch_size "32,96,64" \
        --stride "16,48,32" \
        --device "cuda"
    
    echo "Completed processing ${dataset}"
    echo "Heatmaps saved to: $output_dir"
done

echo ""
echo "=== nnUNet Inference Complete ==="
echo "All heatmaps saved to: /projects/weilab/gohaina/nnunet_heatmaps/"
echo "Current time: $(date)"
echo ""
echo "Next step: Extract centroids from heatmaps using src/postprocess.py"
