#!/bin/bash
#SBATCH --job-name=nnunet_inference
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/nnunet_%A_%a_inference_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/nnunet_%A_%a_inference_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohaina@bc.edu

# nnUNet Inference Slurm Script
# This script runs the WormID nnUNet model on all converted TIFF files to generate heatmaps
# Accepts fold parameters via environment variables

set -e

# Set default values if not provided
FOLD_ID=${FOLD_ID:-0}
FOLD_NAME=${FOLD_NAME:-"fold_0"}
MODEL_PATH=${MODEL_PATH:-"/projects/weilab/gohaina/nucworm/scripts/methods/nnunet/models/nnunet3d_fold_0_model.pth"}
OUTPUT_BASE=${OUTPUT_BASE:-"/projects/weilab/gohaina/nucworm/outputs"}
METHOD_NAME=${METHOD_NAME:-"nnunet"}

echo "=== Starting nnUNet Inference for ${FOLD_NAME} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Fold: ${FOLD_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Output Base: ${OUTPUT_BASE}"
echo "Method: ${METHOD_NAME}"
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

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH!"
    exit 1
fi

# Create output directories
echo "Creating output directories..."
mkdir -p "${OUTPUT_BASE}/${METHOD_NAME}/${FOLD_NAME}/nnunet_heatmaps/nejatbakhsh20"
mkdir -p "${OUTPUT_BASE}/${METHOD_NAME}/${FOLD_NAME}/nnunet_heatmaps/yemini21"
mkdir -p "${OUTPUT_BASE}/${METHOD_NAME}/${FOLD_NAME}/nnunet_heatmaps/wen20"

# Define datasets to process
datasets=("nejatbakhsh20" "yemini21" "wen20")

# Process each dataset
for dataset in "${datasets[@]}"; do
    echo ""
    echo "=== Processing dataset: ${dataset} ==="
    
    input_dir="/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/${dataset}"
    output_dir="${OUTPUT_BASE}/${METHOD_NAME}/${FOLD_NAME}/nnunet_heatmaps/${dataset}"
    
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
        --model_path "$MODEL_PATH" \
        --patch_size "32,96,64" \
        --stride "16,48,32" \
        --device "cuda"
    
    echo "Completed processing ${dataset}"
    echo "Heatmaps saved to: $output_dir"
done

echo ""
echo "=== nnUNet Inference Complete for ${FOLD_NAME} ==="
echo "All heatmaps saved to: ${OUTPUT_BASE}/${METHOD_NAME}/${FOLD_NAME}/nnunet_heatmaps/"
echo "Current time: $(date)"
echo ""
echo "Next step: Extract centroids from heatmaps using src/postprocess.py"
