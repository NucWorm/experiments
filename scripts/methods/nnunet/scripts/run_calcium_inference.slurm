#!/bin/bash
#SBATCH --job-name=calcium_inference
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/calcium_inference_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/calcium_inference_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohaina@bc.edu

# TIFF Inference Slurm Script
# This script runs the inference pipeline on TIFF files
# Runs nnUNet inference with channel duplication and extracts centroids

set -e

# Set default values if not provided
TIFF_DIR=${TIFF_DIR:-"/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff"}
MODEL_PATH=${MODEL_PATH:-"/projects/weilab/gohaina/nucworm/scripts/methods/nnunet/models/nnunet3d_fold_0_model.pth"}
OUTPUT_DIR=${OUTPUT_DIR:-"/projects/weilab/gohaina/nucworm/outputs/calcium_inference/fold_0"}
INPUT_CHANNELS=${INPUT_CHANNELS:-3}
DUPLICATE_CHANNELS=${DUPLICATE_CHANNELS:-"true"}
EXTRACT_CENTROIDS=${EXTRACT_CENTROIDS:-"true"}
GT_FOLDER=${GT_FOLDER:-""}

echo "=== Starting TIFF Inference Pipeline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "TIFF Directory: ${TIFF_DIR}"
echo "Model Path: ${MODEL_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Input Channels: ${INPUT_CHANNELS}"
echo "Duplicate Channels: ${DUPLICATE_CHANNELS}"
echo "Extract Centroids: ${EXTRACT_CENTROIDS}"
echo "Ground Truth Folder: ${GT_FOLDER}"
echo "Node: $SLURM_NODELIST"
echo "Current directory: $(pwd)"
echo "Current time: $(date)"

# Load miniconda
echo "Loading miniconda module..."
module load miniconda

# Change to the nnunet directory
cd /projects/weilab/gohaina/nucworm/scripts/methods/nnunet

echo "Working directory: $(pwd)"

# Create log directory
mkdir -p /projects/weilab/gohaina/logs

# Check if required files exist
if [ ! -f "src/inference.py" ]; then
    echo "Error: src/inference.py not found!"
    exit 1
fi

if [ ! -f "src/postprocess.py" ]; then
    echo "Error: src/postprocess.py not found!"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH!"
    exit 1
fi

# Check if TIFF directory exists and has TIFF files
if [ ! -d "$TIFF_DIR" ]; then
    echo "Error: TIFF directory $TIFF_DIR does not exist!"
    exit 1
fi

tiff_count=$(find "$TIFF_DIR" -name "*.tiff" -o -name "*.tif" | wc -l)
if [ "$tiff_count" -eq 0 ]; then
    echo "Error: No TIFF files found in $TIFF_DIR!"
    exit 1
fi

echo "Found $tiff_count TIFF files in $TIFF_DIR"

# Create output directory
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=== Running TIFF Inference Pipeline ==="

# Build the command
cmd="python src/inference.py"
cmd="$cmd --input \"$TIFF_DIR\""
cmd="$cmd --output_dir \"$OUTPUT_DIR\""
cmd="$cmd --model_path \"$MODEL_PATH\""
cmd="$cmd --patch_size \"32,96,64\""
cmd="$cmd --stride \"16,48,32\""
cmd="$cmd --device \"cuda\""
cmd="$cmd --input_channels $INPUT_CHANNELS"

if [ "$DUPLICATE_CHANNELS" = "true" ]; then
    cmd="$cmd --duplicate_channels"
fi

if [ "$EXTRACT_CENTROIDS" = "true" ]; then
    cmd="$cmd --extract_centroids"
fi

if [ -n "$GT_FOLDER" ] && [ "$GT_FOLDER" != "" ]; then
    cmd="$cmd --gt_folder \"$GT_FOLDER\""
fi

echo "Running command: $cmd"
echo ""

# Run the TIFF inference pipeline
eval $cmd

echo ""
echo "=== TIFF Inference Pipeline Complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo "Current time: $(date)"
echo ""
echo "Output structure:"
echo "  - Heatmaps: $OUTPUT_DIR/heatmaps/"
echo "  - Centroids: $OUTPUT_DIR/centroids/"
echo "  - Results summary: $OUTPUT_DIR/inference_results.txt"
