#!/bin/bash
#SBATCH --job-name=nnunet_training
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/nnunet_training_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/nnunet_training_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohaina@bc.edu

# nnUNet Training Slurm Script
# This script trains the WormID nnUNet model on training data

set -e

echo "=== Starting nnUNet Training ==="
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
if [ ! -f "src/train.py" ]; then
    echo "Error: src/train.py not found!"
    exit 1
fi

if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found!"
    exit 1
fi

# Create output directories
echo "Creating output directories..."
mkdir -p /projects/weilab/gohaina/nucworm/outputs/models/nnunet

echo "Starting nnUNet training..."
echo "Training configuration: config.yaml"

# Run training
python src/train.py \
    -c config.yaml \
    -f 0

echo ""
echo "=== nnUNet Training Complete ==="
echo "Model saved to: /projects/weilab/gohaina/nucworm/outputs/models/nnunet/"
echo "Current time: $(date)"
echo ""
echo "Next step: Run inference using scripts/run_inference.slurm"
