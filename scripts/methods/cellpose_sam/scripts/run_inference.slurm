#!/bin/bash
#SBATCH --job-name=cellpose_sam_inference
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose_sam_inference_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose_sam_inference_%j.err

# Cellpose-SAM Inference Script for NucWorm Benchmark
# This script runs Cellpose-SAM inference on neuropal volumes

echo "Starting Cellpose-SAM inference job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Set up environment
module load miniconda
source activate cellpose_sam

# Navigate to method directory
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose_sam

# Check if conda environment exists, create if not
echo "Checking for cellpose_sam conda environment..."
if conda env list | grep -q "cellpose_sam"; then
    echo "Using existing cellpose_sam conda environment"
    conda activate cellpose_sam
    echo "Checking if requirements are installed..."
    if ! python -c "import cellpose" 2>/dev/null; then
        echo "Installing missing requirements..."
        pip install -r requirements.txt
    else
        echo "All requirements already installed"
    fi
else
    echo "Creating cellpose_sam conda environment..."
    conda create -n cellpose_sam python=3.9 -y
    conda activate cellpose_sam
    echo "Installing requirements..."
    pip install -r requirements.txt
    echo "Environment created and packages installed"
fi

# Verify environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"

# Verify Cellpose-SAM installation
python -c "
import cellpose
print(f'Cellpose version: {cellpose.__version__}')

# Test Cellpose-SAM model
from cellpose import models, core
use_GPU = core.use_gpu()
print(f'GPU available: {use_GPU}')

# Try to load Cellpose-SAM model
try:
    model = models.Cellpose(gpu=False, model_type='cellpose_sam')
    print('Cellpose-SAM model loaded successfully')
except Exception as e:
    print(f'Cellpose-SAM model not available: {e}')
    print('Falling back to cyto3 model for testing...')
    model = models.Cellpose(gpu=False, model_type='cyto3')
    print('Fallback to cyto3 model successful')
"

# Run inference
echo "Starting Cellpose-SAM inference..."
python src/inference.py --config config.yaml

# Check if inference completed successfully
if [ $? -eq 0 ]; then
    echo "Cellpose-SAM inference completed successfully!"
    echo "End time: $(date)"
else
    echo "Cellpose-SAM inference failed!"
    echo "End time: $(date)"
    exit 1
fi

echo "Job completed at: $(date)"
