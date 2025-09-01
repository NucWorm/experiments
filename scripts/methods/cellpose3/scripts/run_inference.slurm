#!/bin/bash
#SBATCH --job-name=cellpose3_inference
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose3_inference_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose3_inference_%j.err

# Cellpose3 Inference Script for NucWorm Benchmark
# This script runs Cellpose3 inference on neuropal volumes

echo "Starting Cellpose3 inference job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Set up environment
module load miniforge
source activate cellpose3

# Navigate to method directory
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose3

# Check if conda environment exists, create if not
echo "Checking for cellpose3 conda environment..."
if conda env list | grep -q "cellpose3"; then
    echo "Using existing cellpose3 conda environment"
    conda activate cellpose3
    echo "Checking if requirements are installed..."
    if ! python -c "import cellpose" 2>/dev/null; then
        echo "Installing missing requirements..."
        pip install -r requirements.txt
    else
        echo "All requirements already installed"
    fi
else
    echo "Creating cellpose3 conda environment..."
    conda create -n cellpose3 python=3.9 -y
    conda activate cellpose3
    echo "Installing requirements..."
    pip install -r requirements.txt
    echo "Environment created and packages installed"
fi

# Verify environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"

# Verify Cellpose3 installation
python -c "
import cellpose
print(f'Cellpose version: {cellpose.__version__}')

# Test Cellpose3 model
from cellpose import models, core
use_GPU = core.use_gpu()
print(f'GPU available: {use_GPU}')

model = models.Cellpose(gpu=False, model_type='cyto3')
print('Cellpose3 model loaded successfully')
"

# Run inference
echo "Starting Cellpose3 inference..."
python src/inference.py --config config.yaml

# Check if inference completed successfully
if [ $? -eq 0 ]; then
    echo "Cellpose3 inference completed successfully!"
    echo "End time: $(date)"
else
    echo "Cellpose3 inference failed!"
    echo "End time: $(date)"
    exit 1
fi

echo "Job completed at: $(date)"
