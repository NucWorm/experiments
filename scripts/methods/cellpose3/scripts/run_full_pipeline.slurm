#!/bin/bash
#SBATCH --job-name=cellpose3_pipeline
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=320G
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose3_pipeline_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose3_pipeline_%j.err

# Cellpose3 Full Pipeline Script for NucWorm Benchmark
# This script runs the complete Cellpose3 pipeline: inference -> post-processing -> evaluation

echo "Starting Cellpose3 full pipeline job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Set up environment
module load miniconda
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

# Step 1: Run inference
echo "Step 1: Starting Cellpose3 inference..."
python src/inference.py --config config.yaml

if [ $? -ne 0 ]; then
    echo "Cellpose3 inference failed!"
    exit 1
fi

echo "Step 1 completed: Cellpose3 inference successful!"

# Step 2: Run post-processing
echo "Step 2: Starting Cellpose3 post-processing..."
python src/postprocess.py --config config.yaml

if [ $? -ne 0 ]; then
    echo "Cellpose3 post-processing failed!"
    exit 1
fi

echo "Step 2 completed: Cellpose3 post-processing successful!"

# Step 3: Run evaluation (optional - requires ground truth)
echo "Step 3: Starting Cellpose3 evaluation..."
GT_DIR="/projects/weilab/gohaina/nucworm/ground_truth"

if [ -d "$GT_DIR" ]; then
    python src/evaluate.py --config config.yaml --gt_dir "$GT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "Step 3 completed: Cellpose3 evaluation successful!"
    else
        echo "Step 3 failed: Cellpose3 evaluation failed!"
        # Don't exit here as evaluation is optional
    fi
else
    echo "Step 3 skipped: Ground truth directory not found at $GT_DIR"
    echo "To run evaluation, please update the GT_DIR variable in this script"
fi

echo "Cellpose3 full pipeline completed!"
echo "End time: $(date)"
