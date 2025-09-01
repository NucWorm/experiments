#!/bin/bash
#SBATCH --job-name=cp_sam_pipeline
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=320G
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose_sam_pipeline_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose_sam_pipeline_%j.err

# Cellpose-SAM Full Pipeline Script for NucWorm Benchmark
# This script runs the complete Cellpose-SAM pipeline: inference -> post-processing -> evaluation

echo "Starting Cellpose-SAM full pipeline job..."
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

# Step 1: Run inference
echo "Step 1: Starting Cellpose-SAM inference..."
python src/inference.py --config config.yaml

if [ $? -ne 0 ]; then
    echo "Cellpose-SAM inference failed!"
    exit 1
fi

echo "Step 1 completed: Cellpose-SAM inference successful!"

# Step 2: Run post-processing
echo "Step 2: Starting Cellpose-SAM post-processing..."
python src/postprocess.py --config config.yaml

if [ $? -ne 0 ]; then
    echo "Cellpose-SAM post-processing failed!"
    exit 1
fi

echo "Step 2 completed: Cellpose-SAM post-processing successful!"

# Step 3: Run evaluation (optional - requires ground truth)
echo "Step 3: Starting Cellpose-SAM evaluation..."
GT_DIR="/projects/weilab/gohaina/nucworm/ground_truth"

if [ -d "$GT_DIR" ]; then
    echo "Running comprehensive evaluation..."
    python ../../../utils/evaluation/evaluate_centroids.py \
        --gt_dir "$GT_DIR" \
        --pred_dir "/projects/weilab/gohaina/nucworm/outputs/cellpose_sam/center_point" \
        --output_dir "/projects/weilab/gohaina/nucworm/outputs/cellpose_sam/evaluation_results" \
        --dataset_thresholds "nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4" \
        --save_csv --save_summary --verbose
    
    if [ $? -eq 0 ]; then
        echo "Step 3 completed: Cellpose-SAM evaluation successful!"
    else
        echo "Step 3 failed: Cellpose-SAM evaluation failed!"
        # Don't exit here as evaluation is optional
    fi
else
    echo "Step 3 skipped: Ground truth directory not found at $GT_DIR"
    echo "To run evaluation, please update the GT_DIR variable in this script"
fi

echo "Cellpose-SAM full pipeline completed!"
echo "End time: $(date)"
