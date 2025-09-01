#!/bin/bash
#SBATCH --job-name=cellpose_sam_test
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose_sam_test_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose_sam_test_%j.err

# Cellpose-SAM Single Volume Test Script for NucWorm Benchmark
# This script tests Cellpose-SAM on a single volume for validation

echo "Starting Cellpose-SAM single volume test..."
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

# Test on a single volume from nejatbakhsh20 dataset
echo "Testing Cellpose-SAM on single volume..."
python src/inference.py --config config.yaml --dataset nejatbakhsh20

# Check if test completed successfully
if [ $? -eq 0 ]; then
    echo "Cellpose-SAM single volume test completed successfully!"
    
    # Run post-processing on the test volume
    echo "Running post-processing on test volume..."
    python src/postprocess.py --config config.yaml --dataset nejatbakhsh20
    
    if [ $? -eq 0 ]; then
        echo "Post-processing test completed successfully!"
        echo "Check outputs in: /projects/weilab/gohaina/nucworm/outputs/cellpose_sam/"
    else
        echo "Post-processing test failed!"
        exit 1
    fi
else
    echo "Cellpose-SAM single volume test failed!"
    exit 1
fi

echo "End time: $(date)"
echo "Job completed at: $(date)"
