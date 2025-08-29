#!/bin/bash
#SBATCH --job-name=nnunet_evaluation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/wormid_evaluation_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/wormid_evaluation_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohaina@bc.edu

# WormID nnUNet Evaluation Slurm Script
# This script evaluates predicted centroids against ground truth

set -e

echo "=== Starting WormID nnUNet Evaluation ==="
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
if [ ! -f "src/evaluate.py" ]; then
    echo "Error: src/evaluate.py not found!"
    exit 1
fi

# Create output directory for evaluation results
echo "Creating evaluation output directory..."
mkdir -p /projects/weilab/gohaina/nucworm/outputs/evaluation_results

# Define datasets to evaluate
datasets=("nejatbakhsh20" "yemini21" "wen20")

# Evaluate each dataset
for dataset in "${datasets[@]}"; do
    echo ""
    echo "=== Evaluating dataset: ${dataset} ==="
    
    gt_dir="/projects/weilab/dataset/NucWorm/neuropal/${dataset}"
    pred_dir="/projects/weilab/gohaina/nucworm/outputs/center_point/${dataset}"
    output_dir="/projects/weilab/gohaina/nucworm/outputs/evaluation_results/${dataset}"
    
    # Check if directories exist
    if [ ! -d "$gt_dir" ]; then
        echo "Warning: Ground truth directory $gt_dir does not exist, skipping..."
        continue
    fi
    
    if [ ! -d "$pred_dir" ]; then
        echo "Warning: Prediction directory $pred_dir does not exist, skipping..."
        continue
    fi
    
    echo "Ground truth directory: $gt_dir"
    echo "Prediction directory: $pred_dir"
    echo "Output directory: $output_dir"
    
    # Run evaluation
    echo "Running evaluation..."
    python src/evaluate.py \
        --gt_dir "$gt_dir" \
        --pred_dir "$pred_dir" \
        --output_dir "$output_dir" \
        --dataset "$dataset" \
        --threshold 15.0
    
    echo "Completed evaluation for ${dataset}"
done

echo ""
echo "=== Evaluation Complete ==="
echo "All evaluation results saved to: /projects/weilab/gohaina/nucworm/outputs/evaluation_results/"
echo "Current time: $(date)"
echo ""
echo "Evaluation complete! Check the output directories for detailed results."
