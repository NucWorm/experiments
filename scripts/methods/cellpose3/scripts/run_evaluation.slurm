#!/bin/bash
#SBATCH --job-name=cellpose3_evaluation
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose3_evaluation_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose3_evaluation_%j.err

# Cellpose3 Evaluation Script for NucWorm Benchmark
# This script evaluates Cellpose3 results using the NucWorm evaluation framework

echo "Starting Cellpose3 evaluation job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Set up environment
module load miniforge
source activate cellpose3

# Navigate to method directory
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose3

# Activate conda environment
source activate cellpose3

# Set ground truth directory (you may need to adjust this path)
GT_DIR="/projects/weilab/gohaina/nucworm/ground_truth"

# Check if ground truth directory exists
if [ ! -d "$GT_DIR" ]; then
    echo "Warning: Ground truth directory not found at $GT_DIR"
    echo "Please update the GT_DIR variable in this script with the correct path"
    echo "Evaluation will be skipped"
    exit 0
fi

# Run evaluation
echo "Starting Cellpose3 evaluation..."
python src/evaluate.py --config config.yaml --gt_dir "$GT_DIR"

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo "Cellpose3 evaluation completed successfully!"
    echo "End time: $(date)"
else
    echo "Cellpose3 evaluation failed!"
    echo "End time: $(date)"
    exit 1
fi

echo "Job completed at: $(date)"
