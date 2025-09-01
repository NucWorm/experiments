#!/bin/bash
#SBATCH --job-name=cellpose_sam_evaluation
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose_sam_evaluation_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose_sam_evaluation_%j.err

# Cellpose-SAM Evaluation Script for NucWorm Benchmark
# This script evaluates Cellpose-SAM results against ground truth

echo "Starting Cellpose-SAM evaluation job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Set up environment
module load miniconda
source activate cellpose_sam

# Navigate to method directory
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose_sam

# Activate conda environment
source activate cellpose_sam

# Run evaluation using the comprehensive evaluation tools
echo "Starting Cellpose-SAM evaluation..."

# Use the comprehensive evaluation framework
GT_DIR="/projects/weilab/gohaina/nucworm/ground_truth"
PRED_DIR="/projects/weilab/gohaina/nucworm/outputs/cellpose_sam/center_point"
OUTPUT_DIR="/projects/weilab/gohaina/nucworm/outputs/cellpose_sam/evaluation_results"

if [ -d "$GT_DIR" ]; then
    echo "Running comprehensive evaluation..."
    python ../../../utils/evaluation/evaluate_centroids.py \
        --gt_dir "$GT_DIR" \
        --pred_dir "$PRED_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_thresholds "nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4" \
        --save_csv --save_summary --verbose
    
    if [ $? -eq 0 ]; then
        echo "Cellpose-SAM evaluation completed successfully!"
    else
        echo "Cellpose-SAM evaluation failed!"
        exit 1
    fi
else
    echo "Ground truth directory not found at $GT_DIR"
    echo "Skipping evaluation - please update GT_DIR path in this script"
fi

echo "End time: $(date)"
echo "Job completed at: $(date)"
