#!/bin/bash
#SBATCH --job-name=cellpose_sam_postprocess
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose_sam_postprocess_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose_sam_postprocess_%j.err

# Cellpose-SAM Post-processing Script for NucWorm Benchmark
# This script extracts centroids from Cellpose-SAM instance masks

echo "Starting Cellpose-SAM post-processing job..."
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

# Run post-processing
echo "Starting Cellpose-SAM post-processing..."
python src/postprocess.py --config config.yaml

# Check if post-processing completed successfully
if [ $? -eq 0 ]; then
    echo "Cellpose-SAM post-processing completed successfully!"
    echo "End time: $(date)"
else
    echo "Cellpose-SAM post-processing failed!"
    echo "End time: $(date)"
    exit 1
fi

echo "Job completed at: $(date)"
