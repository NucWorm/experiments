#!/bin/bash
#SBATCH --job-name=cellpose3_postprocess
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose3_postprocess_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose3_postprocess_%j.err

# Cellpose3 Post-processing Script for NucWorm Benchmark
# This script extracts centroids from Cellpose3 instance masks

echo "Starting Cellpose3 post-processing job..."
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

# Run post-processing
echo "Starting Cellpose3 post-processing..."
python src/postprocess.py --config config.yaml

# Check if post-processing completed successfully
if [ $? -eq 0 ]; then
    echo "Cellpose3 post-processing completed successfully!"
    echo "End time: $(date)"
else
    echo "Cellpose3 post-processing failed!"
    echo "End time: $(date)"
    exit 1
fi

echo "Job completed at: $(date)"
