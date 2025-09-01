#!/bin/bash
#SBATCH --job-name=cellpose3_test
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose3_test_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose3_test_%j.err

# Cellpose3 Single Volume Test Script
# This script tests Cellpose3 on a single volume for quick validation

echo "Starting Cellpose3 single volume test..."
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

# Test on a single volume from nejatbakhsh20 dataset
echo "Testing on a single volume from nejatbakhsh20 dataset..."
python src/inference.py --config config.yaml --dataset nejatbakhsh20

if [ $? -eq 0 ]; then
    echo "✅ Single volume inference test successful!"
    
    # Test post-processing on the same dataset
    echo "Testing post-processing..."
    python src/postprocess.py --config config.yaml --dataset nejatbakhsh20
    
    if [ $? -eq 0 ]; then
        echo "✅ Single volume post-processing test successful!"
        echo "Test completed successfully!"
    else
        echo "❌ Post-processing test failed!"
        exit 1
    fi
else
    echo "❌ Single volume inference test failed!"
    exit 1
fi

echo "End time: $(date)"
