#!/bin/bash
#SBATCH --job-name=nnunet_full_pipeline
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/full_wormid_pipeline_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/full_wormid_pipeline_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohaina@bc.edu

# Full WormID Pipeline Slurm Script
# This script runs the complete pipeline: H5→TIFF → nnUNet → Centroids

set -e

echo "=== Starting Full WormID Pipeline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Current directory: $(pwd)"
echo "Current time: $(date)"

# Load miniconda module
echo "Loading miniconda module..."
module load miniconda

# Change to the wormid_nnunet directory
cd /projects/weilab/gohaina/nucworm/scripts/methods/nnunet

echo "Working directory: $(pwd)"

# Create log directory
mkdir -p /projects/weilab/gohaina/logs

echo ""
echo "=== Step 1: H5 to TIFF Conversion ==="
echo "Submitting H5 to TIFF conversion job..."

# Submit H5 to TIFF conversion
h5_job=$(sbatch --parsable /projects/weilab/gohaina/nucworm/scripts/data/vol_conversion/run_h5_to_tiff_conversion.slurm)
echo "H5 to TIFF conversion job submitted with ID: $h5_job"

# Wait for H5 to TIFF conversion to complete
echo "Waiting for H5 to TIFF conversion to complete..."
while squeue -j $h5_job | grep -q "$h5_job"; do
    echo "H5 to TIFF conversion still running... waiting 60 seconds"
    sleep 60
done

echo "H5 to TIFF conversion completed!"

echo ""
echo "=== Step 2: nnUNet Inference ==="
echo "Submitting nnUNet inference job..."

# Submit nnUNet inference
nnunet_job=$(sbatch --parsable scripts/run_inference.slurm)
echo "nnUNet inference job submitted with ID: $nnunet_job"

# Wait for nnUNet inference to complete
echo "Waiting for nnUNet inference to complete..."
while squeue -j $nnunet_job | grep -q "$nnunet_job"; do
    echo "nnUNet inference still running... waiting 60 seconds"
    sleep 60
done

echo "nnUNet inference completed!"

echo ""
echo "=== Step 3: Centroid Extraction ==="
echo "Submitting centroid extraction job..."

# Submit centroid extraction
centroid_job=$(sbatch --parsable scripts/run_postprocess.slurm)
echo "Centroid extraction job submitted with ID: $centroid_job"

# Wait for centroid extraction to complete
echo "Waiting for centroid extraction to complete..."
while squeue -j $centroid_job | grep -q "$centroid_job"; do
    echo "Centroid extraction still running... waiting 60 seconds"
    sleep 60
done

echo "Centroid extraction completed!"

echo ""
echo "=== Full Pipeline Complete! ==="
echo "All steps completed successfully!"
echo "Final outputs:"
echo "  - TIFF files: /projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/"
echo "  - Heatmaps: /projects/weilab/gohaina/nnunet_heatmaps/"
echo "  - Centroids: /projects/weilab/gohaina/center_point/"
echo "Current time: $(date)"
