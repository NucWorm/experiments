#!/bin/bash
#SBATCH --job-name=nnunet_full_pipeline
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/projects/weilab/gohaina/logs/nnunet_fold_%A_%a_pipeline_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/nnunet_fold_%A_%a_pipeline_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohaina@bc.edu

# Full WormID Pipeline Slurm Script with Array Jobs
# This script runs the complete pipeline: H5→TIFF → nnUNet → Centroids
# Array indices: 0-4 corresponding to fold_0 through fold_4

set -e

# Define fold parameters
FOLD_ID=$SLURM_ARRAY_TASK_ID
FOLD_NAME="fold_${FOLD_ID}"
MODEL_PATH="/projects/weilab/gohaina/nucworm/scripts/methods/nnunet/models/nnunet3d_${FOLD_NAME}_model.pth"
OUTPUT_BASE="/projects/weilab/gohaina/nucworm/outputs"
METHOD_NAME="nnunet"

echo "=== Starting Full WormID Pipeline for ${FOLD_NAME} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Fold: ${FOLD_NAME}"
echo "Model: ${MODEL_PATH}"
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

# Only run H5 to TIFF conversion for the first array job (fold_0)
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo ""
    echo "=== Step 1: H5 to TIFF Conversion (Array Task 0 only) ==="
    echo "Submitting H5 to TIFF conversion job..."

    # Submit H5 to TIFF conversion (single job, not array)
    h5_job=$(sbatch --parsable /projects/weilab/gohaina/nucworm/scripts/data/vol_conversion/scripts/run_h5_to_tiff_conversion_single.slurm)
    echo "H5 to TIFF conversion job submitted with ID: $h5_job"

    # Wait for H5 to TIFF conversion to complete
    echo "Waiting for H5 to TIFF conversion to complete..."
    while squeue -j $h5_job | grep -q "$h5_job"; do
        echo "H5 to TIFF conversion still running... waiting 60 seconds"
        sleep 60
    done

    echo "H5 to TIFF conversion completed!"
else
    echo ""
    echo "=== Step 1: Waiting for H5 to TIFF Conversion (Array Task ${SLURM_ARRAY_TASK_ID}) ==="
    echo "This array task is waiting for H5 to TIFF conversion to complete..."
    
    # Wait for TIFF files to be available
    TIFF_BASE="/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff"
    while [ ! -d "$TIFF_BASE" ] || [ $(find "$TIFF_BASE" -name "*.tiff" | wc -l) -eq 0 ]; do
        echo "Waiting for TIFF files to be available in $TIFF_BASE..."
        sleep 60
    done
    
    echo "TIFF files are now available!"
fi

echo ""
echo "=== Step 2: nnUNet Inference ==="
echo "Submitting nnUNet inference job..."

# Submit nnUNet inference with fold parameters
nnunet_job=$(sbatch --parsable --export=FOLD_ID=${FOLD_ID},FOLD_NAME=${FOLD_NAME},MODEL_PATH=${MODEL_PATH},OUTPUT_BASE=${OUTPUT_BASE},METHOD_NAME=${METHOD_NAME} scripts/run_inference.slurm)
echo "nnUNet inference job submitted with ID: $nnunet_jobay "

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

# Submit centroid extraction with fold parameters
centroid_job=$(sbatch --parsable --export=FOLD_ID=${FOLD_ID},FOLD_NAME=${FOLD_NAME},OUTPUT_BASE=${OUTPUT_BASE},METHOD_NAME=${METHOD_NAME} scripts/run_postprocess.slurm)
echo "Centroid extraction job submitted with ID: $centroid_job"

# Wait for centroid extraction to complete
echo "Waiting for centroid extraction to complete..."
while squeue -j $centroid_job | grep -q "$centroid_job"; do
    echo "Centroid extraction still running... waiting 60 seconds"
    sleep 60
done

echo "Centroid extraction completed!"

echo ""
echo "=== Full Pipeline Complete for ${FOLD_NAME}! ==="
echo "All steps completed successfully!"
echo "Final outputs:"
echo "  - TIFF files: /projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff/"
echo "  - Heatmaps: ${OUTPUT_BASE}/${METHOD_NAME}/${FOLD_NAME}/nnunet_heatmaps/"
echo "  - Centroids: ${OUTPUT_BASE}/${METHOD_NAME}/${FOLD_NAME}/center_point/"
echo "Current time: $(date)"
