#!/bin/bash
#SBATCH --job-name=extract_centers
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=1-48
#SBATCH --output=/projects/weilab/gohaina/logs/extract_centers_%A_%a.out
#SBATCH --error=/projects/weilab/gohaina/logs/extract_centers_%A_%a.err

# Center Point Extraction Array Job for NucWorm Benchmark
# This script extracts centroids from instance segmentation masks across all methods and flow thresholds

echo "Starting center point extraction job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Set up environment
module load miniconda
source activate cellpose_sam

# Define the tasks array
# Format: "method:flow_threshold:dataset:case_id"
declare -a tasks=(
    # Cellpose-SAM (16 flow thresholds × 3 datasets = 48 tasks)
    "cellpose_sam:0.1:nejatbakhsh20:000541"
    "cellpose_sam:0.1:wen20:000692"
    "cellpose_sam:0.1:yemini21:000715"
    "cellpose_sam:0.15:nejatbakhsh20:000541"
    "cellpose_sam:0.15:wen20:000692"
    "cellpose_sam:0.15:yemini21:000715"
    "cellpose_sam:0.2:nejatbakhsh20:000541"
    "cellpose_sam:0.2:wen20:000692"
    "cellpose_sam:0.2:yemini21:000715"
    "cellpose_sam:0.25:nejatbakhsh20:000541"
    "cellpose_sam:0.25:wen20:000692"
    "cellpose_sam:0.25:yemini21:000715"
    "cellpose_sam:0.3:nejatbakhsh20:000541"
    "cellpose_sam:0.3:wen20:000692"
    "cellpose_sam:0.3:yemini21:000715"
    "cellpose_sam:0.35:nejatbakhsh20:000541"
    "cellpose_sam:0.35:wen20:000692"
    "cellpose_sam:0.35:yemini21:000715"
    "cellpose_sam:0.4:nejatbakhsh20:000541"
    "cellpose_sam:0.4:wen20:000692"
    "cellpose_sam:0.4:yemini21:000715"
    "cellpose_sam:0.45:nejatbakhsh20:000541"
    "cellpose_sam:0.45:wen20:000692"
    "cellpose_sam:0.45:yemini21:000715"
    "cellpose_sam:0.5:nejatbakhsh20:000541"
    "cellpose_sam:0.5:wen20:000692"
    "cellpose_sam:0.5:yemini21:000715"
    "cellpose_sam:0.55:nejatbakhsh20:000541"
    "cellpose_sam:0.55:wen20:000692"
    "cellpose_sam:0.55:yemini21:000715"
    "cellpose_sam:0.6:nejatbakhsh20:000541"
    "cellpose_sam:0.6:wen20:000692"
    "cellpose_sam:0.6:yemini21:000715"
    "cellpose_sam:0.65:nejatbakhsh20:000541"
    "cellpose_sam:0.65:wen20:000692"
    "cellpose_sam:0.65:yemini21:000715"
    "cellpose_sam:0.7:nejatbakhsh20:000541"
    "cellpose_sam:0.7:wen20:000692"
    "cellpose_sam:0.7:yemini21:000715"
    "cellpose_sam:0.75:nejatbakhsh20:000541"
    "cellpose_sam:0.75:wen20:000692"
    "cellpose_sam:0.75:yemini21:000715"
    "cellpose_sam:0.8:nejatbakhsh20:000541"
    "cellpose_sam:0.8:wen20:000692"
    "cellpose_sam:0.8:yemini21:000715"
)

# Get the task for this array element
task="${tasks[$((SLURM_ARRAY_TASK_ID - 1))]}"
IFS=':' read -r method flow_threshold dataset case_id <<< "$task"

echo "Processing: $method, flow_threshold: $flow_threshold, dataset: $dataset, case: $case_id"

# Set up paths
base_dir="/projects/weilab/gohaina/nucworm/outputs"
if [ "$method" = "cellpose_sam" ]; then
    method_dir="${base_dir}/cellpose_sam/flow_${flow_threshold}"
elif [ "$method" = "cellpose3" ]; then
    method_dir="${base_dir}/cellpose3_flow_${flow_threshold}"
elif [ "$method" = "cellpose3_denoising" ]; then
    method_dir="${base_dir}/cellpose3_denoising_flow_${flow_threshold}"
fi

mask_dir="${method_dir}/${dataset}/${case_id}/masks"
output_dir="${method_dir}/center_point/${dataset}"

echo "Mask directory: $mask_dir"
echo "Output directory: $output_dir"

# Create output directory
mkdir -p "$output_dir"

# Check if mask directory exists
if [ ! -d "$mask_dir" ]; then
    echo "ERROR: Mask directory does not exist: $mask_dir"
    exit 1
fi

# Python script to extract centroids
python3 << EOF
import os
import sys
import numpy as np
import tifffile as tiff
from pathlib import Path
from scipy import ndimage
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_centroids_from_masks(mask_dir, output_dir):
    """Extract centroids from all mask files in a directory"""
    mask_path = Path(mask_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all mask files
    mask_files = list(mask_path.glob('*_masks.tiff')) + list(mask_path.glob('*_masks.tif'))
    
    if not mask_files:
        logger.warning(f"No mask files found in {mask_dir}")
        return 0
    
    total_centroids = 0
    successful_files = 0
    
    for mask_file in mask_files:
        try:
            # Load mask
            mask = tiff.imread(mask_file)
            logger.info(f"Processing {mask_file.name}: shape={mask.shape}, dtype={mask.dtype}, max_label={mask.max()}")
            
            # Extract centroids
            centroids = []
            unique_labels = np.unique(mask)
            unique_labels = unique_labels[unique_labels > 0]  # Remove background
            
            for label in unique_labels:
                binary_mask = (mask == label)
                center_of_mass = ndimage.center_of_mass(binary_mask)
                centroids.append(center_of_mass)
            
            if centroids:
                centroids = np.array(centroids).astype(np.float32)
                
                # Create output filename
                base_name = mask_file.stem.replace('_masks', '')
                output_filename = f'{base_name}_im_points.npy'
                output_file = output_path / output_filename
                
                # Save centroids
                np.save(output_file, centroids)
                logger.info(f'Extracted {len(centroids)} centroids from {mask_file.name}')
                total_centroids += len(centroids)
                successful_files += 1
            else:
                logger.warning(f'No centroids found in {mask_file.name}')
                # Still create empty file for consistency
                base_name = mask_file.stem.replace('_masks', '')
                output_filename = f'{base_name}_im_points.npy'
                output_file = output_path / output_filename
                np.save(output_file, np.empty((0, 3), dtype=np.float32))
                successful_files += 1
                
        except Exception as e:
            logger.error(f'Error processing {mask_file}: {e}')
    
    logger.info(f'Completed processing {successful_files}/{len(mask_files)} files, total centroids: {total_centroids}')
    return successful_files

# Run the extraction
successful = extract_centroids_from_masks("$mask_dir", "$output_dir")
if successful == 0:
    sys.exit(1)
EOF

# Check if extraction was successful
if [ $? -eq 0 ]; then
    echo "Center point extraction completed successfully!"
    echo "End time: $(date)"
else
    echo "Center point extraction failed!"
    echo "End time: $(date)"
    exit 1
fi

echo "Job completed at: $(date)"
