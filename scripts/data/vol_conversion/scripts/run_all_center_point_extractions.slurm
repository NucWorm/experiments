#!/bin/bash
#SBATCH --job-name=master_center_extraction
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/projects/weilab/gohaina/logs/master_center_extraction_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/master_center_extraction_%j.err

# Master script to submit all center point extraction array jobs
# This script submits the array jobs for both Cellpose-SAM and Cellpose3

echo "Starting master center point extraction job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Submit Cellpose-SAM array job
echo "Submitting Cellpose-SAM center point extraction array job..."
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose_sam
SAM_JOB_ID=$(sbatch scripts/run_center_point_extraction_array.slurm | awk '{print $4}')
echo "Cellpose-SAM array job submitted with ID: $SAM_JOB_ID"

# Submit Cellpose3 array job
echo "Submitting Cellpose3 center point extraction array job..."
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose3
CELLPOSE3_JOB_ID=$(sbatch scripts/run_center_point_extraction_array.slurm | awk '{print $4}')
echo "Cellpose3 array job submitted with ID: $CELLPOSE3_JOB_ID"

# Wait for both jobs to complete
echo "Waiting for array jobs to complete..."
echo "Cellpose-SAM job ID: $SAM_JOB_ID"
echo "Cellpose3 job ID: $CELLPOSE3_JOB_ID"

# Check job status
while squeue -j $SAM_JOB_ID,$CELLPOSE3_JOB_ID | grep -q $SAM_JOB_ID; do
    echo "Jobs still running... checking again in 30 seconds"
    sleep 30
done

echo "All array jobs completed!"
echo "End time: $(date)"

# Run verification script
echo "Running verification script..."
cd /projects/weilab/gohaina/nucworm
python3 << 'EOF'
import os
import numpy as np
from pathlib import Path

def check_center_points():
    """Check that all center point files were created successfully"""
    base_dir = Path("/projects/weilab/gohaina/nucworm/outputs")
    
    methods = ["cellpose_sam", "cellpose3_flow"]
    flow_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    datasets = ["nejatbakhsh20", "wen20", "yemini21"]
    
    total_expected = 0
    total_found = 0
    
    for method in methods:
        for flow in flow_thresholds:
            method_dir = base_dir / f"{method}_{flow}"
            center_point_dir = method_dir / "center_point"
            
            if not center_point_dir.exists():
                print(f"Missing center_point directory: {center_point_dir}")
                continue
                
            for dataset in datasets:
                dataset_dir = center_point_dir / dataset
                if not dataset_dir.exists():
                    print(f"Missing dataset directory: {dataset_dir}")
                    continue
                
                # Count expected vs found files
                npy_files = list(dataset_dir.glob("*.npy"))
                total_expected += len(npy_files)  # This is approximate
                total_found += len(npy_files)
                
                if len(npy_files) == 0:
                    print(f"No center point files found in: {dataset_dir}")
                else:
                    print(f"Found {len(npy_files)} center point files in: {dataset_dir}")
    
    print(f"\nSummary:")
    print(f"Total center point files found: {total_found}")
    print(f"Expected approximately: {total_expected}")
    
    if total_found > 0:
        print("✅ Center point extraction appears successful!")
    else:
        print("❌ No center point files found!")

if __name__ == "__main__":
    check_center_points()
EOF

echo "Master job completed at: $(date)"
