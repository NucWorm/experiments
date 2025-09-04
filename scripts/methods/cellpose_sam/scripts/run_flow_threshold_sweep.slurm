#!/bin/bash
#SBATCH --job-name=cp_sam_flow_sweep
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose_sam_flow_sweep_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose_sam_flow_sweep_%j.err

# Cellpose-SAM Flow Threshold Sweep Script for NucWorm Benchmark
# This script runs the complete Cellpose-SAM pipeline with different flow thresholds
# and organizes outputs in proper subdirectories

echo "Starting Cellpose-SAM flow threshold sweep..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Set up environment
module load miniconda
source activate cellpose_sam

# Navigate to method directory
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose_sam

# Check if conda environment exists, create if not
echo "Checking for cellpose_sam conda environment..."
if conda env list | grep -q "cellpose_sam"; then
    echo "Using existing cellpose_sam conda environment"
    conda activate cellpose_sam
    echo "Checking if requirements are installed..."
    if ! python -c "import cellpose" 2>/dev/null; then
        echo "Installing missing requirements..."
        pip install -r requirements.txt
    else
        echo "All requirements already installed"
    fi
else
    echo "Creating cellpose_sam conda environment..."
    conda create -n cellpose_sam python=3.9 -y
    conda activate cellpose_sam
    echo "Installing requirements..."
    pip install -r requirements.txt
    echo "Environment created and packages installed"
fi

# Verify environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"

# Verify Cellpose-SAM installation
python -c "
import cellpose
print(f'Cellpose version: {cellpose.__version__}')

# Test Cellpose-SAM model
from cellpose import models, core
use_GPU = core.use_gpu()
print(f'GPU available: {use_GPU}')

# Try to load Cellpose-SAM model
try:
    model = models.CellposeModel(gpu=False, model_type='cellpose_sam')
    print('Cellpose-SAM model loaded successfully')
except Exception as e:
    print(f'Cellpose-SAM model not available: {e}')
    print('Falling back to cyto3 model for testing...')
    model = models.CellposeModel(gpu=False, model_type='cyto3')
    print('Fallback to cyto3 model successful')
"

# Define flow thresholds to test
FLOW_THRESHOLDS=(0.35 0.3 0.2 0.25 0.1 0.15)

# Base output directory
BASE_OUTPUT_DIR="/projects/weilab/gohaina/nucworm/outputs/cellpose_sam"

echo "Testing flow thresholds: ${FLOW_THRESHOLDS[*]}"
echo "Base output directory: $BASE_OUTPUT_DIR"

# Loop through each flow threshold
for flow_threshold in "${FLOW_THRESHOLDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing flow threshold: $flow_threshold"
    echo "=========================================="
    echo "Start time: $(date)"
    
    # Create threshold-specific output directory
    THRESHOLD_OUTPUT_DIR="${BASE_OUTPUT_DIR}/flow_${flow_threshold}"
    echo "Output directory: $THRESHOLD_OUTPUT_DIR"
    
    # Create threshold-specific config file
    THRESHOLD_CONFIG="config_flow_${flow_threshold}.yaml"
    echo "Creating config file: $THRESHOLD_CONFIG"
    
    # Copy base config and modify flow threshold
    cp config.yaml "$THRESHOLD_CONFIG"
    
    # Update the flow threshold in the config file
    sed -i "s/flow_threshold: 0.4/flow_threshold: $flow_threshold/" "$THRESHOLD_CONFIG"
    
    # Update output directory in config
    sed -i "s|output_dir: \"/projects/weilab/gohaina/nucworm/outputs/cellpose_sam\"|output_dir: \"$THRESHOLD_OUTPUT_DIR\"|" "$THRESHOLD_CONFIG"
    
    echo "Config updated for flow threshold $flow_threshold"
    
    # Step 1: Run inference
    echo "Step 1: Running Cellpose-SAM inference with flow_threshold=$flow_threshold..."
    python src/inference.py --config "$THRESHOLD_CONFIG"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Cellpose-SAM inference failed for flow_threshold=$flow_threshold!"
        echo "Skipping to next threshold..."
        continue
    fi
    
    echo "Step 1 completed: Cellpose-SAM inference successful for flow_threshold=$flow_threshold!"
    
    # Step 2: Run post-processing
    echo "Step 2: Running Cellpose-SAM post-processing with flow_threshold=$flow_threshold..."
    python src/postprocess.py --config "$THRESHOLD_CONFIG"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Cellpose-SAM post-processing failed for flow_threshold=$flow_threshold!"
        echo "Skipping to next threshold..."
        continue
    fi
    
    echo "Step 2 completed: Cellpose-SAM post-processing successful for flow_threshold=$flow_threshold!"
    
    # Step 3: Run evaluation (optional - requires ground truth)
    echo "Step 3: Running Cellpose-SAM evaluation with flow_threshold=$flow_threshold..."
    GT_DIR="/projects/weilab/gohaina/nucworm/ground_truth"
    
    if [ -d "$GT_DIR" ]; then
        echo "Running comprehensive evaluation..."
        python ../../../utils/evaluation/evaluate_centroids.py \
            --gt_dir "$GT_DIR" \
            --pred_dir "$THRESHOLD_OUTPUT_DIR/center_point" \
            --output_dir "$THRESHOLD_OUTPUT_DIR/evaluation_results" \
            --dataset_thresholds "nejatbakhsh20:15,2 wen20:10,2 yemini21:15,4" \
            --save_csv --save_summary --verbose
        
        if [ $? -eq 0 ]; then
            echo "Step 3 completed: Cellpose-SAM evaluation successful for flow_threshold=$flow_threshold!"
        else
            echo "Step 3 failed: Cellpose-SAM evaluation failed for flow_threshold=$flow_threshold!"
            # Don't exit here as evaluation is optional
        fi
    else
        echo "Step 3 skipped: Ground truth directory not found at $GT_DIR"
        echo "To run evaluation, please update the GT_DIR variable in this script"
    fi
    
    # Clean up threshold-specific config file
    rm "$THRESHOLD_CONFIG"
    
    echo "Completed processing flow_threshold=$flow_threshold"
    echo "End time: $(date)"
    
    # Create a summary file for this threshold
    SUMMARY_FILE="$THRESHOLD_OUTPUT_DIR/flow_threshold_summary.txt"
    echo "Flow Threshold: $flow_threshold" > "$SUMMARY_FILE"
    echo "Processing Date: $(date)" >> "$SUMMARY_FILE"
    echo "Output Directory: $THRESHOLD_OUTPUT_DIR" >> "$SUMMARY_FILE"
    echo "Job ID: $SLURM_JOB_ID" >> "$SUMMARY_FILE"
    echo "Node: $SLURM_NODELIST" >> "$SUMMARY_FILE"
    
    echo "Summary saved to: $SUMMARY_FILE"
done

echo ""
echo "=========================================="
echo "Flow threshold sweep completed!"
echo "=========================================="
echo "Processed flow thresholds: ${FLOW_THRESHOLDS[*]}"
echo "Results saved in: $BASE_OUTPUT_DIR/flow_*"
echo "End time: $(date)"

# Create overall summary
OVERALL_SUMMARY="$BASE_OUTPUT_DIR/flow_threshold_sweep_summary.txt"
echo "Cellpose-SAM Flow Threshold Sweep Summary" > "$OVERALL_SUMMARY"
echo "=========================================" >> "$OVERALL_SUMMARY"
echo "Job ID: $SLURM_JOB_ID" >> "$OVERALL_SUMMARY"
echo "Node: $SLURM_NODELIST" >> "$OVERALL_SUMMARY"
echo "Start Time: $(date)" >> "$OVERALL_SUMMARY"
echo "Flow Thresholds Tested: ${FLOW_THRESHOLDS[*]}" >> "$OVERALL_SUMMARY"
echo "Base Output Directory: $BASE_OUTPUT_DIR" >> "$OVERALL_SUMMARY"
echo "" >> "$OVERALL_SUMMARY"
echo "Individual Results:" >> "$OVERALL_SUMMARY"
for flow_threshold in "${FLOW_THRESHOLDS[@]}"; do
    echo "  flow_${flow_threshold}/ - Flow threshold $flow_threshold" >> "$OVERALL_SUMMARY"
done

echo "Overall summary saved to: $OVERALL_SUMMARY"
echo "Job completed at: $(date)"
