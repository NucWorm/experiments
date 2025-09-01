#!/bin/bash
#SBATCH --job-name=cp3_flow_sweep
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose3_flow_sweep_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose3_flow_sweep_%j.err

# Cellpose3 Flow Threshold Sweep Script for NucWorm Benchmark
# This script runs the complete Cellpose3 pipeline for multiple flow threshold values
# Flow thresholds: 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8

echo "Starting Cellpose3 flow threshold sweep job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Set up environment
module load miniconda
source activate cellpose3

# Navigate to method directory
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose3

# Check if conda environment exists, create if not
echo "Checking for cellpose3 conda environment..."
if conda env list | grep -q "cellpose3"; then
    echo "Using existing cellpose3 conda environment"
    conda activate cellpose3
    echo "Checking if requirements are installed..."
    if ! python -c "import cellpose" 2>/dev/null; then
        echo "Installing missing requirements..."
        pip install -r requirements.txt
    else
        echo "All requirements already installed"
    fi
else
    echo "Creating cellpose3 conda environment..."
    conda create -n cellpose3 python=3.9 -y
    conda activate cellpose3
    echo "Installing requirements..."
    pip install -r requirements.txt
    echo "Environment created and packages installed"
fi

# Verify environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"

# Verify Cellpose3 installation
python -c "
import cellpose
print(f'Cellpose version: {cellpose.__version__}')

# Test Cellpose3 model
from cellpose import models, core
use_GPU = core.use_gpu()
print(f'GPU available: {use_GPU}')

model = models.Cellpose(gpu=False, model_type='cyto3')
print('Cellpose3 model loaded successfully')
"

# Define flow threshold values to test
FLOW_THRESHOLDS=(0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8)

# Base output directory
BASE_OUTPUT_DIR="/projects/weilab/gohaina/nucworm/outputs/cellpose3"

echo "Starting flow threshold sweep with values: ${FLOW_THRESHOLDS[*]}"

# Loop through each flow threshold value
for flow_threshold in "${FLOW_THRESHOLDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing flow threshold: $flow_threshold"
    echo "=========================================="
    
    # Create flow threshold specific output directory
    FLOW_OUTPUT_DIR="${BASE_OUTPUT_DIR}_flow_${flow_threshold}"
    echo "Output directory: $FLOW_OUTPUT_DIR"
    
    # Create temporary config file with modified flow threshold
    TEMP_CONFIG="config_flow_${flow_threshold}.yaml"
    echo "Creating temporary config: $TEMP_CONFIG"
    
    # Copy base config and modify flow threshold
    cp config.yaml "$TEMP_CONFIG"
    
    # Use sed to replace the flow_threshold value in the config
    sed -i "s/flow_threshold: 0.4/flow_threshold: $flow_threshold/" "$TEMP_CONFIG"
    
    # Verify the change was made
    echo "Flow threshold in config:"
    grep "flow_threshold:" "$TEMP_CONFIG"
    
    # Step 1: Run inference
    echo "Step 1: Starting Cellpose3 inference with flow_threshold=$flow_threshold..."
    python src/inference.py --config "$TEMP_CONFIG" --output_dir "$FLOW_OUTPUT_DIR"
    
    if [ $? -ne 0 ]; then
        echo "❌ Cellpose3 inference failed for flow_threshold=$flow_threshold!"
        rm "$TEMP_CONFIG"
        continue
    fi
    
    echo "✅ Step 1 completed: Cellpose3 inference successful for flow_threshold=$flow_threshold"
    
    # Step 2: Run post-processing
    echo "Step 2: Starting Cellpose3 post-processing with flow_threshold=$flow_threshold..."
    python src/postprocess.py --config "$TEMP_CONFIG" --output_dir "$FLOW_OUTPUT_DIR"
    
    if [ $? -ne 0 ]; then
        echo "❌ Cellpose3 post-processing failed for flow_threshold=$flow_threshold!"
        rm "$TEMP_CONFIG"
        continue
    fi
    
    echo "✅ Step 2 completed: Cellpose3 post-processing successful for flow_threshold=$flow_threshold"
    
    # Step 3: Run evaluation (optional - requires ground truth)
    echo "Step 3: Starting Cellpose3 evaluation with flow_threshold=$flow_threshold..."
    GT_DIR="/projects/weilab/gohaina/nucworm/ground_truth"
    
    if [ -d "$GT_DIR" ]; then
        python src/evaluate.py --config "$TEMP_CONFIG" --gt_dir "$GT_DIR" --output_dir "$FLOW_OUTPUT_DIR"
        
        if [ $? -eq 0 ]; then
            echo "✅ Step 3 completed: Cellpose3 evaluation successful for flow_threshold=$flow_threshold"
        else
            echo "⚠️  Step 3 failed: Cellpose3 evaluation failed for flow_threshold=$flow_threshold"
            # Don't exit here as evaluation is optional
        fi
    else
        echo "⚠️  Step 3 skipped: Ground truth directory not found at $GT_DIR"
        echo "To run evaluation, please update the GT_DIR variable in this script"
    fi
    
    # Clean up temporary config file
    rm "$TEMP_CONFIG"
    
    echo "✅ Completed processing for flow_threshold=$flow_threshold"
    echo "Results saved to: $FLOW_OUTPUT_DIR"
    
done

echo ""
echo "🎉 Cellpose3 flow threshold sweep completed!"
echo "Processed flow thresholds: ${FLOW_THRESHOLDS[*]}"
echo "Results saved in directories:"
for flow_threshold in "${FLOW_THRESHOLDS[@]}"; do
    echo "  - ${BASE_OUTPUT_DIR}_flow_${flow_threshold}"
done
echo "End time: $(date)"
