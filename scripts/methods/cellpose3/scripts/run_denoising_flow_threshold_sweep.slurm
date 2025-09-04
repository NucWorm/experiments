#!/bin/bash
#SBATCH --job-name=cp_denoise_flow_sweep
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose3_denoising_flow_sweep_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose3_denoising_flow_sweep_%j.err

# Cellpose3 Denoising Flow Threshold Sweep Script for NucWorm Benchmark
# This script runs the complete Cellpose3 denoising pipeline for multiple flow threshold values
# Flow thresholds: 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8

echo "Starting Cellpose3 denoising flow threshold sweep job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load required modules
module load miniconda

# Navigate to the script directory
cd /projects/weilab/gohaina/nucworm/scripts/methods/cellpose3

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

# Verify Cellpose3 installation
echo "Verifying Cellpose3 installation..."
python -c "
from cellpose import denoise, models, core, utils
print('✅ Cellpose3 denoising modules imported successfully')
print('✅ GPU available:', core.use_gpu())
"

# Define flow threshold values to test
FLOW_THRESHOLDS=(0.35 0.3 0.25 0.2 0.15 0.1)

# Base output directory
BASE_OUTPUT_DIR="/projects/weilab/gohaina/nucworm/outputs/cellpose3_denoising"

echo "Starting denoising flow threshold sweep with values: ${FLOW_THRESHOLDS[*]}"

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
    TEMP_CONFIG="config_denoising_flow_${flow_threshold}.yaml"
    echo "Creating temporary config: $TEMP_CONFIG"
    
    # Copy base config and modify flow threshold
    cp config_denoising.yaml "$TEMP_CONFIG"
    
    # Use sed to replace the flow_threshold value in the config
    sed -i "s/flow_threshold: 0.4/flow_threshold: $flow_threshold/" "$TEMP_CONFIG"
    
    # Verify the change was made
    echo "Flow threshold in config:"
    grep "flow_threshold:" "$TEMP_CONFIG"
    
    # Step 1: Run denoising inference
    echo "Step 1: Starting Cellpose3 denoising inference with flow_threshold=$flow_threshold..."
    python src/inference_denoising.py --config "$TEMP_CONFIG" --output_dir "$FLOW_OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ Step 1 completed successfully: Denoising inference for flow_threshold=$flow_threshold"
    else
        echo "❌ Step 1 failed: Denoising inference for flow_threshold=$flow_threshold"
        rm "$TEMP_CONFIG"
        continue
    fi
    
    # Step 2: Run post-processing
    echo "Step 2: Starting post-processing (centroid extraction) with flow_threshold=$flow_threshold..."
    python src/postprocess_denoising.py --config "$TEMP_CONFIG" --output_dir "$FLOW_OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ Step 2 completed successfully: Post-processing for flow_threshold=$flow_threshold"
    else
        echo "❌ Step 2 failed: Post-processing for flow_threshold=$flow_threshold"
        rm "$TEMP_CONFIG"
        continue
    fi
    
    # Step 3: Run evaluation
    echo "Step 3: Starting evaluation with flow_threshold=$flow_threshold..."
    python src/evaluate_denoising.py --config "$TEMP_CONFIG" --output_dir "$FLOW_OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ Step 3 completed successfully: Evaluation for flow_threshold=$flow_threshold"
    else
        echo "⚠️  Step 3 failed: Evaluation for flow_threshold=$flow_threshold"
        # Don't exit here as evaluation might be optional
    fi
    
    # Clean up temporary config file
    rm "$TEMP_CONFIG"
    
    echo "✅ Completed processing for flow_threshold=$flow_threshold"
    echo "Results saved to: $FLOW_OUTPUT_DIR"
    
done

echo ""
echo "🎉 Cellpose3 denoising flow threshold sweep completed!"
echo "Processed flow thresholds: ${FLOW_THRESHOLDS[*]}"
echo "Results saved in directories:"
for flow_threshold in "${FLOW_THRESHOLDS[@]}"; do
    echo "  - ${BASE_OUTPUT_DIR}_flow_${flow_threshold}"
done
echo "End time: $(date)"
