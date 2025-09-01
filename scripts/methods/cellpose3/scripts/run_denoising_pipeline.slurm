#!/bin/bash
#SBATCH --job-name=cyto3_d
#SBATCH --output=/projects/weilab/gohaina/logs/cellpose3_denoising_pipeline_%j.out
#SBATCH --error=/projects/weilab/gohaina/logs/cellpose3_denoising_pipeline_%j.err
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

echo "Starting Cellpose3 denoising pipeline job..."
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

# Create output directory
mkdir -p /projects/weilab/gohaina/nucworm/outputs/cellpose3_denoising

echo "Step 1: Starting Cellpose3 denoising inference..."
python src/inference_denoising.py --config config_denoising.yaml

if [ $? -eq 0 ]; then
    echo "✅ Step 1 completed successfully: Denoising inference"
else
    echo "❌ Step 1 failed: Denoising inference"
    exit 1
fi

echo "Step 2: Starting post-processing (centroid extraction)..."
python src/postprocess_denoising.py --config config_denoising.yaml

if [ $? -eq 0 ]; then
    echo "✅ Step 2 completed successfully: Post-processing"
else
    echo "❌ Step 2 failed: Post-processing"
    exit 1
fi

echo "Step 3: Starting evaluation..."
python src/evaluate_denoising.py --config config_denoising.yaml

if [ $? -eq 0 ]; then
    echo "✅ Step 3 completed successfully: Evaluation"
else
    echo "❌ Step 3 failed: Evaluation"
    exit 1
fi

echo "🎉 Cellpose3 denoising pipeline completed successfully!"
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
