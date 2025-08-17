#!/bin/bash
set -eu -o pipefail # Exit on error, unset variable, or failed pipe

# --- OAR Job Directives ---

#OAR -n Final_SwinUNet_CombinedLoss_WarmRestart
#OAR -l nodes=1,walltime=30:00:00
#OAR -q abaca
#OAR -p musa-1 OR musa-2 OR musa-3 OR musa-4 OR musa-5 OR musa-6 OR esterel42-1 OR esterel43-1

# --- Configuration ---
PROJECT_DIR='/home/gkhanal/fwi_project/fwi_test/swinv2'
RUNTIME_BASE_DIR='/tmp/gkhanal-runtime-dir'
OUTPUT_DIR="${PROJECT_DIR}/results/july27" 
RUNTIME_DATA_DIR="${RUNTIME_BASE_DIR}/data/train" 
DATASET_ZIP='openfwi-preprocessed-72x72.zip'
EXTRACTED_DATA_DIR="openfwi_72x72" 
CHECKPOINT_PATH="{PROJECT_DIR}/results/july29"
SCRIPT_NAME="FINAL_SwinUNet_CombinedLoss_WarmRestartLR.py"

# --- Start Execution ---
echo "==============================================" 
echo "Script started at $(date)"                    
echo "Host: $(hostname)"                             
echo "User: $(whoami)"                               
echo "Runtime data directory: ${RUNTIME_DATA_DIR}"   
echo "=============================================="

# --- 1. Initialize Environment ---
echo -e '\n=== 1. Initializing Environment ==='

# Source conda and check environment
echo 'Initializing conda...'
source ~/.bashrc || { echo 'Error: Could not source ~/.bashrc'; exit 1; }

# Verify conda is available
if ! command -v conda &> /dev/null; then 
    echo "Error: conda not found in PATH"; exit 1
fi

echo 'Activating conda environment...'
conda activate fwi_training || { echo 'Error: Could not activate conda environment "fwi_test"'; exit 1; }
echo 'Conda environment 'fwi_training' activated.'

# Add local pip packages path
echo 'Adding ~/.local/bin to PATH...'
export PATH="${PATH}:${HOME}/.local/bin" # Corrected: Use double quotes for variable expansion

# --- 2. Setup Data Directory ---
echo -e '\n=== 2. Setting Up Data Directory ==='

echo "Using runtime directory: ${RUNTIME_DATA_DIR}" # Corrected: Use double quotes
mkdir -p "${RUNTIME_DATA_DIR}" || { echo 'Error: Could not create runtime data directory'; exit 1; } # Corrected: Use double quotes
cd "${RUNTIME_DATA_DIR}" || { echo 'Error: Could not cd to runtime data directory'; exit 1; } # Corrected: Use double quotes

# Check for existing data
if [ -f "${DATASET_ZIP}" ]; then # Corrected: Use double quotes for consistency (works with single here too)
    echo 'Note: Dataset zip file already exists (skipping download)'
else
    # --- 3. Download Dataset ---
    echo -e '\n=== 3. Downloading Dataset ==='

    # Verify kaggle is installed
    if ! command -v kaggle &> /dev/null; then
        echo 'Error: kaggle CLI not found. Install with: pip install kaggle'
        exit 1
    fi

    echo 'Downloading Kaggle dataset...'
    kaggle datasets download -d brendanartley/openfwi-preprocessed-72x72 || {
        echo 'Error: Kaggle dataset download failed'; exit 1
    }
fi

# --- 4. Extract Dataset (if needed) ---
echo -e '\n=== 4. Extracting Dataset ==='

if [ ! -f "${DATASET_ZIP}" ]; then # Corrected: Use double quotes for consistency
    echo 'Error: Dataset zip file not found'; exit 1
fi

# Robust check for any openfwi directory
if [ ! -d "${EXTRACTED_DATA_DIR}" ]; then # Corrected: Use EXTRACTED_DATA_DIR variable
    echo "Unzipping dataset..."
    unzip -q "${DATASET_ZIP}" || { echo "Error: Unzipping dataset failed"; exit 1; }
    echo "Dataset extracted successfully to ${EXTRACTED_DATA_DIR}/" # Corrected: Use EXTRACTED_DATA_DIR
else
    echo "Note: Dataset already extracted in ${EXTRACTED_DATA_DIR}/ (skipping unzip)" # Corrected: Use EXTRACTED_DATA_DIR
fi

# --- 5. Verify Project Directory ---
echo -e "\n=== 5. Preparing Training ==="

echo "Checking project directory..."
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "Error: Project directory not found: ${PROJECT_DIR}"
    exit 1
fi

cd "${PROJECT_DIR}" || { echo "Error: Could not cd to project directory"; exit 1; }

# --- 6. GPU Configuration ---
echo -e "\n=== 6. GPU Configuration ==="

# Check for NVIDIA tools
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "Detected ${NUM_GPUS} GPU(s)"
else
    echo "Warning: nvidia-smi not found. Assuming CPU-only operation."
    NUM_GPUS=0
fi

# --- 7. Launch Training ---
echo -e "\n=== 7. Starting Training ==="

# Log environment info
echo "Environment Summary:"
echo "- Python: $(python3 --version 2>&1)"
echo "- PyTorch: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not available")"
echo "- CUDA: $(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "Not available")"

# Determine launch method
if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Starting DDP training on ${NUM_GPUS} GPUs..."
    torchrun --nproc_per_node="${NUM_GPUS}" "${SCRIPT_NAME}"
elif [ "${NUM_GPUS}" -eq 1 ]; then
    echo "Starting single-GPU training..."
    python3 "${SCRIPT_NAME}"
else
    echo "Starting CPU-only training..."
    python3 "${SCRIPT_NAME}"
fi

# --- Finalization ---
echo -e "\n=============================================="
echo "Script finished at $(date)" # Corrected: Use double quotes for command substitution
echo "Data directory contents:"
ls -lh "${OUTPUT_DIR}" || echo "No output directory found"
echo  "\n=============================================="
