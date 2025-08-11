# Swin-UNet: Full Waveform Inversion with Swin Transformers

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains a high-performance, feature-rich PyTorch pipeline for training a deep learning model for Full-Waveform Inversion (FWI) using the OpenFWI dataset. The model, based on a U-Net architecture with a **Swin Transformer v2** backbone, predicts subsurface velocity models from multi-source seismic data.

The codebase is designed for robustness and performance, incorporating modern deep learning techniques such as Distributed Data Parallel (DDP), torch.compile, mixed-precision training, and advanced data augmentation.

## Key Features

### Advanced Model Architecture

A U-Net style model with a powerful, pretrained Swin Transformer v2 encoder and an enhanced decoder featuring learned upsampling, residual blocks, and SCSE attention.

### High Performance Training

- **Distributed Data Parallel (DDP)** support for efficient multi-GPU training.
- **Automatic Mixed Precision (AMP)** with `bfloat16` for faster training and reduced memory usage.
- **`torch.compile` Integration**: Leverages PyTorch 2.0+ for significant speedups.
- **Optimized Data Loading**: High-performance `DataLoader` with persistent, pre-fetching workers.

### Comprehensive Loss Function

- **Huber loss** for robust regression
- **Gradient matching loss** for structural preservation
- **Total variation regularization** for smooth outputs

### State of the Art (SOTA) Techniques

- **Advanced LR Scheduling**: Cosine Annealing with Warm Restarts and a linear warmup phase.
- **Exponential Moving Average (EMA)** of model weights for improved generalization.
- **Gradient Clipping** and accumulation for stable training.

### Sophisticated Data Handling

- **Rich GPU-Based Augmentations**: Includes elastic deformation, fault simulation, Gaussian noise, and amplitude jitter.
- **Adaptive Augmentation**: Dynamically adjusts augmentation strength based on validation performance.
- **Stratified Data Splitting**: Maintains the proportion of different data groups between training and validation sets.

### Comprehensive Experiment Management

- **Centralized Configuration**: All parameters are managed in a single `Config` class.
- **Robust Checkpointing**: Save and resume training, including optimizer, scheduler, and RNG states.
- **Detailed Logging**: Logs to console, file, and TensorBoard for real-time monitoring.
- **Automated Visualization**: Generates plots of training metrics and validation predictions.
- **ONNX Export**: Easily export the final model for deployment.

## Setup and Installation

### 1. Clone the repository

    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name

### 2. Create a virtual environment (recommended):

    python -m venv venv
    source venv/bin/activate

### 3. Install the required dependencies:

A `requirements.txt` file can be created with the following contents.

    # Core DL Framework
    torch
    torchvision
    torchaudio

    # Model and Utilities
    timm
    numpy
    psutil
    scipy
    matplotlib
    torchinfo
    tqdm
    torchmetrics
    tensorboard

Install them using pip:

    pip install -r requirements.txt

## Dataset Structure

The script expects the training data to be organized in a specific structure. The root directory contains subdirectories for different data categories (e.g., "Vel", "Style", "Fault"). Each subdirectory contains the input seismic data and target velocity models as `.npy` files.

The script automatically pairs `seis*.npy` with `vel*.npy` or `data*.npy` with `model*.npy`.

    /path/to/your/dataset/
    ├── Vel_Style_1/
    │   ├── seis0.npy
    │   ├── vel0.npy
    │   ├── seis1.npy
    │   └── vel1.npy
    │
    ├── Fault_Models/
    │   ├── data0.npy
    │   ├── model0.npy
    │   ├── data1.npy
    │   └── model1.npy

## Configuration

All hyperparameters and settings are managed in the `Config` class within the script. Before running, update the paths and review the key parameters.

## How to Run

### Single-GPU Training

To run the training on a single GPU, simply execute the Python script:

    python fwi_swinunet_pipeline.py

### Multi-GPU Training (DDP)

For distributed training across multiple GPUs on a single node, use `torchrun`. This script will automatically detect the DDP environment and handle the setup.

    # Replace NUM_GPUS with the number of GPUs you want to use (e.g., 4)
    torchrun --nproc_per_node=NUM_GPUS fwi_swinunet_pipeline.py

## Outputs and Results

The script will create an output directory (specified by `OUTPUT_DIR`) containing the following:

- Log File: A detailed log of the entire training process.
- Checkpoints: 

    - `best_model_ema.pth`: The model state with the best validation score (based on EMA weights).
    - `checkpoint_epoch_*.pth`: Periodic checkpoints saved every `CHECKPOINT_EVERY` epochs.

- TensorBoard Logs:

    tensorboard --logdir /path/to/your/output_dir/tensorboard

- Plots:

    - `*_metrics.png`: A comprehensive plot showing training/validation loss, MAE, learning rate, and gradient norm over epochs
    - `*_predictions.png`: A visual comparison of ground truth velocity models, model predictions, and their absolute difference on a sample of the validation set.

- ONNX Model:

    - `*.onnx`: The final model exported to the ONNX format if `EXPORT_ONNX` is `True`.

## Code Structure

The script is organized into logical, reusable components:

- `Config`: A centralized class for all hyperparameters and settings.
- `DDPManager`: Handles the setup and cleanup of the distributed training environment.
- Data and Preprocessing:

    - `FWIDataset`: Custom PyTorch Dataset for loading `.npy` files efficiently.
    - `GPUBatchProcessor`: Performs all batch processing (normalization, augmentation) on the GPU to maximize throughput.

- Model Architecture (`MultiSourceUNetSwin`):

    - `SCSEBlock`: Squeeze-and-Channel-and-Spatial-Excitation attention module.
    - `LearnedUpsample`: An upsampling block using `PixelShuffle` for higher-quality outputs.
    - `EnhancedUNetDecoderBlock`: A sophisticated decoder block combining upsampling, skip connections, and residual blocks.

- Utilities (`ModelEMA`, `EarlyStopping`, `MetricsLogger`): Helper classes for EMA, early stopping, and logging.

- Loss Function (`CombinedLoss`): A custom loss function combining Huber, Gradient, and Total Variation losses with support for class-based weighting.

- Training and Validation Loops: The core `train_one_epoch` and `validate_one_epoch` functions.

- `main()`: The main execution function that orchestrates the entire pipeline.

## Citation

If you use this code in your research, please cite:

    @misc{swinunet2025,
        author = {Gaurav Khanal},
        title = {Swin-UNet: Full Waveform Inversion with Swin Transformers},
        year = {2025},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/gk408829/Geophysical-Full-Waveform-Inversion}}
    }

