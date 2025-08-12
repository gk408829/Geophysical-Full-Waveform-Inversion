# Standard library imports
import os
import re
import random
import math
import time
import warnings
import sys
import copy
import logging
import datetime
import contextlib
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# Numerical and scientific computing
import numpy as np
import psutil
from scipy.ndimage import map_coordinates

# Visualization
import matplotlib.pyplot as plt

# Deep learning frameworks
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR
)
from torch.utils.tensorboard import SummaryWriter
import torch.profiler
from torchvision.transforms import functional as TF

# DDP and metrics
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import MeanAbsoluteError, StructuralSimilarityIndexMeasure

# Third-party libraries
import timm  # For Swin Transformer models
from torchinfo import summary  # For model summary
from tqdm.auto import tqdm  # Progress bars

# Warning suppressions
def suppress_all_warnings():
    """Suppress all common deep learning framework warnings"""
    # Python warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # PyTorch optimizations
    try:
        import torch
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

# Apply comprehensive warning suppression
suppress_all_warnings()

# Configurations

class Config:
    """
    Central configuration class for all model and training parameters.
    Provides default values and validation for the experiment setup.
    """

    # --- Paths, Logging & Resuming ---
    TRAIN_PATH = Path("/tmp/gkhanal-runtime-dir/data/train/openfwi_72x72")
    OUTPUT_DIR = Path("/home/gkhanal/fwi_project/fwi_test/swinv2/results/july25")
    EXPERIMENT_NAME = "SwinV2_FWI_Combined_Loss_LRWarmRestart"
    RESUME_CHECKPOINT = None  # Path to checkpoint file to resume training

    # --- Experiment Controls ---
    RUN_PROFILER = False            # Enable PyTorch profiler
    EXPORT_ONNX = True              # Export final model to ONNX format
    USE_LOSS_WEIGHTING = True       # Whether to use class-weighted loss

    # --- Model & Architecture ---
    MODEL_NAME = 'swinv2_tiny_window8_256.ms_in1k'  # Timm model name
    PRETRAINED = True                               # Use pretrained weights
    OUTPUT_CHANNELS = 1                             # Output velocity channels
    NUM_SOURCES = 5                                 # Number of seismic sources
    DECODER_CHANNELS = [512, 256, 128]              # Enhanced decoder channel sizes
    DECODER_DROPOUT = 0.2                           # Dropout rate in decoder
    USE_ENHANCED_DECODER = True                     # Use enhanced decoder with learned upsampling

    # Model dimensions
    INPUT_HEIGHT = 72                               # Input seismic data height
    INPUT_WIDTH = 72                                # Input seismic data width
    OUTPUT_HEIGHT = 70                              # Output velocity model height
    OUTPUT_WIDTH = 70                               # Output velocity model width
    BACKBONE_INPUT_SIZE = 256                       # Interpolation size for backbone
    STEM_CHANNELS = 32                              # Initial stem convolution channels
    RGB_CHANNELS = 3                                # RGB channels for backbone input
    GROUPNORM_GROUPS = 8                            # GroupNorm groups
    SCSE_REDUCTION = 16                             # Default reduction factor in SCSE block

    # --- Loss function weights ---
    HUBER_LOSS_WEIGHT = 0.4         # Weight for Huber loss
    GRAD_LOSS_WEIGHT = 0.55         # Weight for gradient loss
    TV_LOSS_WEIGHT = 0.05           # Weight for total variation loss

    # --- Training ---
    DEVICE = "cuda"                 # Default device
    USE_AMP = True                  # Automatic Mixed Precision
    NUM_EPOCHS = 50                 # Total training epochs
    BATCH_SIZE = 256                # Batch size per GPU
    ACCUMULATION_STEPS = 1          # Gradient accumulation steps
    WEIGHT_DECAY = 1e-4             # Weight decay
    GRAD_CLIP_NORM = 1.0            # Gradient clipping norm

    # --- Learning Rate Scheduler
    LEARNING_RATE = 1.5e-4          # Initial learning rate (increased for stronger restarts)
    LR_MIN = 1e-6                   # Minimum learning rate
    WARMUP_EPOCHS = 5               # Warmup epochs
    WARMUP_LR_START_FACTOR = 0.01   # Starting factor for warmup learning rate

    # --- Warm Restarts Configuration
    USE_WARM_RESTARTS = True        # Use CosineAnnealingWarmRestarts instead of CosineAnnealingLR
    T_0 = 8                         # Initial restart period (shorter for more exploration)
    T_MULT = 2                    # Factor to increase T_i after each restart (more gradual)
    ETA_MIN_RESTART = 5e-7          # Minimum learning rate for restarts (lower for deeper exploration)
    LR_RESTART_DETECTION_THRESHOLD = 2.0   # Threshold for detecting learning rate restarts
    LR_RESTART_SECONDARY_THRESHOLD = 1.5   # Secondary threshold for restart detection
    LR_RESTART_MIN_INCREASE_RATIO = 10.0   # Minimum increase ratio from bottom for restart detection

    # --- Validation, Checkpointing & EMA ---
    VALIDATION_SPLIT = 0.20         # Validation set fraction
    PATIENCE = 5                    # Early stopping patience
    CHECKPOINT_EVERY = 5            # Save checkpoint every N epochs
    EMA_DECAY = 0.99                # EMA decay rate

    # --- Data ---
    RANDOM_SEED = 42                # Random seed for reproducibility
    VELOCITY_MIN = 1500.0           # Minimum velocity value (m/s)
    VELOCITY_MAX = 4500.0           # Maximum velocity value (m/s)
    MAX_RETRIES = 3                 # Max retries in dataset loading
    MIN_STD_CLAMP = 1e-6            # Minimum std clamp value for normalization

    # --- Training Constants ---
    DDP_TIMEOUT_SECONDS = 60        # DDP timeout seconds
    VALIDATION_BATCH_MULTIPLIER = 2 # Batch size multiplier for validation
    ELASTIC_ALPHA_RANGE = (30, 50)  # Alpha range for elastic deformation
    ELASTIC_SIGMA_RANGE = (4, 6)    # Sigma range for elastic deformation
    ONNX_OPSET_VERSION = 13         # ONNX opset version

    # --- Torch Compile ---
    USE_TORCH_COMPILE = False         # Enable torch.compile for model optimization
    COMPILE_MODE = "default"         # Compile mode: "default", "reduce-overhead", "max-autotune"

    # --- Dataloader ---
    NUM_WORKERS = 48                # DataLoader workers
    PIN_MEMORY = True               # Pin memory for faster transfer
    PERSISTENT_WORKERS = True       # Maintain workers between epochs
    PREFETCH_FACTOR = 8             # Prefetch batches
    MEMORY_WORKERS_RATIO = 2        # GB of memory per worker for optimal worker calculation

    # Augmentation configuration classes
    class AugmentationToggles:
        """Toggle switches for different augmentation types"""
        AMP_JITTER = False          # Amplitude jitter
        RECEIVER_DROP = False       # Random receiver dropout
        GAUSSIAN_NOISE = True       # Add Gaussian noise
        VELOCITY_AUG = True         # Velocity scaling
        VELOCITY_SMOOTH = False     # Velocity smoothing
        FAULT_SIMULATION = False    # Fault simulation
        ELASTIC_DEFORM = False      # Elastic deformation

    class AugmentationParams:
        """Parameters for data augmentations"""
        # Noise parameters
        NOISE_STD = 0.02            # Std of Gaussian noise
        RECEIVER_DROP_PROB = 0.3    # Probability of receiver drop
        MAX_RECEIVER_DROPS = 5      # Maximum receivers to drop

        # Amplitude jitter
        AMP_JITTER_PROB = 0.2       # Probability of amplitude jitter
        AMP_JITTER_SCALE = 0.1      # Scale of amplitude variation

        # Fault simulation
        FAULT_NOISE_PROB = 0.2      # Probability of fault noise
        FAULT_NOISE_STRENGTH = 0.1  # Strength of fault displacement

        # Velocity augmentations
        VEL_AUG_PROB = 0.2          # Probability of velocity scaling
        VEL_AUG_SCALE = 0.1         # Scale of velocity variation
        VEL_SMOOTH_PROB = 0.2       # Probability of smoothing

        # Elastic deformation
        ELASTIC_DEFORM_PROB = 0.3   # Probability of elastic deformation
        GAUSSIAN_BLUR_KERNEL_MULTIPLIER = 6    # Multiplier for gaussian blur kernel size (6 * sigma + 1)

    def validate(self) -> None:
        """Validate configuration parameters"""
        if not self.TRAIN_PATH.exists():
            raise FileNotFoundError(f"Train path not found: {self.TRAIN_PATH}")
        if self.ACCUMULATION_STEPS > 0 and self.BATCH_SIZE % self.ACCUMULATION_STEPS != 0:
            raise ValueError(f"BATCH_SIZE ({self.BATCH_SIZE}) must be divisible by ACCUMULATION_STEPS ({self.ACCUMULATION_STEPS}) for proper gradient accumulation.")
        if self.NUM_WORKERS < 0:
            raise ValueError(f"NUM_WORKERS must be non-negative, got {self.NUM_WORKERS}")
        if self.BATCH_SIZE <= 0:
            raise ValueError(f"BATCH_SIZE must be positive, got {self.BATCH_SIZE}")
        if self.NUM_EPOCHS <= 0:
            raise ValueError(f"NUM_EPOCHS must be positive, got {self.NUM_EPOCHS}")

# Initialize configuration
cfg = Config()
augs = cfg.AugmentationToggles()
aug_params = cfg.AugmentationParams()

# Distributed Data Parallel (DDP) Manager)

class DDPManager:
    """
    Handles distributed training setup and cleanup.
    Automatically falls back to single-GPU mode if DDP initialization fails.
    """

    def __init__(self) -> None:
        """Initialize DDP environment if available"""
        self.is_ddp = 'WORLD_SIZE' in os.environ and torch.cuda.is_available()

        if self.is_ddp:
            try:
                # Initialize distributed process group
                dist.init_process_group(
                    backend="nccl",
                    timeout=datetime.timedelta(seconds=cfg.DDP_TIMEOUT_SECONDS)
                )
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                device_id = self.rank % torch.cuda.device_count()
                self.device = torch.device(f'cuda:{device_id}')
                torch.cuda.set_device(device_id)
            except Exception as e:
                logging.error(f"DDP initialization failed: {e}. Falling back to single-GPU.")
                self.is_ddp = False
                self._setup_single_gpu()
        else:
            self._setup_single_gpu()

    def _setup_single_gpu(self) -> None:
        """Setup for single-GPU training"""
        self.rank = 0
        self.world_size = 1
        self.device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    def cleanup(self) -> None:
        """Cleanup distributed process group if initialized"""
        if self.is_ddp and dist.is_initialized():
            dist.destroy_process_group()

    def is_main_process(self) -> bool:
        """Check if current process is the main process"""
        return self.rank == 0

# Setup Utilities

def set_seed(seed_value: int = cfg.RANDOM_SEED, rank: int = 0) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed_value: Base seed value
        rank: Process rank to ensure different seeds across processes
    """
    seed = seed_value + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id: int) -> None:
    """Seed numpy and random for DataLoader workers"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_logging(log_file: str = 'training.log') -> None:
    """Configure logging to file and console"""
    log_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

def safe_barrier(ddp_manager: DDPManager, timeout_seconds: int = cfg.DDP_TIMEOUT_SECONDS) -> None:
    """
    A DDP barrier with timeout to prevent hangs.

    Args:
        ddp_manager: DDPManager instance
        timeout_seconds: Maximum time to wait for barrier
    """
    if ddp_manager.is_ddp:
        try:
            dist.barrier(device_ids=[ddp_manager.rank])
        except RuntimeError as e:
            logging.error(f"DDP barrier timed out after {timeout_seconds}s: {e}")
            raise

def validate_ddp_setup(ddp_manager: DDPManager) -> None:
    """Validate DDP communication between processes"""
    if not ddp_manager.is_ddp:
        return

    try:
        # Test communication with all_reduce
        tensor = torch.tensor([ddp_manager.rank], device=ddp_manager.device, dtype=torch.float)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = float(sum(range(ddp_manager.world_size)))
        assert tensor.item() == expected, f"DDP validation failed: got {tensor.item()}, expected {expected}"

        if ddp_manager.is_main_process():
            logging.info("DDP communication validated successfully.")
    except Exception as e:
        logging.error(f"DDP validation failed: {e}", exc_info=True)
        raise

def validate_scheduler_config(train_loader: DataLoader) -> None:
    """Validate warm restart scheduler configuration"""
    if not cfg.USE_WARM_RESTARTS:
        return

    steps_per_epoch = len(train_loader) // cfg.ACCUMULATION_STEPS
    warmup_steps = cfg.WARMUP_EPOCHS * steps_per_epoch
    total_steps = cfg.NUM_EPOCHS * steps_per_epoch
    remaining_steps = total_steps - warmup_steps
    t_0_steps = cfg.T_0 * steps_per_epoch

    # Calculate expected restart schedule
    restart_steps = []
    current_period = t_0_steps
    current_step = warmup_steps + current_period

    while current_step < total_steps:
        restart_steps.append(current_step)
        current_period = int(current_period * cfg.T_MULT)
        current_step += current_period

    logging.info(f"Warm restart schedule validation:")
    logging.info(f"  - Total training steps: {total_steps}")
    logging.info(f"  - Warmup steps: {warmup_steps}")
    logging.info(f"  - Steps per epoch: {steps_per_epoch}")
    logging.info(f"  - Initial restart period: {t_0_steps} steps ({cfg.T_0} epochs)")
    logging.info(f"  - Expected restarts at steps: {restart_steps}")
    logging.info(f"  - Total expected restarts: {len(restart_steps)}")

    if len(restart_steps) == 0:
        logging.warning("No restarts will occur with current configuration!")
    elif len(restart_steps) < 2:
        logging.warning("Only 1 restart scheduled. Consider smaller T_0 for more exploration.")

def validate_model_outputs(model: nn.Module, sample_input: torch.Tensor, device: torch.device) -> None:
    """Validate model produces expected outputs without NaNs"""
    logging.info("Performing model output validation...")
    try:
        with torch.no_grad():
            output = model(sample_input.to(device))
            assert output.shape[-2:] == (cfg.OUTPUT_HEIGHT, cfg.OUTPUT_WIDTH), f"Expected output shape ending in ({cfg.OUTPUT_HEIGHT},{cfg.OUTPUT_WIDTH}), got {output.shape[-2:]}"
            assert not torch.isnan(output).any(), "Model produced NaN outputs on sample input"
            assert torch.isfinite(output).all(), "Model produced non-finite values (inf) on sample input"
        logging.info("Model output validation passed.")
    except Exception as e:
        logging.error(f"Model output validation FAILED: {e}", exc_info=True)
        raise

class AdaptiveAugmentation:
    """
    Dynamically adjusts augmentation strength based on validation performance.

    Attributes:
        strength: Current augmentation strength multiplier
        decay_rate: Rate at which strength decays
        patience: Number of validation checks without improvement before decay
        counter: Current patience counter
        best_loss: Best validation loss observed
    """

    def __init__(self, initial_strength: float = 1.0, decay_rate: float = 0.98, patience: int = 2) -> None:
        self.strength = initial_strength
        self.decay_rate = decay_rate
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def update(self, validation_loss: float) -> None:
        """Update augmentation strength based on validation loss"""
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                old_strength = self.strength
                self.strength *= self.decay_rate
                logging.info(f"Augmentation strength decayed from {old_strength:.3f} to {self.strength:.3f}")
                self.counter = 0

class WarmRestartMonitor:
    """Monitor and log warm restarts for CosineAnnealingWarmRestarts"""

    def __init__(self):
        self.last_lr = None
        self.restart_count = 0
        self.step_count = 0
        self.warmup_complete = False
        self.min_lr_seen = float('inf')

    def check_restart(self, current_lr: float) -> bool:
        """Check if a restart occurred based on learning rate increase"""
        restart_detected = False

        # Skip restart detection during warmup
        if not self.warmup_complete:
            if self.last_lr is not None and current_lr < self.last_lr:
                self.warmup_complete = True
                logging.info(f"Warmup completed at step {self.step_count}, starting restart monitoring")

        if self.warmup_complete and self.last_lr is not None:
            # Track minimum LR to detect restarts more reliably
            self.min_lr_seen = min(self.min_lr_seen, self.last_lr)

            # Detect restart: significant LR increase from recent minimum
            lr_increase_ratio = current_lr / self.last_lr
            min_increase_from_bottom = current_lr / self.min_lr_seen

            if (lr_increase_ratio > cfg.LR_RESTART_DETECTION_THRESHOLD or  # Primary threshold for restart detection
                (lr_increase_ratio > cfg.LR_RESTART_SECONDARY_THRESHOLD and min_increase_from_bottom > cfg.LR_RESTART_MIN_INCREASE_RATIO)):  # Secondary threshold + far from minimum

                self.restart_count += 1
                logging.info(f"🔄 Warm restart #{self.restart_count} detected at step {self.step_count} "
                            f"(LR: {self.last_lr:.2e} → {current_lr:.2e}, ratio: {lr_increase_ratio:.1f}x)")
                self.min_lr_seen = float('inf')  # Reset minimum tracking
                restart_detected = True

        self.last_lr = current_lr
        self.step_count += 1
        return restart_detected

# Data and Preprocessing

def scan_files(root_dir: Path) -> List[Dict[str, Any]]:
    """Scan directory for input/target file pairs and group them by type"""
    file_pairs = []
    for item in sorted(root_dir.iterdir()):
        if not item.is_dir():
            continue

        name = item.name
        group = "Unknown"
        if "Vel" in name:
            group = "Vel"
        elif "Style" in name:
            group = "Style"
        elif "Fault" in name:
            group = "Fault"

        # Match data/model pairs
        for data_file in sorted(item.glob("data*.npy")):
            match = re.search(r"data(\d+)\.npy", data_file.name)
            if match:
                idx = match.group(1)
                model_file = item / f"model{idx}.npy"
                if model_file.exists():
                    file_pairs.append({"input": data_file, "target": model_file, "group": group})

        # Match seis/vel pairs
        for seis_file in sorted(item.glob("seis*.npy")):
            vel_file = item / seis_file.name.replace("seis", "vel", 1)
            if vel_file.exists():
                file_pairs.append({"input": seis_file, "target": vel_file, "group": group})

    return file_pairs

def create_stratified_split(file_pairs: List[Dict[str, Any]], val_frac: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create stratified train/validation split maintaining group proportions"""
    groups = defaultdict(list)
    for pair in file_pairs:
        groups[pair["group"]].append(pair)

    train_files, val_files = [], []
    for group, items in groups.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val_files.extend(items[:n_val])
        train_files.extend(items[n_val:])

    random.shuffle(train_files)
    random.shuffle(val_files)
    return train_files, val_files

def calculate_class_weights(train_pairs: List[Dict[str, Any]], ddp_manager: DDPManager) -> Optional[Dict[str, float]]:
    """Calculate class weights for loss balancing"""
    if not cfg.USE_LOSS_WEIGHTING:
        return None

    logging.info("Calculating class weights for loss balancing...")
    group_counts = defaultdict(int)
    total_samples = 0

    for pair in train_pairs:
        group_counts[pair['group']] += 1
        total_samples += 1

    if total_samples == 0:
        return None

    num_classes = len(group_counts)
    weights = {}

    # Inverse frequency weighting
    for group, count in group_counts.items():
        weights[group] = total_samples / (num_classes * count)

    # Normalize weights
    max_weight = max(weights.values())
    for group in weights:
        weights[group] /= max_weight

    if ddp_manager.is_main_process():
        logging.info(f"Calculated loss weights: {weights}")

    return weights

def log_dataset_info(train_pairs: List[Dict[str, Any]], val_pairs: List[Dict[str, Any]]) -> None:
    """Log detailed information about dataset splits"""
    logging.info("--- Dataset Analysis ---")

    def analyze_split(name: str, pairs: List[Dict]) -> None:
        if not pairs:
            logging.info(f"{name} set is empty.")
            return

        group_counts = defaultdict(int)
        total_samples = 0

        for pair in tqdm(pairs, desc=f"Scanning {name} files", leave=False):
            try:
                with open(pair['input'], 'rb') as f:
                    version = np.lib.format.read_magic(f)
                    shape, _, _ = np.lib.format._read_array_header(f, version)
                    num_samples_in_file = shape[0]

                group = pair['group']
                group_counts[group] += num_samples_in_file
                total_samples += num_samples_in_file
            except Exception as e:
                logging.warning(f"Could not read shape for {pair['input']}: {e}")

        logging.info(f"Split: {name} | Total Samples: {total_samples}")
        for group, count in sorted(group_counts.items()):
            logging.info(f"  - Group: {group:<6} | Samples: {count}")

    analyze_split("Training", train_pairs)
    analyze_split("Validation", val_pairs)
    logging.info("------------------------")

class FWIDataset(Dataset):
    """Dataset for loading FWI training samples"""

    def __init__(self, metadata: List[Dict[str, Any]], device: torch.device) -> None:
        """
        Args:
            metadata: List of dicts containing input/target file paths
            device: Target device for data loading
        """
        self.metadata_files = metadata
        self.device = device
        self.flat_index = []

        # Build flat index mapping (file_idx, sample_idx)
        for file_idx, pair in enumerate(self.metadata_files):
            try:
                num_samples = np.load(pair["input"], mmap_mode='r').shape[0]
                self.flat_index.extend([(file_idx, sample_idx) for sample_idx in range(num_samples)])
            except Exception as e:
                logging.error(f"Failed to process file {pair['input']}: {e}")

    def __len__(self) -> int:
        return len(self.flat_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Load and preprocess a single sample with retry mechanism"""
        max_retries = cfg.MAX_RETRIES

        for attempt in range(max_retries):
            try:
                file_idx, sample_idx = self.flat_index[idx]
                sample_info = self.metadata_files[file_idx]

                # Load input and target data
                input_data = np.load(sample_info['input'], mmap_mode='r')[sample_idx].copy()
                target_data = np.load(sample_info['target'], mmap_mode='r')[sample_idx].copy()
                group = sample_info['group']

                # Validate data integrity
                if np.isnan(input_data).any() or np.isnan(target_data).any():
                    raise ValueError(f"NaN values detected in data for sample {idx}")

                if input_data.size == 0 or target_data.size == 0:
                    raise ValueError(f"Empty data arrays for sample {idx}")

                # Convert to tensors
                seismic = torch.from_numpy(input_data).float()
                velocity = torch.from_numpy(target_data).float()

                # Handle different input formats
                if group == 'Fault' and seismic.ndim == 2:
                    seismic = seismic.unsqueeze(0).repeat(cfg.NUM_SOURCES, 1, 1)
                if velocity.ndim == 2:
                    velocity = velocity.unsqueeze(0)

                return seismic, velocity, group

            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Attempt {attempt + 1} failed for sample {idx}: {e}. Retrying...")
                    # Try a different sample on retry to avoid persistent corruption
                    idx = (idx + 1) % len(self.flat_index)
                    continue
                else:
                    # All retries exhausted - raise error to fail fast
                    logging.error(f"Failed to load sample {idx} after {max_retries} attempts: {e}")
                    raise RuntimeError(f"Failed to load sample {idx} after {max_retries} attempts") from e

class GPUBatchProcessor:
    """Handles batch processing and augmentations on GPU"""

    def __init__(self, device: torch.device, aug_scheduler: AdaptiveAugmentation) -> None:
        """
        Args:
            device: Target device for processing
            aug_scheduler: Adaptive augmentation controller
        """
        self.device = device
        self.aug_scheduler = aug_scheduler
        self.velocity_min = torch.tensor(cfg.VELOCITY_MIN, device=device)
        self.velocity_range = torch.tensor(cfg.VELOCITY_MAX - cfg.VELOCITY_MIN, device=device)

    def _elastic_deform(self, image: torch.Tensor, alpha: float, sigma: float) -> torch.Tensor:
        """Apply elastic deformation to input images"""
        B, _, H, W = image.shape
        coords_y, coords_x = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        coords = torch.stack([coords_y, coords_x], dim=0).float()

        # Generate random displacement fields
        dx = torch.randn(B, H, W, device=self.device) * sigma
        dy = torch.randn(B, H, W, device=self.device) * sigma

        # Smooth displacement fields
        kernel_size = int(aug_params.GAUSSIAN_BLUR_KERNEL_MULTIPLIER * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        dx = TF.gaussian_blur(dx.unsqueeze(1), kernel_size=(kernel_size, kernel_size)).squeeze(1)
        dy = TF.gaussian_blur(dy.unsqueeze(1), kernel_size=(kernel_size, kernel_size)).squeeze(1)

        # Apply displacement
        displaced_coords = coords + alpha * torch.stack([dy, dx], dim=1)
        displaced_coords = displaced_coords.permute(0, 2, 3, 1)
        norm_coords = 2 * (displaced_coords / torch.tensor([W-1, H-1], device=self.device) - 0.5)

        return F.grid_sample(
            image, norm_coords,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=False
        )

    def _augment_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        """Apply velocity-specific augmentations"""
        aug_strength = self.aug_scheduler.strength

        # Velocity scaling
        if augs.VELOCITY_AUG and torch.rand(1) < aug_params.VEL_AUG_PROB * aug_strength:
            scales = 1.0 + (torch.rand(velocity.size(0), 1, 1, 1, device=self.device) * 2 - 1) * aug_params.VEL_AUG_SCALE * aug_strength
            velocity = velocity * scales

        # Velocity smoothing
        if augs.VELOCITY_SMOOTH and torch.rand(1) < aug_params.VEL_SMOOTH_PROB * aug_strength:
            kernel_size = random.choice([3, 5])
            velocity = F.avg_pool2d(
                velocity,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            )

        return torch.clamp(velocity, cfg.VELOCITY_MIN, cfg.VELOCITY_MAX)

    def _simulate_faults(self, velocity: torch.Tensor) -> torch.Tensor:
        """Simulate geological faults in velocity models"""
        batch_size, _, h, w = velocity.shape
        x, y = torch.arange(w, device=self.device), torch.arange(h, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        # Generate random fault lines
        angles = torch.rand(batch_size, device=self.device) * math.pi
        offsets = (torch.rand(batch_size, device=self.device) * 2 - 1) * math.sqrt(h**2+w**2) * 0.2
        mask = (xx * torch.cos(angles).view(-1, 1, 1) + yy * torch.sin(angles).view(-1, 1, 1)) > offsets.view(-1, 1, 1)

        # Apply displacement along faults
        displacement = (torch.rand_like(velocity) * 2 - 1) * self.velocity_range * aug_params.FAULT_NOISE_STRENGTH
        velocity[mask.unsqueeze(1)] += displacement[mask.unsqueeze(1)]

        return torch.clamp(velocity, cfg.VELOCITY_MIN, cfg.VELOCITY_MAX)

    def process_batch(self, seismic: torch.Tensor, velocity: Optional[torch.Tensor] = None,
                     groups: Optional[List[str]] = None, is_train: bool = False) -> Tuple:
        """
        Process a batch of data including normalization and augmentations.

        Args:
            seismic: Input seismic data
            velocity: Target velocity data (optional)
            groups: Data group labels (optional)
            is_train: Whether to apply training augmentations

        Returns:
            Tuple of (processed_seismic, normalized_velocity, denormalized_velocity)
        """
        # Transfer to device
        seismic = seismic.to(self.device, non_blocking=True)
        original_velocity = velocity
        if velocity is not None:
            velocity = velocity.to(self.device, non_blocking=True)

        # Apply augmentations during training
        aug_strength = self.aug_scheduler.strength if is_train else 0.0

        if is_train and velocity is not None:
            # Elastic deformation
            if augs.ELASTIC_DEFORM and torch.rand(1) < aug_params.ELASTIC_DEFORM_PROB * aug_strength:
                velocity = self._elastic_deform(
                    velocity,
                    alpha=random.uniform(*cfg.ELASTIC_ALPHA_RANGE),
                    sigma=random.uniform(*cfg.ELASTIC_SIGMA_RANGE)
                )

            # Fault simulation
            if augs.FAULT_SIMULATION and torch.rand(1) < aug_params.FAULT_NOISE_PROB * aug_strength:
                velocity = self._simulate_faults(velocity)

            # Amplitude jitter
            if augs.AMP_JITTER and torch.rand(1) < aug_params.AMP_JITTER_PROB * aug_strength:
                seismic *= (1.0 + (torch.rand(seismic.size(0), 1, 1, 1, device=self.device) * 2 - 1) * aug_params.AMP_JITTER_SCALE)

            # Receiver dropout
            if augs.RECEIVER_DROP and torch.rand(1) < aug_params.RECEIVER_DROP_PROB * aug_strength:
                num_drops = torch.randint(1, aug_params.MAX_RECEIVER_DROPS + 1, (1,)).item()
                if seismic.dim() == 4 and seismic.size(1) == cfg.NUM_SOURCES:
                    perm = torch.rand(seismic.size(0), cfg.NUM_SOURCES, device=self.device).argsort(dim=1)
                    seismic[torch.arange(seismic.size(0), device=self.device).unsqueeze(1), perm[:, :num_drops]] = 0.0

            # Gaussian noise
            if augs.GAUSSIAN_NOISE:
                seismic += torch.randn_like(seismic) * aug_params.NOISE_STD * aug_strength

        # Normalize seismic data
        mean = seismic.mean(dim=(-1, -2), keepdim=True)
        std = torch.clamp(seismic.std(dim=(-1, -2), keepdim=True), min=cfg.MIN_STD_CLAMP)
        seismic = (seismic - mean) / std

        # Handle NaN values
        if torch.isnan(seismic).any():
            logging.warning("NaN values detected in seismic data after normalization. Replacing with zeros.")
            seismic = torch.nan_to_num(seismic)

        # Normalize velocity if provided
        if velocity is not None:
            norm_velocity = (velocity - self.velocity_min) / self.velocity_range
            return seismic, norm_velocity, original_velocity.to(self.device) if original_velocity is not None else None

        return seismic, None, None

    def denormalize(self, norm_vel: torch.Tensor) -> torch.Tensor:
        """Convert normalized velocity back to original scale"""
        return norm_vel * self.velocity_range + self.velocity_min

# Model Architecture

class SCSEBlock(nn.Module):
    """Squeeze-and-Excitation block with spatial attention"""

    def __init__(self, channels: int, reduction: int = cfg.SCSE_REDUCTION) -> None:
        super().__init__()
        # Channel attention
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        # Spatial attention
        self.sSE = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cSE(x) + x * self.sSE(x)

class LearnedUpsample(nn.Module):
    """Learned upsampling using PixelShuffle for smooth, artifact-free upsampling"""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (scale_factor ** 2),
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.norm = nn.GroupNorm(cfg.GROUPNORM_GROUPS, out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        return self.activation(x)

class ResidualBlock(nn.Module):
    """Residual block for decoder with improved skip connections"""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(cfg.GROUPNORM_GROUPS, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(cfg.GROUPNORM_GROUPS, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        self.scse = SCSEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)

        # Add residual connection
        x = x + residual
        x = self.activation(x)

        # Apply attention
        x = self.scse(x)

        return x

class EnhancedUNetDecoderBlock(nn.Module):
    """Enhanced decoder block with learned upsampling and improved residual connections"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()

        # Learned upsampling instead of simple interpolation
        self.learned_upsample = LearnedUpsample(in_ch, in_ch)

        # Channel reduction for concatenated features
        concat_ch = in_ch + skip_ch
        self.channel_reduce = nn.Conv2d(concat_ch, out_ch, 1, bias=False)
        self.reduce_norm = nn.GroupNorm(cfg.GROUPNORM_GROUPS, out_ch)

        # Residual processing blocks
        self.res_block1 = ResidualBlock(out_ch, cfg.DECODER_DROPOUT)
        self.res_block2 = ResidualBlock(out_ch, cfg.DECODER_DROPOUT)

        # Skip connection for the entire block
        self.skip_conv = nn.Conv2d(concat_ch, out_ch, 1, bias=False) if concat_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Learned upsampling
        x_up = self.learned_upsample(x)

        # Concatenate with skip connection
        x_concat = torch.cat([x_up, skip], dim=1)

        # Store for skip connection
        skip_input = x_concat

        # Channel reduction
        x = self.channel_reduce(x_concat)
        x = self.reduce_norm(x)
        x = F.gelu(x)

        # Residual processing
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Block-level skip connection
        if not isinstance(self.skip_conv, nn.Identity):
            skip_processed = self.skip_conv(skip_input)
            x = x + skip_processed

        return x

class MultiSourceUNetSwin(nn.Module):
    """Main model combining Swin Transformer backbone with UNet-like decoder"""

    def __init__(self, ddp_manager: DDPManager) -> None:
        super().__init__()
        # Initial stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(1, cfg.STEM_CHANNELS, 3, padding=1, bias=False),
            nn.GroupNorm(cfg.GROUPNORM_GROUPS, cfg.STEM_CHANNELS),
            nn.GELU(),
            nn.Conv2d(cfg.STEM_CHANNELS, cfg.STEM_CHANNELS, 3, padding=1, bias=False),
            nn.GroupNorm(cfg.GROUPNORM_GROUPS, cfg.STEM_CHANNELS),
            nn.GELU(),
            nn.Conv2d(cfg.STEM_CHANNELS, cfg.RGB_CHANNELS, 1, bias=False)
        )

        # Load backbone (Swin Transformer V2) with warning suppression
        if ddp_manager.is_ddp:
            # Ensure all processes load the same pretrained weights
            if ddp_manager.is_main_process():
                self.backbone = timm.create_model(
                    cfg.MODEL_NAME,
                    pretrained=cfg.PRETRAINED,
                    in_chans=cfg.RGB_CHANNELS,
                    features_only=True
                )
            safe_barrier(ddp_manager)
            if not ddp_manager.is_main_process():
                self.backbone = timm.create_model(
                    cfg.MODEL_NAME,
                    pretrained=cfg.PRETRAINED,
                    in_chans=cfg.RGB_CHANNELS,
                    features_only=True
                )
        else:
            self.backbone = timm.create_model(
                cfg.MODEL_NAME,
                pretrained=cfg.PRETRAINED,
                in_chans=cfg.RGB_CHANNELS,
                features_only=True
            )

        # Enable gradient checkpointing if available
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
            if ddp_manager.is_main_process():
                logging.info("Gradient checkpointing enabled for backbone.")

        # Build enhanced decoder with learned upsampling and residual blocks
        enc_channels = self.backbone.feature_info.channels()
        dec_channels = cfg.DECODER_CHANNELS

        if ddp_manager.is_main_process():
            logging.info(f"Building enhanced decoder with channels: {dec_channels}")
            logging.info(f"Encoder channels: {enc_channels}")

        decoder_layers = []
        in_ch = enc_channels[-1]

        for i in range(len(dec_channels)):
            skip_ch = enc_channels[-(i+2)]
            out_ch = dec_channels[i]
            decoder_layers.append(EnhancedUNetDecoderBlock(in_ch, skip_ch, out_ch))
            if ddp_manager.is_main_process():
                logging.info(f"Decoder layer {i}: {in_ch} + {skip_ch} -> {out_ch}")
            in_ch = out_ch

        self.decoders = nn.ModuleList(decoder_layers)

        # Enhanced final processing with residual refinement
        self.final_refine = nn.Sequential(
            ResidualBlock(dec_channels[-1], cfg.DECODER_DROPOUT * 0.5),  # Less dropout for final layer
            nn.Conv2d(dec_channels[-1], dec_channels[-1] // 2, 3, padding=1, bias=False),
            nn.GroupNorm(cfg.GROUPNORM_GROUPS, dec_channels[-1] // 2),
            nn.GELU(),
            nn.Conv2d(dec_channels[-1] // 2, cfg.OUTPUT_CHANNELS, 1)
        )

        # Log decoder enhancement info
        if ddp_manager.is_main_process():
            decoder_params = sum(p.numel() for p in self.decoders.parameters()) + sum(p.numel() for p in self.final_refine.parameters())
            total_params = sum(p.numel() for p in self.parameters())
            logging.info(f"Enhanced decoder parameters: {decoder_params:,} ({decoder_params/total_params*100:.1f}% of total)")
            logging.info("Decoder enhancements: Learned upsampling + Residual blocks + Enhanced final processing")

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode input through backbone and return feature maps"""
        x = self.stem(x)
        x = F.interpolate(x, size=(cfg.BACKBONE_INPUT_SIZE, cfg.BACKBONE_INPUT_SIZE), mode='bilinear', align_corners=False)
        features = self.backbone(x)

        # Handle potential channel-last format
        if features and features[0].ndim == 4 and features[0].shape[-1] == self.backbone.feature_info[0]['num_chs']:
            features = [o.permute(0, 3, 1, 2) for o in features]

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-source aggregation"""
        B, S, H, W = x.shape

        # Process each source independently
        x_batched = x.reshape(B * S, 1, H, W)
        all_skips = self.encode(x_batched)

        # Average features across sources
        avg_skips = []
        for skip_tensor in all_skips:
            _, C, H_feat, W_feat = skip_tensor.shape
            reshaped_skip = skip_tensor.reshape(B, S, C, H_feat, W_feat)
            avg_skip = reshaped_skip.mean(dim=1)
            avg_skips.append(avg_skip)

        # Decode averaged features
        dec = avg_skips[-1]
        for i in range(len(self.decoders)):
            dec = self.decoders[i](dec, avg_skips[-(i+2)])

        # Final output with learned upsampling to target size
        if dec.shape[-2:] != (cfg.OUTPUT_HEIGHT, cfg.OUTPUT_WIDTH):
            dec = F.interpolate(dec, size=(cfg.OUTPUT_HEIGHT, cfg.OUTPUT_WIDTH), mode='bilinear', align_corners=False)

        return self.final_refine(dec)

"""# Utilities"""

class ModelEMA:
    """Exponential Moving Average of model weights"""

    def __init__(self, model: nn.Module, decay: float = cfg.EMA_DECAY) -> None:
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        """Update EMA weights"""
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.copy_(self.decay * ema_p + (1.0 - self.decay) * p)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.ema_model.load_state_dict(state_dict)

    def to(self, device: torch.device) -> 'ModelEMA':
        self.ema_model.to(device)
        return self

class EarlyStopping:
    """Early stopping based on validation metric"""

    def __init__(self, patience: int = cfg.PATIENCE, verbose: bool = True) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric: float, model_to_save: nn.Module, path: Path) -> None:
        """
        Check if early stopping criteria met

        Args:
            val_metric: Current validation metric
            model_to_save: Model to save if metric improves
            path: Path to save best model
        """
        if self.best_score is None or val_metric < self.best_score:
            self.best_score = val_metric
            if self.verbose:
                logging.info(f"Metric improved to {self.best_score:.4f}. Saving best EMA model.")
            torch.save(model_to_save.state_dict(), path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

class MetricsLogger:
    """Handles logging metrics to TensorBoard"""

    def __init__(self, log_dir: Path) -> None:
        self.writer = SummaryWriter(log_dir)

    def log(self, metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics for current epoch"""
        for key, value in metrics.items():
            if not math.isnan(value):
                self.writer.add_scalar(key, value, epoch)

    def close(self) -> None:
        self.writer.close()

def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ema: ModelEMA,
    path: Path,
    history: Dict[str, List[float]],
    ddp_manager: DDPManager
) -> None:
    # Ensure only the main process saves the checkpoint in DDP
    if hasattr(ddp_manager, 'is_main_process') and not ddp_manager.is_main_process():
        return

    try:
        logging.info(f"Saving checkpoint to {path} at epoch {epoch + 1}")

        # Ensure the directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get the unwrapped model state dict
        model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

        state = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ema_model_state_dict': ema.ema_model.state_dict() if ema else None,
            'history': history,
            # Save RNG states for reproducibility
            'rng_state_torch': torch.get_rng_state(),
            'rng_state_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'rng_state_numpy': np.random.get_state(),
            'rng_state_random': random.getstate(),
        }

        # Save to temporary file first, then rename for atomic operation
        temp_path = path.with_suffix('.tmp')
        torch.save(state, temp_path)
        temp_path.rename(path)

        logging.info(f"Successfully saved checkpoint to {path}")

    except Exception as e:
        logging.error(f"Failed to save checkpoint to {path}: {e}")
        # Clean up temporary file if it exists
        temp_path = path.with_suffix('.tmp')
        if temp_path.exists():
            temp_path.unlink()
        raise

def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ema: ModelEMA,
    ddp_manager: DDPManager
) -> Tuple[int, Dict[str, List[float]]]:

    if not checkpoint_path.exists():
        logging.warning(f"Checkpoint not found at {checkpoint_path}")
        return 0, defaultdict(list) # Return start_epoch 0 and empty history

    try:
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        # Use device-agnostic map_location
        map_location = ddp_manager.device if ddp_manager.is_ddp else 'cpu'

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Load model states (ensure model is unwrapped if DDP)
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer and scheduler states
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load EMA model if available
        if ema and 'ema_model_state_dict' in checkpoint and checkpoint['ema_model_state_dict'] is not None:
            ema.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

        # Restore history
        loaded_history = checkpoint.get('history', defaultdict(list))

        # Restore RNG states with rank-specific seeding for DDP
        if 'rng_state_torch' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state_torch'])
        if 'rng_state_cuda' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['rng_state_cuda'])
        if 'rng_state_numpy' in checkpoint:
            np.random.set_state(checkpoint['rng_state_numpy'])
        if 'rng_state_random' in checkpoint:
            random.setstate(checkpoint['rng_state_random'])

        # Re-seed with rank offset to maintain different seeds across processes
        if ddp_manager.is_ddp:
            set_seed(cfg.RANDOM_SEED, ddp_manager.rank)

        start_epoch = checkpoint['epoch'] + 1

        logging.info(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")

        # Sync all processes before continuing
        if ddp_manager.is_ddp:
            safe_barrier(ddp_manager)

        return start_epoch, loaded_history

    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        if ddp_manager.is_ddp:
            # Ensure all processes fail together
            safe_barrier(ddp_manager)
        raise

def get_scheduler(optimizer: torch.optim.Optimizer, train_loader: DataLoader):
    """Create learning rate scheduler with warmup and optional warm restarts"""
    warmup_steps = cfg.WARMUP_EPOCHS * len(train_loader) // cfg.ACCUMULATION_STEPS
    total_steps = cfg.NUM_EPOCHS * len(train_loader) // cfg.ACCUMULATION_STEPS

    if cfg.USE_WARM_RESTARTS:
        # For warm restarts, we need to account for the warmup period
        # The restart periods should start AFTER warmup
        remaining_steps = total_steps - warmup_steps
        t_0_steps = cfg.T_0 * len(train_loader) // cfg.ACCUMULATION_STEPS

        # Ensure T_0 doesn't exceed remaining training steps
        if t_0_steps > remaining_steps:
            t_0_steps = max(remaining_steps // 3, 1)  # At least 3 restarts or minimum 1 step
            logging.warning(f"T_0 too large for remaining steps. Adjusted to {t_0_steps} steps")

        warmup = LinearLR(
            optimizer,
            start_factor=cfg.WARMUP_LR_START_FACTOR,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0_steps,
            T_mult=cfg.T_MULT,
            eta_min=cfg.ETA_MIN_RESTART
        )

        logging.info(f"Using CosineAnnealingWarmRestarts with T_0={t_0_steps} steps ({cfg.T_0} epochs), "
                    f"T_mult={cfg.T_MULT}, starting after {warmup_steps} warmup steps")

        return SequentialLR(
            optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[warmup_steps]
        )
    else:
        warmup = LinearLR(
            optimizer,
            start_factor=cfg.WARMUP_LR_START_FACTOR,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=cfg.LR_MIN
        )

        logging.info(f"Using CosineAnnealingLR with T_max={total_steps - warmup_steps} steps")

        return SequentialLR(
            optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[warmup_steps]
        )

def rescale_to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """Rescale tensor to [0, 1] range for SSIM calculation"""
    min_v = x.amin(dim=(-2, -1), keepdim=True)
    max_v = x.amax(dim=(-2, -1), keepdim=True)
    return (x - min_v) / (max_v - min_v + 1e-8)

def plot_metrics(history: Dict[str, List[float]], output_dir: Path) -> None:
    """
    Plot and save training metrics.

    Args:
        history: Dictionary containing metric history
        output_dir: Directory to save plot image
    """
    logging.info("Plotting training history...")
    num_epochs = len(history['train_loss'])

    if num_epochs == 0:
        logging.warning("History is empty, skipping plotting.")
        return

    epochs = range(1, num_epochs + 1)
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f'Performance Metrics: {cfg.EXPERIMENT_NAME}', fontsize=16, fontweight='bold')

    # Plot 1: Normalized Training Loss and Denormalized MAE (Train and Val)
    ax_loss = axes[0, 0]
    p1, = ax_loss.plot(epochs, history["train_loss"], ".-", label="Train Loss (Normalized)", color="blue")
    ax_loss.set_ylabel("Normalized Loss", color="blue")
    ax_loss.tick_params(axis="y", labelcolor="blue")
    ax_loss.set_xlabel("Epoch")
    ax_loss.grid(True, ls="--")

    ax_mae = ax_loss.twinx()
    p2, = ax_mae.plot(epochs, history["train_denorm_mae"], ".-", label="Train MAE (Denormalized)", color="orange")
    p3, = ax_mae.plot(epochs, history["val_avg_mae"], ".-", label="Val MAE (Denormalized, EMA)", color="green")
    ax_mae.set_ylabel("MAE (m/s)", color="orange")
    ax_mae.tick_params(axis="y", labelcolor="orange")
    ax_mae.grid(False)  # Avoid grid overlap

    ax_loss.set_title("Normalized Loss & Denormalized MAE")
    ax_loss.legend(handles=[p1, p2, p3], loc='upper right')


    # Plot 2: Validation MAE by data group
    for group in ["Vel", "Style", "Fault"]:
        if f"val_{group}_mae" in history and history[f"val_{group}_mae"]:
            axes[0, 1].plot(epochs, history[f"val_{group}_mae"], '.-', label=f"{group} MAE")
    axes[0, 1].set_title('Validation MAE by Group (EMA)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, ls='--')

    # Plot 3: Validation SSIM by data group
    for group in ["Vel", "Style", "Fault"]:
        if f"val_{group}_ssim" in history and history[f"val_{group}_ssim"]:
            axes[1, 0].plot(epochs, history[f"val_{group}_ssim"], '.-', label=f"{group} SSIM")
    axes[1, 0].set_title('Validation SSIM by Group (EMA)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].legend()
    axes[1, 0].grid(True, ls='--')

    # Plot 4: Learning Rate and Gradient Norm
    ax_lr = axes[1, 1]
    p1, = ax_lr.plot(epochs, history['lr'], '.-', label='LR', color='purple')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.set_yscale('log')

    ax_gn = ax_lr.twinx()
    p2, = ax_gn.plot(epochs, history['grad_norm'], '.-', label='Grad Norm', color='teal', alpha=0.6)
    ax_gn.set_ylabel('Gradient Norm')

    ax_lr.set_title('Learning Rate & Gradient Norm')
    ax_lr.set_xlabel('Epoch')
    ax_lr.legend(handles=[p1, p2])
    ax_lr.grid(True, ls='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / f"{cfg.EXPERIMENT_NAME}_metrics.png")
    plt.close()

def visualize_predictions(model: nn.Module, loader: DataLoader,
                         batch_processor: GPUBatchProcessor, output_dir: Path):
    """Visualize model predictions on validation samples"""
    logging.info("Plotting validation predictions...")
    model.eval()

    # Step 1: Pre-select random samples to visualize (with error handling)
    grouped_indices = defaultdict(list)
    dataset_size = len(loader.dataset)
    max_samples_to_check = min(1000, dataset_size)  # Limit to avoid long iteration

    # Randomly sample indices to check instead of iterating through all
    indices_to_check = random.sample(range(dataset_size), max_samples_to_check)

    for i in indices_to_check:
        try:
            _, _, group = loader.dataset[i]
            grouped_indices[group].append(i)
        except Exception as e:
            logging.warning(f"Failed to access sample {i}: {e}")
            continue

    samples_to_plot_indices = []
    for group in ["Vel", "Style", "Fault"]:
        if len(grouped_indices[group]) > 0:
            indices = random.sample(grouped_indices[group], min(2, len(grouped_indices[group])))
            samples_to_plot_indices.extend(indices)

    if not samples_to_plot_indices:
        logging.warning("No samples found for visualization. Using first available samples from loader.")
        # Fallback: try to get samples directly from the loader
        try:
            for batch_idx, (seismic, velocity, groups) in enumerate(loader):
                if batch_idx == 0:  # Just use first batch
                    samples_to_plot_indices = list(range(min(6, len(groups))))
                    break
        except Exception as e:
            logging.error(f"Failed to get samples from loader: {e}")
            return

    # Step 2: Get samples for visualization
    all_preds, all_targets, all_groups = [], [], []

    # Try to use subset if we have valid indices
    if samples_to_plot_indices and all(isinstance(idx, int) and 0 <= idx < len(loader.dataset) for idx in samples_to_plot_indices):
        try:
            vis_dataset = Subset(loader.dataset, samples_to_plot_indices)
            vis_loader = DataLoader(vis_dataset, batch_size=len(samples_to_plot_indices))

            # Step 3: Run inference on subset
            with torch.no_grad():
                for seismic, velocity, groups in vis_loader:
                    _, _, denorm_vel = batch_processor.process_batch(seismic, velocity, groups, is_train=False)
                    norm_seismic, _, _ = batch_processor.process_batch(seismic, is_train=False)

                    with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
                        preds_norm = model(norm_seismic)

                    denorm_preds = batch_processor.denormalize(preds_norm).cpu()
                    all_preds.append(denorm_preds)
                    all_targets.append(denorm_vel.cpu())
                    all_groups.extend(groups)

            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)

        except Exception as e:
            logging.warning(f"Failed to create subset visualization: {e}. Using first batch from loader.")
            all_preds, all_targets, all_groups = [], [], []

    # Fallback: use first batch from original loader
    if len(all_preds) == 0:
        try:
            with torch.no_grad():
                for seismic, velocity, groups in loader:
                    # Only take first few samples
                    max_vis_samples = min(6, len(groups))
                    seismic = seismic[:max_vis_samples]
                    velocity = velocity[:max_vis_samples]
                    groups = groups[:max_vis_samples]

                    _, _, denorm_vel = batch_processor.process_batch(seismic, velocity, groups, is_train=False)
                    norm_seismic, _, _ = batch_processor.process_batch(seismic, is_train=False)

                    with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
                        preds_norm = model(norm_seismic)

                    denorm_preds = batch_processor.denormalize(preds_norm).cpu()
                    all_preds.append(denorm_preds)
                    all_targets.append(denorm_vel.cpu())
                    all_groups.extend(groups)
                    break  # Only use first batch

            if len(all_preds) > 0:
                all_preds = torch.cat(all_preds)
                all_targets = torch.cat(all_targets)
            else:
                logging.error("No samples available for visualization.")
                return

        except Exception as e:
            logging.error(f"Failed to get samples for visualization: {e}")
            return

    # Step 4: Plot results
    try:
        num_samples = len(all_preds)
        if num_samples == 0:
            logging.warning("No predictions to plot.")
            return

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples), squeeze=False)
        fig.suptitle(f'Prediction Analysis: {cfg.EXPERIMENT_NAME}', fontsize=18, fontweight='bold')

        for i in range(num_samples):
            try:
                pred = all_preds[i].squeeze().numpy()
                target = all_targets[i].squeeze().numpy()
                group = all_groups[i] if i < len(all_groups) else "Unknown"

                # Validate data
                if pred.size == 0 or target.size == 0:
                    logging.warning(f"Empty data for sample {i}, skipping.")
                    continue

                # Calculate dynamic color range
                vmin, vmax = np.percentile(target, [1, 99])
                if vmin == vmax:  # Handle constant images
                    vmin, vmax = target.min(), target.max()
                    if vmin == vmax:
                        vmin, vmax = vmin - 0.1, vmax + 0.1

                # Ground truth
                axes[i, 0].imshow(target, cmap='seismic', vmin=vmin, vmax=vmax)
                axes[i, 0].set_ylabel(f"Group: {group}", fontsize=12, rotation=90, labelpad=20)
                if i == 0:
                    axes[i, 0].set_title("Ground Truth", fontsize=14)

                # Prediction
                axes[i, 1].imshow(pred, cmap='seismic', vmin=vmin, vmax=vmax)
                if i == 0:
                    axes[i, 1].set_title("Prediction", fontsize=14)

                # Difference
                diff = np.abs(target - pred)
                diff_im = axes[i, 2].imshow(diff, cmap='hot')
                if i == 0:
                    axes[i, 2].set_title("Absolute Difference", fontsize=14)

                fig.colorbar(diff_im, ax=axes[i, 2], fraction=0.046, pad=0.04)

                # Remove ticks
                for ax in axes[i]:
                    ax.set_xticks([])
                    ax.set_yticks([])

            except Exception as e:
                logging.warning(f"Failed to plot sample {i}: {e}")
                continue

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = output_dir / f"{cfg.EXPERIMENT_NAME}_predictions.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Validation predictions saved to {output_path}")

    except Exception as e:
        logging.error(f"Failed to create visualization plot: {e}")
        plt.close('all')  # Clean up any open figures

def export_to_onnx(model: nn.Module, sample_input: torch.Tensor, output_path: Path) -> None:
    """Export model to ONNX format"""
    model.eval()
    try:
        torch.onnx.export(
            model,
            sample_input,
            str(output_path),
            opset_version=cfg.ONNX_OPSET_VERSION,
            input_names=['seismic'],
            output_names=['velocity'],
            dynamic_axes={
                'seismic': {0: 'batch_size'},
                'velocity': {0: 'batch_size'}
            }
        )
    except Exception as e:
        logging.error(f"ONNX export failed: {e}")

# Loss Function

class CombinedLoss(nn.Module):
    """Combined loss function with Huber, gradient, and total variation terms,
       returning unreduced (per-pixel) loss for group-based weighting."""

    def __init__(
        self,
        huber_w: float = cfg.HUBER_LOSS_WEIGHT,
        grad_w: float = cfg.GRAD_LOSS_WEIGHT,
        tv_w: float = cfg.TV_LOSS_WEIGHT,
        device: torch.device = torch.device('cpu')
    ) -> None:
        """
        Initialize combined loss.

        Args:
            huber_w: Weight for Huber loss
            grad_w: Weight for gradient loss
            tv_w: Weight for total variation loss
            device: Target device
        """
        super().__init__()
        self.huber_w = huber_w
        self.grad_w = grad_w
        self.tv_w = tv_w
        self.device = device

        # Change reduction to 'none' to return per-pixel loss
        self.huber = nn.HuberLoss(reduction='none')

        # Initialize Sobel filters for gradient loss - register as buffers for device handling
        sobel_x_kernel = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y_kernel = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        # Register as buffers so they move with the module
        self.register_buffer('sobel_x', sobel_x_kernel.repeat(cfg.OUTPUT_CHANNELS, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y_kernel.repeat(cfg.OUTPUT_CHANNELS, 1, 1, 1))

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate gradient matching loss (per-pixel).

        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)

        Returns:
            Gradient loss tensor (B, C, H, W)
        """
        # Cast filters to match the input's dtype (e.g., bfloat16)
        sobel_x = self.sobel_x.to(device=pred.device, dtype=pred.dtype)
        sobel_y = self.sobel_y.to(device=pred.device, dtype=pred.dtype)

        pred_grad_x = F.conv2d(
            pred, sobel_x,
            padding=1,
            groups=cfg.OUTPUT_CHANNELS
        )
        pred_grad_y = F.conv2d(
            pred, sobel_y,
            padding=1,
            groups=cfg.OUTPUT_CHANNELS
        )

        target_grad_x = F.conv2d(
            target, sobel_x,
            padding=1,
            groups=cfg.OUTPUT_CHANNELS
        )
        target_grad_y = F.conv2d(
            target, sobel_y,
            padding=1,
            groups=cfg.OUTPUT_CHANNELS
        )

        # Use reduction='none' for per-pixel gradient loss
        return F.l1_loss(pred_grad_x, target_grad_x, reduction='none') + \
               F.l1_loss(pred_grad_y, target_grad_y, reduction='none')

    def tv_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate total variation loss per sample for noise reduction.
        This returns a (B, 1, 1, 1) tensor to be broadcasted.

        Args:
            pred: Predicted tensor (B, C, H, W)

        Returns:
            TV loss tensor (B, 1, 1, 1)
        """
        # Calculate horizontal and vertical differences
        diff_h = torch.abs(pred[..., :, :-1] - pred[..., :, 1:])
        diff_v = torch.abs(pred[..., :-1, :] - pred[..., 1:, :])

        # Sum differences across spatial dimensions for each channel, then sum channels
        # This gives a TV value for each image in the batch (B, C) -> sum(C) -> (B,)
        tv_per_sample = diff_h.sum(dim=(-1, -2)).sum(dim=-1) + \
                        diff_v.sum(dim=(-1, -2)).sum(dim=-1)

        # Normalize by number of pixels to make it less dependent on image size,
        # then reshape for broadcasting
        num_pixels = pred.shape[-1] * pred.shape[-2]
        # Keepdim=True to maintain original dims if wanted. Here we want (B, 1, 1, 1)
        return (tv_per_sample / num_pixels).view(-1, 1, 1, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss (per-pixel).

        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)

        Returns:
            Combined loss tensor (B, C, H, W)
        """
        # Input validation
        if pred.shape != target.shape:
            raise ValueError(f"Prediction and target shapes must match: {pred.shape} vs {target.shape}")

        if torch.isnan(pred).any() or torch.isnan(target).any():
            raise ValueError("NaN values detected in input tensors")

        # Calculate individual loss components
        huber_loss_unreduced = self.huber(pred, target)  # (B, C, H, W)
        grad_loss_unreduced = self.gradient_loss(pred, target)  # (B, C, H, W)

        # TV loss per sample, then broadcast
        total_variation_loss_per_sample = self.tv_loss(pred)  # (B, 1, 1, 1)
        total_variation_loss_broadcasted = total_variation_loss_per_sample.expand_as(huber_loss_unreduced)

        # Combine with weights
        combined_unreduced_loss = (
            self.huber_w * huber_loss_unreduced +
            self.grad_w * grad_loss_unreduced +
            self.tv_w * total_variation_loss_broadcasted
        )

        return combined_unreduced_loss  # (B, C, H, W)

# Training and Validation Loops

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    scaler: torch.cuda.amp.GradScaler,
    ema: ModelEMA,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batch_processor: GPUBatchProcessor,
    ddp_manager: DDPManager,
    amp_dtype: torch.dtype,
    class_weights: Optional[Dict[str, float]],
    restart_monitor: Optional[WarmRestartMonitor] = None,
    profiler: Optional[torch.profiler.profile] = None
) -> Tuple[float, float, float]:
    """
    Train model for one epoch.

    Args:
        model: Model to train
        loader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        scaler: Gradient scaler for AMP
        ema: EMA model
        scheduler: Learning rate scheduler
        batch_processor: Batch processing utility
        ddp_manager: DDP manager
        amp_dtype: AMP data type
        class_weights: Class weights for loss
        profiler: Optional profiler

    Returns:
        Tuple of (average_loss, average_mae, average_gradient_norm)
    """
    model.train()
    total_loss, total_denorm_mae, total_grad_norm = 0.0, 0.0, 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training",
                leave=False, disable=(not ddp_manager.is_main_process()))

    for i, (seismic, velocity, groups) in pbar:
        # Convert to channels-last format
        seismic = seismic.to(memory_format=torch.channels_last)

        # Process batch (normalization + augmentations)
        norm_seismic, norm_velocity, denorm_velocity = batch_processor.process_batch(
            seismic, velocity, groups, is_train=True
        )

        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=cfg.USE_AMP, dtype=amp_dtype):
            preds_norm = model(norm_seismic)

            # Calculate combined loss (per pixel)
            per_pixel_loss = criterion(preds_norm, norm_velocity)

            # Apply class weights if enabled
            if cfg.USE_LOSS_WEIGHTING and class_weights:
                # Handle missing groups gracefully
                weights = []
                for g in groups:
                    if g in class_weights:
                        weights.append(class_weights[g])
                    else:
                        logging.warning(f"Group '{g}' not found in class_weights, using weight 1.0")
                        weights.append(1.0)

                weights_tensor = torch.tensor(weights, device=ddp_manager.device, dtype=torch.float32)
                weights_tensor = weights_tensor.view(-1, 1, 1, 1)  # Reshape for broadcasting
                loss = (per_pixel_loss * weights_tensor).mean()  # Apply weights and then take mean
            else:
                loss = per_pixel_loss.mean()  # Just take mean if no weighting

        # Backward pass with gradient accumulation
        scaler.scale(loss / cfg.ACCUMULATION_STEPS).backward()

        # Gradient accumulation step
        if (i + 1) % cfg.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)

            # Check for finite gradients
            is_finite = all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )

            if is_finite:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=cfg.GRAD_CLIP_NORM
                )
                total_grad_norm += grad_norm.item()

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
            else:
                logging.warning(f"Skipping optimizer step at iteration {i+1} due to non-finite gradients.")

            # Update scheduler and reset gradients
            scheduler.step()

            # Monitor warm restarts if enabled
            if restart_monitor is not None:
                current_lr = optimizer.param_groups[0]['lr']
                restart_monitor.check_restart(current_lr)

            optimizer.zero_grad(set_to_none=True)

        # Update EMA model
        ema.update(model.module if ddp_manager.is_ddp else model)

        # Update metrics
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            logging.warning(f"Non-finite loss detected at iteration {i+1}: {loss_val}")
            continue

        total_loss += loss_val
        with torch.no_grad():
            denorm_preds = batch_processor.denormalize(preds_norm)
            mae_val = F.l1_loss(denorm_preds, denorm_velocity).item()
            if math.isfinite(mae_val):
                total_denorm_mae += mae_val

        # Update progress bar
        if ddp_manager.is_main_process():
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Profiler step
        if profiler:
            profiler.step()

    # Calculate epoch averages
    num_steps = max(1, len(loader) // cfg.ACCUMULATION_STEPS)  # Prevent division by zero
    return (
        total_loss / len(loader),
        total_denorm_mae / len(loader),
        total_grad_norm / num_steps
    )

def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    metrics: Dict[str, MeanAbsoluteError],
    batch_processor: GPUBatchProcessor,
    ddp_manager: DDPManager,
    amp_dtype: torch.dtype
) -> Dict[str, float]:
    """
    Validate model for one epoch.

    Args:
        model: Model to validate
        loader: Validation DataLoader
        metrics: Dictionary of metric trackers
        batch_processor: Batch processing utility
        ddp_manager: DDP manager
        amp_dtype: AMP data type

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    pbar = tqdm(loader, desc="Validating", leave=False,
                disable=(not ddp_manager.is_main_process()))

    with torch.no_grad():
        for seismic, velocity, groups in pbar:
            # Convert to channels-last format
            seismic = seismic.to(memory_format=torch.channels_last)

            # Process batch
            norm_seismic, norm_velocity, denorm_vel = batch_processor.process_batch(
                seismic, velocity, groups, is_train=False
            )

            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=cfg.USE_AMP, dtype=amp_dtype):
                norm_preds = model(norm_seismic)

            # Denormalize predictions
            denorm_preds = batch_processor.denormalize(norm_preds)

            # Update metrics per group
            for i, group in enumerate(groups):
                pred = denorm_preds[i:i+1]
                target_d = denorm_vel[i:i+1]
                target_n = norm_velocity[i:i+1]

                metrics[f"val_{group}_loss"].update(norm_preds[i:i+1], target_n)
                metrics[f"val_{group}_mae"].update(pred, target_d)
                metrics[f"val_{group}_ssim"].update(
                    rescale_to_unit_range(pred),
                    rescale_to_unit_range(target_d)
                )

    # Compute and reset metrics
    results = {key: metric.compute().item() for key, metric in metrics.items()}
    for metric in metrics.values():
        metric.reset()

    return results

# Main Execution

def main() -> None:
    """Main training pipeline"""
    # Initialize distributed training
    ddp_manager = DDPManager()

    try:
        # Main process setup
        if ddp_manager.is_main_process():
            # Check GPU status
            subprocess.run(["nvidia-smi"])

            # Create output directory
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

            # Setup logging
            log_file = cfg.OUTPUT_DIR / f"{cfg.EXPERIMENT_NAME}_{datetime.datetime.now().strftime('%Y%m%d')}.log"
            setup_logging(log_file)

            logging.info(f"--- Starting Experiment: {cfg.EXPERIMENT_NAME} ---")
            logging.info(f"Mode: {'DDP' if ddp_manager.is_ddp else 'Single-GPU'} | World Size: {ddp_manager.world_size}")

            try:
                cfg.validate()
            except (AssertionError, FileNotFoundError) as e:
                logging.error(f"Config validation failed: {e}")
                return

        # Validate DDP setup
        validate_ddp_setup(ddp_manager)

        # Set random seeds
        set_seed(cfg.RANDOM_SEED)

        # Scan and split dataset files
        file_pairs = scan_files(cfg.TRAIN_PATH)
        train_pairs, val_pairs = create_stratified_split(file_pairs, cfg.VALIDATION_SPLIT)

        # Calculate class weights if needed
        class_weights = calculate_class_weights(train_pairs, ddp_manager)

        # Set seed again with rank offset
        set_seed(cfg.RANDOM_SEED, rank=ddp_manager.rank)

        # Log dataset info
        if ddp_manager.is_main_process():
            log_dataset_info(train_pairs, val_pairs)

        # Create datasets
        train_dataset = FWIDataset(train_pairs, device=ddp_manager.device)
        val_dataset = FWIDataset(val_pairs, device=ddp_manager.device)

        # Create distributed samplers if using DDP
        train_sampler = (DistributedSampler(
            train_dataset,
            num_replicas=ddp_manager.world_size,
            rank=ddp_manager.rank,
            shuffle=True
        ) if ddp_manager.is_ddp else None)

        # Determine optimal number of workers based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        max_workers_by_memory = int(available_memory_gb / cfg.MEMORY_WORKERS_RATIO)
        optimal_workers = min(
            cfg.NUM_WORKERS,
            os.cpu_count() // (ddp_manager.world_size if ddp_manager.world_size > 0 else 1),
            max_workers_by_memory
        )

        if ddp_manager.is_main_process():
            logging.info(f"Using {optimal_workers} workers per DataLoader process.")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=optimal_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4 if optimal_workers > 0 else 2,
            worker_init_fn=seed_worker,
            multiprocessing_context='fork' if sys.platform != 'win32' else None,
            drop_last=True
        )

        val_sampler = (DistributedSampler(
            val_dataset,
            num_replicas=ddp_manager.world_size,
            rank=ddp_manager.rank,
            shuffle=False
        ) if ddp_manager.is_ddp else None)

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.BATCH_SIZE*cfg.VALIDATION_BATCH_MULTIPLIER,
            sampler=val_sampler,
            num_workers=optimal_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4 if optimal_workers > 0 else 2
        )

        if ddp_manager.is_main_process():
            logging.info(f"DataLoaders created. Train batches per GPU: {len(train_loader)}")

        # Initialize model
        model = MultiSourceUNetSwin(ddp_manager=ddp_manager).to(ddp_manager.device)
        model = model.to(memory_format=torch.channels_last)

        # Validate model outputs
        if ddp_manager.is_main_process():
            sample_input = torch.randn(2, cfg.NUM_SOURCES, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH)
            validate_model_outputs(model, sample_input, ddp_manager.device)

        # Compile model for optimization (before DDP wrapping)
        if cfg.USE_TORCH_COMPILE and hasattr(torch, 'compile'):
            if ddp_manager.is_main_process():
                logging.info(f"Compiling model with torch.compile (mode: {cfg.COMPILE_MODE})...")
            try:
                model = torch.compile(model, mode=cfg.COMPILE_MODE)
                if ddp_manager.is_main_process():
                    logging.info("Model compilation successful.")
            except Exception as e:
                if ddp_manager.is_main_process():
                    logging.warning(f"Model compilation failed: {e}. Continuing without compilation.")

        # Enable FlashAttention if available
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)

        # Determine AMP dtype
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        # Check for fused AdamW optimizer
        use_fused_optimizer = torch.cuda.is_available()
        try:
            _ = torch.optim.AdamW([torch.tensor(0)], lr=1e-4, fused=True)
        except (RuntimeError, TypeError):
            use_fused_optimizer = False

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
            fused=use_fused_optimizer
        )

        # Initialize training components
        aug_scheduler = AdaptiveAugmentation()

        criterion = CombinedLoss(
            huber_w=cfg.HUBER_LOSS_WEIGHT,
            grad_w=cfg.GRAD_LOSS_WEIGHT,
            tv_w=cfg.TV_LOSS_WEIGHT,
            device=ddp_manager.device
        )

        ema = ModelEMA(model, decay=cfg.EMA_DECAY).to(ddp_manager.device)

        # Compile EMA model for validation optimization
        if cfg.USE_TORCH_COMPILE and hasattr(torch, 'compile'):
            try:
                ema.ema_model = torch.compile(ema.ema_model, mode=cfg.COMPILE_MODE)
                if ddp_manager.is_main_process():
                    logging.info("EMA model compilation successful.")
            except Exception as e:
                if ddp_manager.is_main_process():
                    logging.warning(f"EMA model compilation failed: {e}. Continuing without compilation.")

        scheduler = get_scheduler(optimizer, train_loader)

        # Validate scheduler configuration
        if ddp_manager.is_main_process():
            validate_scheduler_config(train_loader)

        # Initialize restart monitor for warm restarts
        restart_monitor = WarmRestartMonitor() if cfg.USE_WARM_RESTARTS else None

        # Resume from checkpoint if specified
        start_epoch = 0
        history = defaultdict(list) # Initialize history
        if cfg.RESUME_CHECKPOINT:
            if ddp_manager.is_main_process():
                logging.info(f"Resuming from checkpoint: {cfg.RESUME_CHECKPOINT}")

            # Capture the returned history from load_checkpoint
            returned_start_epoch, loaded_history = load_checkpoint(
                Path(cfg.RESUME_CHECKPOINT),
                model,
                optimizer,
                scheduler,
                ema,
                ddp_manager
            )
            start_epoch = returned_start_epoch
            history.update(loaded_history)

        # Wrap model in DDP if using distributed training
        if ddp_manager.is_ddp:
            safe_barrier(ddp_manager)
            model = DDP(
                model,
                device_ids=[ddp_manager.rank % torch.cuda.device_count()],
                find_unused_parameters=False
            )

        # Model summary and metrics setup
        metrics_logger = None
        if ddp_manager.is_main_process():
            logging.info("--- Model & Training Setup ---")
            summary(
                model,
                input_size=(cfg.BATCH_SIZE, cfg.NUM_SOURCES, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH),
                device=ddp_manager.device,
                depth=5
            )
            metrics_logger = MetricsLogger(log_dir=cfg.OUTPUT_DIR / 'tensorboard')

        # Training components (all processes need these)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP)
        early_stopper = EarlyStopping(verbose=ddp_manager.is_main_process())
        batch_processor = GPUBatchProcessor(
            device=ddp_manager.device,
            aug_scheduler=aug_scheduler
        )

        # Initialize validation metrics
        validation_metrics = {}
        for group in ["Vel", "Style", "Fault"]:
            for metric_name, metric_class in [
                ("loss", MeanAbsoluteError),
                ("mae", MeanAbsoluteError),
                ("ssim", StructuralSimilarityIndexMeasure)
            ]:
                key = f"val_{group}_{metric_name}"
                params = {'data_range': 1.0} if metric_name == "ssim" else {}
                validation_metrics[key] = metric_class(**params).to(ddp_manager.device)

        start_time = time.time()

        # Profiler setup
        profiler_context = (
            torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(cfg.OUTPUT_DIR / 'profiler')),
                record_shapes=True,
                with_stack=True
            ) if cfg.RUN_PROFILER and ddp_manager.is_main_process() else contextlib.nullcontext()
        )

        if ddp_manager.is_main_process():
            logging.info(f"Starting training from epoch {start_epoch+1}...")

        # Training loop
        with profiler_context as profiler:
            for epoch in range(start_epoch, cfg.NUM_EPOCHS):
                epoch_start_time = time.time()

                # Set epoch for distributed sampler
                if ddp_manager.is_ddp:
                    train_sampler.set_epoch(epoch)

                # Train for one epoch
                train_loss, train_mae, grad_norm = train_one_epoch(
                    model, train_loader, optimizer, criterion, scaler,
                    ema, scheduler, batch_processor, ddp_manager,
                    amp_dtype, class_weights, restart_monitor, profiler
                )

                # Validate
                val_results = validate_one_epoch(
                    ema.ema_model, val_loader, validation_metrics,
                    batch_processor, ddp_manager, amp_dtype
                )

                # Main process logging and checkpointing
                if ddp_manager.is_main_process():
                    # Update augmentation strength
                    avg_val_loss = val_results.get(f"val_Vel_loss", float('nan'))
                    if not math.isnan(avg_val_loss):
                        aug_scheduler.update(avg_val_loss)

                    # Get current gradient scale
                    current_scale = scaler.get_scale()

                    # Prepare epoch metrics
                    epoch_metrics = {
                        'train/loss': train_loss,
                        'train/mae': train_mae,
                        'train/grad_norm': grad_norm,
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'train/grad_scaler': current_scale
                    }

                    # Update history
                    history['train_loss'].append(train_loss)
                    history['train_denorm_mae'].append(train_mae)
                    history['grad_norm'].append(grad_norm)
                    history['lr'].append(optimizer.param_groups[0]['lr'])

                    # Group-specific metrics
                    val_losses, val_maes = [], []
                    logging.info(f"--- Epoch {epoch+1}/{cfg.NUM_EPOCHS} | Time: {time.time()-epoch_start_time:.2f}s ---")
                    logging.info(f"  Train -> Loss: {train_loss:.4f} | MAE: {train_mae:.2f} | Grad Norm: {grad_norm:.2f}")

                    for group in ["Vel", "Style", "Fault"]:
                        loss = val_results.get(f"val_{group}_loss", float('nan'))
                        mae = val_results.get(f"val_{group}_mae", float('nan'))
                        ssim = val_results.get(f"val_{group}_ssim", float('nan'))

                        history[f"val_{group}_mae"].append(mae)
                        history[f"val_{group}_ssim"].append(ssim)

                        if not math.isnan(mae):
                            val_maes.append(mae)

                        logging.info(f"  - Val {group:<5} -> Loss: {loss:.4f} | MAE: {mae:.2f} | SSIM: {ssim:.4f}")

                    # Calculate average validation MAE
                    avg_val_mae = sum(val_maes)/len(val_maes) if val_maes else float('nan')
                    history['val_avg_mae'].append(avg_val_mae)
                    epoch_metrics['val/avg_mae'] = avg_val_mae

                    logging.info(f"  Overall Valid -> Avg MAE: {avg_val_mae:.2f}")

                    # Log metrics
                    if metrics_logger is not None:
                        metrics_logger.log(epoch_metrics, epoch)

                    # Early stopping check
                    if not math.isnan(avg_val_mae):
                        early_stopper(
                            avg_val_mae,
                            ema.ema_model,
                            cfg.OUTPUT_DIR / "best_model_ema.pth"
                        )

                    # Periodic checkpointing
                    if (epoch + 1) % cfg.CHECKPOINT_EVERY == 0:
                        model_to_save = model.module if ddp_manager.is_ddp else model
                        save_checkpoint(
                            epoch,
                            model_to_save,
                            optimizer,
                            scheduler,
                            ema,
                            cfg.OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}.pth",
                            history,
                            ddp_manager
                        )

                    # Break if early stopping triggered
                    if early_stopper.early_stop:
                        logging.info("Early stopping triggered.")
                        break

                # Broadcast stop signal to all processes
                if ddp_manager.is_ddp:
                    stop_tensor = torch.tensor(int(early_stopper.early_stop), device=ddp_manager.device)
                    dist.broadcast(stop_tensor, src=0)
                    if stop_tensor.item() == 1:
                        break

        # Final cleanup and logging
        safe_barrier(ddp_manager)

        if ddp_manager.is_main_process():
            logging.info(f"--- Training Complete | Total Time: {(time.time()-start_time)/3600:.2f} hours ---")
            best_model_path = cfg.OUTPUT_DIR / "best_model_ema.pth"

            if best_model_path.exists():
                logging.info(f"Best validation MAE achieved: {early_stopper.best_score:.4f}")

                # Load best model for final evaluation
                final_model = MultiSourceUNetSwin(ddp_manager=ddp_manager)
                final_model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
                final_model.to(ddp_manager.device)
                final_model.to(memory_format=torch.channels_last)

                # Generate final plots
                plot_metrics(history, cfg.OUTPUT_DIR)
                visualize_predictions(final_model, val_loader, batch_processor, cfg.OUTPUT_DIR)

                # Export to ONNX if enabled
                if cfg.EXPORT_ONNX:
                    logging.info("--- Exporting model to ONNX ---")
                    sample_input = torch.randn(1, cfg.NUM_SOURCES, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, device=ddp_manager.device)
                    sample_input = sample_input.to(memory_format=torch.channels_last)
                    export_path = cfg.OUTPUT_DIR / f"{cfg.EXPERIMENT_NAME}.onnx"
                    export_to_onnx(final_model, sample_input, export_path)
            else:
                logging.warning("No best model saved. Skipping final visualizations.")

            # Close metrics logger
            if metrics_logger is not None:
                metrics_logger.close()

            # Log restart summary
            if restart_monitor is not None and ddp_manager.is_main_process():
                logging.info(f"🔄 Warm restart summary: {restart_monitor.restart_count} restarts occurred during training")

    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            logging.warning("Training interrupted by user.")
        else:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # Cleanup distributed training
        ddp_manager.cleanup()
        if ddp_manager.is_main_process():
            logging.info("Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()