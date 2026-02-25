"""
GPU Utilities for RDT Trading System ML Models

Provides GPU detection, configuration, and optimization utilities for
TensorFlow/Keras-based models, including support for NVIDIA CUDA and Apple MPS.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import sys
from loguru import logger

# GPU availability flags
CUDA_AVAILABLE = False
MPS_AVAILABLE = False
TF_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True

    # Check for CUDA GPUs
    cuda_gpus = tf.config.list_physical_devices('GPU')
    CUDA_AVAILABLE = len(cuda_gpus) > 0 and any(
        'GPU' in gpu.device_type for gpu in cuda_gpus
    )

    # Check for Apple MPS (Metal Performance Shaders)
    try:
        # MPS is available on Apple Silicon Macs with TensorFlow-metal
        if sys.platform == 'darwin':
            # TensorFlow 2.x on macOS with metal plugin
            import platform
            if platform.processor() == 'arm':
                # Check if MPS devices are available
                mps_devices = [d for d in tf.config.list_physical_devices() if 'GPU' in d.device_type]
                MPS_AVAILABLE = len(mps_devices) > 0
    except Exception:
        MPS_AVAILABLE = False

except ImportError:
    logger.warning("TensorFlow not available. GPU utilities will be limited.")


@dataclass
class GPUInfo:
    """Information about an available GPU device."""
    device_id: int
    name: str
    memory_total_mb: float
    memory_free_mb: float
    compute_capability: Optional[str] = None
    is_cuda: bool = True
    is_mps: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'device_id': self.device_id,
            'name': self.name,
            'memory_total_mb': self.memory_total_mb,
            'memory_free_mb': self.memory_free_mb,
            'compute_capability': self.compute_capability,
            'is_cuda': self.is_cuda,
            'is_mps': self.is_mps
        }


def is_gpu_available() -> bool:
    """
    Check if any GPU (CUDA or MPS) is available for training.

    Returns:
        bool: True if GPU is available, False otherwise
    """
    return CUDA_AVAILABLE or MPS_AVAILABLE


def is_cuda_available() -> bool:
    """
    Check if NVIDIA CUDA GPU is available.

    Returns:
        bool: True if CUDA GPU is available
    """
    return CUDA_AVAILABLE


def is_mps_available() -> bool:
    """
    Check if Apple MPS (Metal Performance Shaders) is available.

    Returns:
        bool: True if MPS is available (Apple Silicon)
    """
    return MPS_AVAILABLE


def get_gpu_count() -> int:
    """
    Get the number of available GPUs.

    Returns:
        int: Number of available GPUs
    """
    if not TF_AVAILABLE:
        return 0

    gpus = tf.config.list_physical_devices('GPU')
    return len(gpus)


def get_gpu_info() -> List[GPUInfo]:
    """
    Get detailed information about available GPUs.

    Returns:
        List[GPUInfo]: List of GPU information objects
    """
    if not TF_AVAILABLE:
        return []

    gpu_info_list = []
    gpus = tf.config.list_physical_devices('GPU')

    for idx, gpu in enumerate(gpus):
        try:
            # Try to get GPU details
            name = gpu.name
            memory_total = 0.0
            memory_free = 0.0
            compute_capability = None
            is_cuda = True
            is_mps = False

            # Check if this is MPS (Apple Silicon)
            if sys.platform == 'darwin' and 'GPU' in gpu.device_type:
                is_mps = True
                is_cuda = False
                name = "Apple Silicon GPU (MPS)"
                # MPS doesn't report memory in the same way
                try:
                    import subprocess
                    result = subprocess.run(
                        ['sysctl', '-n', 'hw.memsize'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        # Apple Silicon uses unified memory, report system memory
                        total_bytes = int(result.stdout.strip())
                        memory_total = total_bytes / (1024 * 1024)  # Convert to MB
                        memory_free = memory_total * 0.5  # Estimate 50% available
                except Exception:
                    pass
            else:
                # CUDA GPU - try to get memory info
                try:
                    # Use nvidia-smi if available
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,compute_cap',
                         '--format=csv,noheader,nounits', '-i', str(idx)],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        parts = result.stdout.strip().split(',')
                        if len(parts) >= 4:
                            name = parts[0].strip()
                            memory_total = float(parts[1].strip())
                            memory_free = float(parts[2].strip())
                            compute_capability = parts[3].strip()
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    # nvidia-smi not available, use TF's reported name
                    name = str(gpu)
                except Exception as e:
                    logger.debug(f"Error getting GPU info via nvidia-smi: {e}")

            gpu_info = GPUInfo(
                device_id=idx,
                name=name,
                memory_total_mb=memory_total,
                memory_free_mb=memory_free,
                compute_capability=compute_capability,
                is_cuda=is_cuda,
                is_mps=is_mps
            )
            gpu_info_list.append(gpu_info)

        except Exception as e:
            logger.warning(f"Error getting info for GPU {idx}: {e}")

    return gpu_info_list


def get_gpu_memory_info(device_id: int = 0) -> Tuple[float, float]:
    """
    Get memory information for a specific GPU.

    Args:
        device_id: GPU device ID (default: 0)

    Returns:
        Tuple of (total_memory_mb, free_memory_mb)
    """
    gpu_infos = get_gpu_info()
    if device_id < len(gpu_infos):
        info = gpu_infos[device_id]
        return info.memory_total_mb, info.memory_free_mb
    return 0.0, 0.0


def configure_gpu(
    memory_limit: Optional[float] = None,
    memory_growth: bool = True,
    device_id: Optional[int] = None,
    visible_devices: Optional[List[int]] = None,
    allow_soft_placement: bool = True,
    log_device_placement: bool = False
) -> bool:
    """
    Configure GPU settings for TensorFlow/Keras.

    This should be called BEFORE any TensorFlow operations or model creation.

    Args:
        memory_limit: Memory limit in MB (None for no limit, or fraction 0-1 for percentage)
        memory_growth: Enable memory growth to prevent OOM (recommended)
        device_id: Specific GPU device ID to use (None for all)
        visible_devices: List of GPU device IDs to make visible (None for all)
        allow_soft_placement: Allow operations on CPU if GPU op not available
        log_device_placement: Log device placement for debugging

    Returns:
        bool: True if GPU was configured successfully, False otherwise
    """
    if not TF_AVAILABLE:
        logger.warning("TensorFlow not available, cannot configure GPU")
        return False

    try:
        gpus = tf.config.list_physical_devices('GPU')

        if not gpus:
            logger.info("No GPUs found. Running on CPU.")
            return False

        logger.info(f"Found {len(gpus)} GPU(s)")

        # Set visible devices if specified
        if visible_devices is not None:
            visible_gpus = [gpus[i] for i in visible_devices if i < len(gpus)]
            tf.config.set_visible_devices(visible_gpus, 'GPU')
            gpus = visible_gpus
            logger.info(f"Set visible GPUs: {visible_devices}")
        elif device_id is not None and device_id < len(gpus):
            tf.config.set_visible_devices([gpus[device_id]], 'GPU')
            gpus = [gpus[device_id]]
            logger.info(f"Using GPU device {device_id}")

        # Configure each GPU
        for gpu in gpus:
            if memory_growth:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Enabled memory growth for {gpu}")
                except RuntimeError as e:
                    logger.warning(f"Could not enable memory growth: {e}")

            if memory_limit is not None:
                try:
                    # If memory_limit is a fraction (0-1), interpret as percentage
                    if 0 < memory_limit < 1:
                        total_memory, _ = get_gpu_memory_info()
                        if total_memory > 0:
                            memory_limit_mb = int(total_memory * memory_limit)
                        else:
                            # Default to 4GB if we can't detect memory
                            memory_limit_mb = int(4096 * memory_limit)
                    else:
                        memory_limit_mb = int(memory_limit)

                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                    )
                    logger.info(f"Set memory limit to {memory_limit_mb}MB for {gpu}")
                except RuntimeError as e:
                    logger.warning(f"Could not set memory limit: {e}")

        # Configure soft placement and device logging
        tf.config.set_soft_device_placement(allow_soft_placement)
        if log_device_placement:
            tf.debugging.set_log_device_placement(True)

        # Verify configuration
        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info(f"Configured {len(logical_gpus)} logical GPU(s)")

        return True

    except Exception as e:
        logger.error(f"Failed to configure GPU: {e}")
        return False


def configure_multi_gpu(
    strategy_type: str = 'mirrored',
    devices: Optional[List[str]] = None,
    memory_growth: bool = True
) -> Optional[Any]:
    """
    Configure multi-GPU training strategy.

    Args:
        strategy_type: Type of distribution strategy:
            - 'mirrored': MirroredStrategy for synchronous training on multiple GPUs
            - 'multi_worker': MultiWorkerMirroredStrategy for distributed training
            - 'parameter_server': ParameterServerStrategy
        devices: List of device strings (e.g., ['/gpu:0', '/gpu:1'])
        memory_growth: Enable memory growth for all GPUs

    Returns:
        TensorFlow distribution strategy or None if not available
    """
    if not TF_AVAILABLE:
        logger.warning("TensorFlow not available")
        return None

    try:
        gpus = tf.config.list_physical_devices('GPU')

        if len(gpus) < 2:
            logger.info("Single GPU detected, multi-GPU strategy not needed")
            return None

        # Enable memory growth first
        if memory_growth:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass

        if strategy_type == 'mirrored':
            if devices:
                strategy = tf.distribute.MirroredStrategy(devices=devices)
            else:
                strategy = tf.distribute.MirroredStrategy()
            logger.info(f"Created MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
            return strategy

        elif strategy_type == 'multi_worker':
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            logger.info("Created MultiWorkerMirroredStrategy")
            return strategy

        elif strategy_type == 'parameter_server':
            # Requires cluster configuration
            logger.warning("ParameterServerStrategy requires cluster configuration")
            return None

        else:
            logger.warning(f"Unknown strategy type: {strategy_type}")
            return None

    except Exception as e:
        logger.error(f"Failed to configure multi-GPU strategy: {e}")
        return None


def get_optimal_batch_size(
    model_size_mb: float,
    sequence_length: int = 20,
    n_features: int = 10,
    target_memory_usage: float = 0.7,
    min_batch_size: int = 8,
    max_batch_size: int = 512
) -> int:
    """
    Suggest an optimal batch size based on GPU memory and model size.

    This is a heuristic estimate. Actual optimal batch size may vary based on
    specific model architecture and operations.

    Args:
        model_size_mb: Approximate model size in MB
        sequence_length: Length of input sequences (for LSTM/Transformer)
        n_features: Number of features per time step
        target_memory_usage: Target fraction of GPU memory to use (default: 0.7)
        min_batch_size: Minimum batch size to return
        max_batch_size: Maximum batch size to return

    Returns:
        int: Suggested batch size
    """
    if not is_gpu_available():
        # CPU fallback - use smaller batch size
        return min(32, max_batch_size)

    try:
        _, free_memory_mb = get_gpu_memory_info()

        if free_memory_mb == 0:
            # Couldn't detect memory, use conservative estimate
            free_memory_mb = 4096  # Assume 4GB

        # Available memory for training
        available_memory = free_memory_mb * target_memory_usage

        # Reserve memory for model parameters (rough estimate)
        memory_for_batches = available_memory - model_size_mb * 2  # Model + gradients

        if memory_for_batches < 100:
            logger.warning("Very limited GPU memory available")
            return min_batch_size

        # Estimate memory per sample (bytes)
        # LSTM input: sequence_length * n_features * 4 bytes (float32)
        # Intermediate activations: ~4x input size (rough estimate)
        # Gradients: ~2x model size per sample (rough estimate)
        bytes_per_sample = (
            sequence_length * n_features * 4 * 5 +  # Input + activations
            model_size_mb * 1024 * 1024 / 100  # Gradient contribution per sample
        )
        mb_per_sample = bytes_per_sample / (1024 * 1024)

        # Calculate batch size
        suggested_batch_size = int(memory_for_batches / mb_per_sample)

        # Round to power of 2 for efficiency
        import math
        if suggested_batch_size > 0:
            power = math.floor(math.log2(suggested_batch_size))
            suggested_batch_size = 2 ** power

        # Clamp to min/max
        suggested_batch_size = max(min_batch_size, min(suggested_batch_size, max_batch_size))

        logger.info(
            f"Suggested batch size: {suggested_batch_size} "
            f"(free memory: {free_memory_mb:.0f}MB, model: {model_size_mb:.0f}MB)"
        )

        return suggested_batch_size

    except Exception as e:
        logger.warning(f"Error calculating batch size: {e}")
        return 32  # Safe default


def get_device_strategy() -> str:
    """
    Get the recommended device strategy based on available hardware.

    Returns:
        str: Device strategy ('gpu', 'mps', 'multi_gpu', 'cpu')
    """
    if not is_gpu_available():
        return 'cpu'

    gpu_count = get_gpu_count()

    if is_mps_available():
        return 'mps'
    elif gpu_count > 1:
        return 'multi_gpu'
    else:
        return 'gpu'


def with_device(device: Optional[str] = None):
    """
    Context manager for running operations on a specific device.

    Args:
        device: Device string (e.g., '/gpu:0', '/cpu:0', or None for default)

    Returns:
        TensorFlow device context

    Example:
        with with_device('/gpu:0'):
            model.fit(X, y)
    """
    if not TF_AVAILABLE:
        # Return a no-op context manager
        from contextlib import nullcontext
        return nullcontext()

    if device is None:
        if is_gpu_available():
            device = '/GPU:0'
        else:
            device = '/CPU:0'

    return tf.device(device)


def get_mixed_precision_policy() -> Optional[str]:
    """
    Get the recommended mixed precision policy for the current hardware.

    Mixed precision (float16) can significantly speed up training on
    modern GPUs with Tensor Cores.

    Returns:
        str: Policy name ('mixed_float16', 'float32') or None
    """
    if not TF_AVAILABLE:
        return None

    if not is_gpu_available():
        return 'float32'

    gpu_infos = get_gpu_info()

    for info in gpu_infos:
        if info.compute_capability:
            try:
                # Tensor Cores available on compute capability 7.0+
                major, minor = info.compute_capability.split('.')
                if int(major) >= 7:
                    return 'mixed_float16'
            except (ValueError, AttributeError):
                pass

    return 'float32'


def enable_mixed_precision(policy: str = 'mixed_float16') -> bool:
    """
    Enable mixed precision training for better performance.

    Args:
        policy: Precision policy ('mixed_float16' or 'mixed_bfloat16')

    Returns:
        bool: True if enabled successfully
    """
    if not TF_AVAILABLE:
        return False

    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy(policy)
        logger.info(f"Enabled mixed precision policy: {policy}")
        return True
    except Exception as e:
        logger.warning(f"Could not enable mixed precision: {e}")
        return False


def get_gpu_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of GPU availability and configuration.

    Returns:
        Dict with GPU summary information
    """
    summary = {
        'tensorflow_available': TF_AVAILABLE,
        'gpu_available': is_gpu_available(),
        'cuda_available': is_cuda_available(),
        'mps_available': is_mps_available(),
        'gpu_count': get_gpu_count(),
        'device_strategy': get_device_strategy(),
        'recommended_precision': get_mixed_precision_policy(),
        'gpus': []
    }

    if is_gpu_available():
        gpu_infos = get_gpu_info()
        summary['gpus'] = [gpu.to_dict() for gpu in gpu_infos]

    return summary


def setup_gpu_for_training(
    use_gpu: str = 'auto',
    memory_limit: Optional[float] = None,
    device_id: Optional[int] = None,
    enable_mixed_prec: bool = False
) -> Dict[str, Any]:
    """
    One-stop setup for GPU-accelerated training.

    This function handles all GPU configuration in one call.

    Args:
        use_gpu: 'auto' (detect), 'true'/'yes' (force GPU), 'false'/'no' (force CPU)
        memory_limit: Memory limit in MB or fraction (0-1)
        device_id: Specific GPU device ID to use
        enable_mixed_prec: Enable mixed precision training

    Returns:
        Dict with configuration results
    """
    result = {
        'using_gpu': False,
        'device': 'cpu',
        'gpu_configured': False,
        'mixed_precision': False,
        'error': None
    }

    # Determine if we should use GPU
    use_gpu_lower = str(use_gpu).lower()

    if use_gpu_lower in ('false', 'no', '0', 'cpu'):
        logger.info("GPU disabled by configuration, using CPU")
        if TF_AVAILABLE:
            tf.config.set_visible_devices([], 'GPU')
        return result

    should_use_gpu = (
        use_gpu_lower in ('true', 'yes', '1', 'gpu') or
        (use_gpu_lower == 'auto' and is_gpu_available())
    )

    if not should_use_gpu:
        logger.info("No GPU available or GPU disabled, using CPU")
        return result

    # Configure GPU
    try:
        gpu_configured = configure_gpu(
            memory_limit=memory_limit,
            memory_growth=True,
            device_id=device_id
        )

        if gpu_configured:
            result['using_gpu'] = True
            result['gpu_configured'] = True

            if is_mps_available():
                result['device'] = 'mps'
            elif device_id is not None:
                result['device'] = f'/GPU:{device_id}'
            else:
                result['device'] = '/GPU:0'

            # Enable mixed precision if requested and supported
            if enable_mixed_prec:
                policy = get_mixed_precision_policy()
                if policy == 'mixed_float16':
                    if enable_mixed_precision(policy):
                        result['mixed_precision'] = True

            logger.info(f"GPU training configured: {result}")
        else:
            logger.warning("GPU configuration failed, falling back to CPU")

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error setting up GPU: {e}")

    return result


# Prometheus metrics for GPU monitoring - import from monitoring.metrics to avoid duplication
try:
    from monitoring.metrics import (
        rdt_gpu_utilization_percent as rdt_gpu_utilization,
        rdt_gpu_memory_used_mb as rdt_gpu_memory_used,
        rdt_gpu_memory_total_mb as rdt_gpu_memory_total,
        rdt_gpu_temperature_celsius as rdt_gpu_temperature,
    )
    GPU_METRICS_AVAILABLE = True

except ImportError:
    GPU_METRICS_AVAILABLE = False
    logger.debug("Prometheus metrics not available for GPU monitoring")


def update_gpu_metrics() -> None:
    """
    Update Prometheus metrics for GPU utilization.

    Should be called periodically (e.g., every 30 seconds) during training.
    """
    if not GPU_METRICS_AVAILABLE:
        return

    if not is_gpu_available():
        return

    try:
        import subprocess

        # Query nvidia-smi for GPU metrics
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    device_id = parts[0]
                    device_label = f'gpu_{device_id}'

                    try:
                        rdt_gpu_utilization.labels(device=device_label).set(float(parts[1]))
                        rdt_gpu_memory_used.labels(device=device_label).set(float(parts[2]))
                        rdt_gpu_memory_total.labels(device=device_label).set(float(parts[3]))
                        rdt_gpu_temperature.labels(device=device_label).set(float(parts[4]))
                    except (ValueError, IndexError):
                        pass

    except (FileNotFoundError, subprocess.TimeoutExpired):
        # nvidia-smi not available (MPS or no NVIDIA GPU)
        pass
    except Exception as e:
        logger.debug(f"Error updating GPU metrics: {e}")


# Lazy GPU status logging - only runs when first requested
_gpu_status_logged = False

def log_gpu_status():
    """Log GPU status on first call. Safe to call multiple times."""
    global _gpu_status_logged
    if _gpu_status_logged:
        return
    _gpu_status_logged = True
    if TF_AVAILABLE:
        summary = get_gpu_summary()
        if summary['gpu_available']:
            logger.info(f"GPU available: {summary['gpu_count']} device(s), strategy: {summary['device_strategy']}")
        else:
            logger.info("No GPU detected, using CPU")
