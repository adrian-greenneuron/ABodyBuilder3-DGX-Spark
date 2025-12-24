"""
DGX Spark Utilities for ABodyBuilder3

Utilities for detecting and optimizing for DGX Spark's unified memory
architecture (GB10 Superchip with Grace CPU + Blackwell GPU).

Key optimizations:
- Auto-detection of unified memory architecture
- Optimal batch size calculation for 128GB unified memory
- Zero-copy tensor creation
- Configuration helpers
"""

import os
import subprocess
from typing import Optional, Dict, Any
import numpy as np
import torch


def is_unified_memory_architecture() -> bool:
    """
    Detect if running on a unified memory architecture (e.g., DGX Spark).
    
    Checks for Grace-Blackwell GB10 Superchip indicators.
    
    Returns:
        True if unified memory architecture is detected
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Check GPU name for Blackwell indicators
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        # GB10 Superchip uses Blackwell architecture
        blackwell_indicators = ["gb10", "blackwell", "b200", "gb200"]
        if any(ind in gpu_name for ind in blackwell_indicators):
            return True
        
        # Check for NVLink-C2C (cache-coherent interconnect)
        # This is the hallmark of unified memory
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_info = result.stdout.lower()
            if any(ind in gpu_info for ind in blackwell_indicators):
                return True
        
        # Check for ARM64 + NVIDIA combo (Grace CPU indicator)
        import platform
        if platform.machine() == "aarch64":
            # ARM64 with NVIDIA GPU strongly suggests Grace-Blackwell
            return True
            
    except Exception:
        pass
    
    return False


def is_dgx_spark() -> bool:
    """
    Detect if running on DGX Spark specifically.
    
    Returns:
        True if DGX Spark is detected
    """
    return is_unified_memory_architecture()


def get_total_memory_gb() -> float:
    """
    Get total available memory in GB.
    
    On unified memory systems, this returns the total system memory.
    On discrete GPU systems, this returns GPU memory.
    
    Returns:
        Total memory in GB
    """
    if not torch.cuda.is_available():
        return 0.0
    
    if is_unified_memory_architecture():
        # Return system memory for unified architectures
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        # Parse "MemTotal: 123456 kB"
                        parts = line.split()
                        kb = int(parts[1])
                        return kb / 1024 / 1024  # KB to GB
        except Exception:
            pass
        # Fallback to GPU memory query
    
    # Standard GPU memory
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def get_memory_usage_gb() -> float:
    """
    Get current memory usage in GB.
    
    On unified memory systems (DGX Spark), returns system memory usage.
    On discrete GPU systems, returns GPU memory usage.
    
    Returns:
        Current memory usage in GB
    """
    if is_unified_memory_architecture():
        # Use system memory for unified architecture
        try:
            with open('/proc/meminfo') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        meminfo[key] = int(parts[1]) / 1024 / 1024  # KB to GB
                total = meminfo.get('MemTotal', 0)
                available = meminfo.get('MemAvailable', 0)
                return total - available
        except Exception:
            pass
    
    # Fallback to PyTorch CUDA memory
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def get_available_memory_gb() -> float:
    """
    Get available memory in GB.
    
    On unified memory systems (DGX Spark), returns available system memory.
    On discrete GPU systems, returns available GPU memory.
    
    Returns:
        Available memory in GB
    """
    if is_unified_memory_architecture():
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        parts = line.split()
                        return int(parts[1]) / 1024 / 1024  # KB to GB
        except Exception:
            pass
    
    # Fallback to PyTorch CUDA memory
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()
        return (total - allocated) / (1024**3)
    return 0.0


class MemoryTracker:
    """
    Context manager for tracking memory usage.
    
    Works correctly on both unified memory (DGX Spark) and discrete GPU systems.
    
    Usage:
        >>> with MemoryTracker() as tracker:
        ...     # do work
        >>> print(f"Used: {tracker.used_gb:.1f} GB, Peak: {tracker.peak_gb:.1f} GB")
    """
    
    def __init__(self):
        self.start_gb = 0.0
        self.end_gb = 0.0
        self.peak_gb = 0.0
        self._is_unified = is_unified_memory_architecture()
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start_gb = get_memory_usage_gb()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_gb = get_memory_usage_gb()
        
        if self._is_unified:
            # For unified memory, peak is approximated by end usage
            self.peak_gb = max(self.end_gb, self.start_gb)
        else:
            # For discrete GPU, use PyTorch's peak tracking
            self.peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    
    @property
    def used_gb(self) -> float:
        """Memory used during the tracked block."""
        return self.end_gb - self.start_gb


def get_optimal_batch_size(
    model_memory_mb: float = 2500,
    avg_sequence_length: int = 250,
    memory_fraction: float = 0.7,
) -> int:
    """
    Calculate optimal batch size for the current system.
    
    Args:
        model_memory_mb: Approximate model memory usage in MB
        avg_sequence_length: Average antibody sequence length
        memory_fraction: Fraction of memory to use (0.0-1.0)
        
    Returns:
        Recommended batch size
        
    Note:
        Memory scales as O(batch * seq_len^2) for pair representations.
        This is a conservative estimate.
    """
    total_memory_gb = get_total_memory_gb()
    available_mb = (total_memory_gb * 1024 - model_memory_mb) * memory_fraction
    
    # Estimate memory per antibody (empirical formula)
    # pair representation: seq_len^2 * feature_dim * 4 bytes (float32)
    # single representation: seq_len * feature_dim * 4 bytes
    # positions: seq_len * 14 * 3 * 4 bytes * num_blocks
    feature_dim = 132  # pair feature dimension
    single_dim = 23  # single feature dimension
    num_blocks = 8
    
    per_antibody_mb = (
        avg_sequence_length**2 * feature_dim * 4 / (1024**2) +  # pair
        avg_sequence_length * single_dim * 4 / (1024**2) +  # single
        avg_sequence_length * 14 * 3 * 4 * num_blocks / (1024**2) +  # positions
        avg_sequence_length * 37 * 3 * 4 / (1024**2)  # atom37
    )
    
    # Add 50% overhead for intermediate tensors
    per_antibody_mb *= 1.5
    
    batch_size = max(1, int(available_mb / per_antibody_mb))
    
    # Cap at reasonable maximum
    if is_unified_memory_architecture():
        # DGX Spark can handle larger batches
        batch_size = min(batch_size, 128)
    else:
        # Standard GPUs more conservative
        batch_size = min(batch_size, 32)
    
    return batch_size


def configure_for_unified_memory() -> Dict[str, Any]:
    """
    Return optimal PyTorch/DataLoader configuration for unified memory.
    
    Returns:
        Dictionary of configuration parameters
        
    Usage:
        >>> config = configure_for_unified_memory()
        >>> dataloader = DataLoader(dataset, **config)
    """
    if is_unified_memory_architecture():
        return {
            "pin_memory": False,  # No benefit on unified memory
            "num_workers": 0,  # Single process often faster
            "prefetch_factor": None,  # Disable prefetching
            "persistent_workers": False,
        }
    else:
        # Standard GPU configuration
        return {
            "pin_memory": True,
            "num_workers": 4,
            "prefetch_factor": 2,
            "persistent_workers": True,
        }


def zero_copy_tensor(data: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Create a CUDA tensor from numpy array with minimal copy overhead.
    
    On unified memory architectures, this avoids unnecessary data movement.
    
    Args:
        data: Input numpy array
        device: Target device
        
    Returns:
        PyTorch tensor on the specified device
    """
    # torch.as_tensor shares memory when possible
    tensor = torch.as_tensor(data)
    
    if device == "cuda" and torch.cuda.is_available():
        if is_unified_memory_architecture():
            # On unified memory, the GPU can access CPU memory directly
            # so we can be less aggressive about moving data
            return tensor.to(device, non_blocking=True)
        else:
            return tensor.to(device)
    
    return tensor


def get_optimal_workers() -> int:
    """
    Get optimal number of DataLoader workers for current system.
    
    Returns:
        Recommended num_workers setting
    """
    if is_unified_memory_architecture():
        # Unified memory benefits from fewer workers
        return 0
    else:
        # Standard systems benefit from parallel data loading
        cpu_count = os.cpu_count() or 1
        return min(4, cpu_count)


def print_system_info():
    """Print system information relevant to DGX Spark optimization."""
    print("=" * 60)
    print("DGX Spark / Unified Memory System Information")
    print("=" * 60)
    
    import platform
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    
    print(f"Total Memory: {get_total_memory_gb():.1f} GB")
    print(f"Unified Memory: {is_unified_memory_architecture()}")
    print(f"Optimal Batch Size: {get_optimal_batch_size()}")
    print(f"Optimal Workers: {get_optimal_workers()}")
    print("=" * 60)


if __name__ == "__main__":
    print_system_info()
