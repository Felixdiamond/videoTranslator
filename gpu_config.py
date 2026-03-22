"""
GPU Configuration and Optimization Settings
"""
import torch
import logging
from accelerate import Accelerator

# Optional project config override.
try:
    from config import config as project_specific_config
except ImportError:
    project_specific_config = None
    logging.info("Project-specific 'config.py' not found or 'config' object missing. Using default GPUOptimizer settings.")


class GPUOptimizer:
    """
    Centralized GPU optimization configuration.
    """

    def __init__(self, mixed_precision='fp16', gradient_checkpointing=True):
        if project_specific_config:
            self.mixed_precision = project_specific_config.get("gpu", "mixed_precision", fallback=mixed_precision)
            self.gradient_checkpointing = project_specific_config.get("gpu", "gradient_checkpointing", fallback=gradient_checkpointing)
            self.compile_mode = project_specific_config.get("gpu", "compile_mode", fallback="reduce-overhead")
        else:
            self.mixed_precision = mixed_precision
            self.gradient_checkpointing = gradient_checkpointing
            self.compile_mode = "reduce-overhead"
        
        self.accelerator = None
        self.device = None
        self.setup_gpu()

    def setup_gpu(self):
        """
        Initialize GPU settings and accelerator.
        """
        self.accelerator = Accelerator(
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=1,
            cpu=not torch.cuda.is_available()
        )
        
        self.device = self.accelerator.device
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                memory_gb = gpu_props.total_memory / 1024**3
                logging.info(f"GPU {i}: {gpu_props.name} ({memory_gb:.1f} GB)")
        
        logging.info(f"Using device: {self.device}")
        logging.info(f"Mixed precision: {self.mixed_precision}")

    def optimize_model(self, model, compile_mode=None):
        """
        Apply optimizations to a model.
        """
        compile_mode_to_use = compile_mode or self.compile_mode
        
        model = self.accelerator.prepare(model)
        
        if self.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            try:
                model.gradient_checkpointing_enable()
                logging.info("Gradient checkpointing enabled")
            except Exception as e:
                logging.warning(f"Could not enable gradient checkpointing: {e}")

        # Apply torch.compile when available.
        if hasattr(torch, 'compile') and torch.cuda.is_available() and self.device.type == 'cuda':
            logging.info(f"Applying torch.compile with mode: {compile_mode_to_use}")
            try:
                model = torch.compile(model, mode=compile_mode_to_use)
            except Exception as e:
                logging.warning(f"torch.compile failed with mode {compile_mode_to_use}: {e}. Model will not be compiled.")
        
        return model

    def get_memory_info(self):
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total_memory,
                'free_approx_gb': total_memory - reserved
            }
        return None

    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU cache cleared")

    def log_gpu_status(self):
        """Log current GPU memory status."""
        memory_info = self.get_memory_info()
        if memory_info:
            logging.info(
                f"GPU Memory - Allocated: {memory_info['allocated_gb']:.2f}GB, "
                f"Reserved (Cached): {memory_info['reserved_gb']:.2f}GB, "
                f"Free (approx): {memory_info['free_approx_gb']:.2f}GB, "
                f"Total: {memory_info['total_gb']:.2f}GB"
            )

# Global GPU optimizer instance.
gpu_optimizer = GPUOptimizer()


def detect_hardware_tier() -> str:
    """
    Returns one of: 'cpu_low', 'cpu_high', 'gpu_low', 'gpu_medium', 'gpu_high'.
    Used to select appropriately-sized models for the available hardware.
    """
    import psutil
    if not torch.cuda.is_available():
        ram_gb = psutil.virtual_memory().total / 1024 ** 3
        return "cpu_high" if ram_gb >= 16 else "cpu_low"

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if vram_gb < 6:
        return "gpu_low"
    if vram_gb < 12:
        return "gpu_medium"
    return "gpu_high"