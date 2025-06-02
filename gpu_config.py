"""
GPU Configuration and Optimization Settings
"""
import torch
import logging
from accelerate import Accelerator

# Attempt to import a project-specific config if it exists, otherwise use defaults.
# This try-except block is designed to be flexible.
try:
    # Assuming 'config' is a module or an object that can be imported
    # and has a .get() method similar to a config parser.
    # For this project, we'll assume it's not present and defaults will be used.
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
            # Defaults if config.py is not found or 'config' object is not in it
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
        # Initialize accelerator with mixed precision
        self.accelerator = Accelerator(
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=1,
            cpu=not torch.cuda.is_available()
        )
        
        self.device = self.accelerator.device
        
        # Set optimal PyTorch settings for performance
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True # For Ampere and newer
            torch.backends.cudnn.allow_tf32 = True      # For Ampere and newer
            
            # Print GPU information
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
        
        # Move model to device using accelerator
        model = self.accelerator.prepare(model)
        
        # Enable gradient checkpointing if supported
        if self.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            try: # Some models might not actually support it even if the method exists
                model.gradient_checkpointing_enable()
                logging.info("Gradient checkpointing enabled")
            except Exception as e:
                logging.warning(f"Could not enable gradient checkpointing: {e}")

        # Apply torch.compile for potential speed boost (PyTorch 2.0+)
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
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3 # PyTorch uses 'reserved' for 'cached'
            total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved, # Using 'reserved' as it's what PyTorch provides
                'total_gb': total_memory,
                'free_approx_gb': total_memory - reserved # Free is total - reserved (cached)
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

# Global GPU optimizer instance
# This will be initialized when the module is imported.
gpu_optimizer = GPUOptimizer()