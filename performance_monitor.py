"""
Performance monitoring and profiling utilities for video translation.
"""
import time
import psutil
import logging
from contextlib import contextmanager
import functools

# Attempt to import gpu_optimizer from the gpu_config module.
# This assumes gpu_config.py is in the same directory or Python path.
try:
    from gpu_config import gpu_optimizer
except ImportError:
    # Fallback if gpu_optimizer cannot be imported (e.g., running standalone or in a different setup)
    logging.warning("gpu_optimizer could not be imported from gpu_config. GPU status logging will be limited.")
    # Create a mock gpu_optimizer if it's critical for the class structure,
    # or handle its absence gracefully in methods.
    class MockGPUOptimizer:
        def log_gpu_status(self):
            logging.debug("MockGPUOptimizer: log_gpu_status called (GPU info not available).")
        def clear_cache(self):
            logging.debug("MockGPUOptimizer: clear_cache called.")
        def get_memory_info(self):
            return None
    gpu_optimizer = MockGPUOptimizer()


class PerformanceMonitor:
    """
    Monitor system and GPU performance during video translation.
    """

    def __init__(self):
        self.metrics = {}

    @contextmanager
    def timer(self, operation_name: str):
        """Context manager to time operations."""
        start_time = time.time()
        start_memory_mb = self._get_memory_usage_mb()
        
        logging.info(f"Starting {operation_name}...")
        if gpu_optimizer: # Check if gpu_optimizer is available
            gpu_optimizer.log_gpu_status() 
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory_mb = self._get_memory_usage_mb()
            
            duration_s = end_time - start_time
            # Ensure memory values are not None before subtraction
            memory_delta_mb = (end_memory_mb - start_memory_mb) if end_memory_mb is not None and start_memory_mb is not None else 0.0
            
            self.metrics[operation_name] = {
                'duration_s': duration_s,
                'memory_delta_mb': memory_delta_mb,
                'start_memory_mb': start_memory_mb if start_memory_mb is not None else 0.0,
                'end_memory_mb': end_memory_mb if end_memory_mb is not None else 0.0
            }
            
            logging.info(f"Completed {operation_name} in {duration_s:.2f}s (RAM Δ: {memory_delta_mb:+.1f}MB)")
            if gpu_optimizer: # Check if gpu_optimizer is available
                gpu_optimizer.log_gpu_status()

    def _get_memory_usage_mb(self):
        """Get current process memory usage (RSS) in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logging.warning(f"Could not get memory usage: {e}")
            return None

    def log_gpu_status_direct(self):
        """Log current GPU memory status using gpu_optimizer."""
        if gpu_optimizer: # Check if gpu_optimizer is available
            gpu_optimizer.log_gpu_status()

    def get_summary(self):
        """Get performance summary."""
        if not self.metrics:
            return "No performance data collected."
        
        total_time_s = sum(m['duration_s'] for m in self.metrics.values())
        # Net memory change is the sum of deltas. Peak usage would require tracking max end_memory_mb.
        net_memory_change_mb = sum(m['memory_delta_mb'] for m in self.metrics.values())
        
        summary = f"\n{'='*50}\n"
        summary += "PERFORMANCE SUMMARY\n"
        summary += f"{'='*50}\n"
        summary += f"Total Time: {total_time_s:.2f}s\n"
        summary += f"Net RAM Change (RSS): {net_memory_change_mb:+.1f}MB\n\n"
        
        summary += "Operation Breakdown:\n"
        for op_name, metrics_data in self.metrics.items():
            summary += f"  {op_name}: {metrics_data['duration_s']:.2f}s "
            summary += f"(RAM Δ: {metrics_data['memory_delta_mb']:+.1f}MB, "
            summary += f"Start RAM: {metrics_data['start_memory_mb']:.1f}MB, "
            summary += f"End RAM: {metrics_data['end_memory_mb']:.1f}MB)\n"
        
        summary += f"{'='*50}\n"
        return summary

# Global performance monitor instance
# This will be initialized when the module is imported.
performance_monitor = PerformanceMonitor()