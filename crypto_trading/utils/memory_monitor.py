"""
Memory monitoring and management utilities.

This module provides tools for monitoring memory usage and cleaning up
memory when needed to prevent out-of-memory errors.
"""

import os
import time
import psutil
import threading
import gc
import logging
from datetime import datetime
from typing import Optional, Callable, Union, List, Dict, Any
import traceback
import numpy as np
import pandas as pd

# Try to import TensorFlow, but handle the case where it's not installed
try:
    import tensorflow as tf
    from tensorflow.python.keras.backend import clear_session

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    clear_session = lambda: None  # Dummy function if TF not available


class MemoryMonitor:
    """Memory monitoring class with configurable thresholds and actions."""

    def __init__(self, threshold_gb: float = 16,
                 log_dir: str = 'logs/memory',
                 log_interval: int = 60,
                 check_interval: int = 10,
                 cleanup_callbacks: List[Callable] = None,
                 alert_callbacks: List[Callable] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the memory monitor.

        Args:
            threshold_gb: Memory threshold in GB to trigger cleanup
            log_dir: Directory for memory log files
            log_interval: Interval in seconds between memory usage logs
            check_interval: Interval in seconds between memory checks
            cleanup_callbacks: List of callbacks to call when threshold is exceeded
            alert_callbacks: List of callbacks to call for alerts
            logger: Logger to use
        """
        self.threshold_gb = threshold_gb
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.check_interval = check_interval
        self.cleanup_callbacks = cleanup_callbacks or []
        self.alert_callbacks = alert_callbacks or []
        self.logger = logger or logging.getLogger('MemoryMonitor')
        self.is_running = False
        self.last_log_time = 0
        self.monitor_thread = None

        # Initialize log directory
        os.makedirs(log_dir, exist_ok=True)

        # Initialize log file
        self.log_file = os.path.join(log_dir, f'memory_log_{datetime.now():%Y%m%d_%H%M%S}.csv')
        with open(self.log_file, 'w') as f:
            f.write('timestamp,memory_gb,memory_percent,available_gb,swap_gb,cpu_percent\n')

        # Initialize warning log
        self.warning_file = os.path.join(log_dir, f'memory_warnings_{datetime.now():%Y%m%d_%H%M%S}.log')

    def start(self, in_background: bool = True) -> Optional[threading.Thread]:
        """Start memory monitoring.

        Args:
            in_background: Whether to run monitoring in a background thread

        Returns:
            Thread object if running in background, None otherwise
        """
        self.is_running = True

        if in_background:
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True
            )
            self.monitor_thread.start()
            self.logger.info(f"Memory monitoring started in background. Log: {self.log_file}")
            return self.monitor_thread
        else:
            self.logger.info(f"Memory monitoring started in foreground. Log: {self.log_file}")
            self._monitor_loop()
            return None

    def stop(self) -> None:
        """Stop memory monitoring."""
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            if self.monitor_thread.is_alive():
                self.logger.warning("Memory monitor thread did not terminate properly")
        self.logger.info("Memory monitoring stopped")

    def log_memory_usage(self, force: bool = False) -> Dict[str, float]:
        """Log current memory usage.

        Args:
            force: Whether to force logging regardless of interval

        Returns:
            Dict with memory usage metrics
        """
        current_time = time.time()

        # Only log at specified intervals unless forced
        if not force and (current_time - self.last_log_time) < self.log_interval:
            return self._get_memory_metrics()

        self.last_log_time = current_time

        # Get metrics
        metrics = self._get_memory_metrics()

        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()},{metrics['memory_gb']:.4f},{metrics['memory_percent']:.2f},"
                    f"{metrics['available_gb']:.4f},{metrics['swap_gb']:.4f},{metrics['cpu_percent']:.2f}\n")

        return metrics

    def _get_memory_metrics(self) -> Dict[str, float]:
        """Get current memory usage metrics.

        Returns:
            Dict with memory metrics
        """
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # GB

        # System metrics
        system = psutil.virtual_memory()
        memory_percent = system.percent
        available_gb = system.available / (1024 * 1024 * 1024)  # GB

        # Swap metrics
        swap = psutil.swap_memory()
        swap_gb = swap.used / (1024 * 1024 * 1024)  # GB

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)

        return {
            'memory_gb': process_memory,
            'memory_percent': memory_percent,
            'available_gb': available_gb,
            'swap_gb': swap_gb,
            'cpu_percent': cpu_percent
        }

    def check_memory(self, force_cleanup: bool = False) -> bool:
        """Check memory and perform cleanup if threshold exceeded.

        Args:
            force_cleanup: Whether to force cleanup regardless of threshold

        Returns:
            True if cleanup was performed, False otherwise
        """
        metrics = self._get_memory_metrics()
        memory_gb = metrics['memory_gb']

        if memory_gb > self.threshold_gb or force_cleanup:
            self.logger.warning(
                f"Memory usage ({memory_gb:.2f}GB) exceeded threshold ({self.threshold_gb}GB). "
                "Performing cleanup..."
            )

            # Log warning
            with open(self.warning_file, 'a') as f:
                f.write(f"{datetime.now()}: Memory usage reached {memory_gb:.2f}GB "
                        f"(threshold: {self.threshold_gb}GB)\n")

            # Call cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.error(f"Error in cleanup callback: {str(e)}")

            # Default cleanup
            self._cleanup_memory()

            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {str(e)}")

            return True

        return False

    def _cleanup_memory(self) -> None:
        """Perform memory cleanup operations."""
        # Clear TensorFlow session if available
        if TENSORFLOW_AVAILABLE:
            try:
                clear_session()
                self.logger.info("TensorFlow session cleared")
            except Exception as e:
                self.logger.error(f"Error clearing TensorFlow session: {str(e)}")

        # Run garbage collection
        collected = gc.collect(generation=2)
        self.logger.info(f"Garbage collection: {collected} objects collected")

        # Force memory usage logging after cleanup
        post_cleanup_metrics = self.log_memory_usage(force=True)
        self.logger.info(f"Memory after cleanup: {post_cleanup_metrics['memory_gb']:.2f}GB")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info(f"Memory monitoring started. Threshold: {self.threshold_gb}GB")

        try:
            while self.is_running:
                # Log current memory usage
                metrics = self.log_memory_usage()
                memory_gb = metrics['memory_gb']

                # Check if threshold exceeded
                if memory_gb > self.threshold_gb:
                    self.check_memory()

                # Sleep until next check
                time.sleep(self.check_interval)
        except Exception as e:
            self.logger.error(f"Error in memory monitor: {str(e)}")
            self.logger.error(traceback.format_exc())

            # Try to restart if still running
            if self.is_running:
                self.logger.info("Attempting to restart memory monitor...")
                time.sleep(self.check_interval)
                self._monitor_loop()


def log_memory_usage(log_file: Optional[str] = None) -> float:
    """Log current memory usage.

    Args:
        log_file: Optional file path to log memory usage

    Returns:
        Current memory usage in GB
    """
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)

    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Write to log file
        with open(log_file, "a") as f:
            f.write(f"{datetime.now()},{memory_gb:.4f}\n")

    return memory_gb


def clear_memory() -> bool:
    """Clear memory by releasing TensorFlow resources and running garbage collection.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Clear TensorFlow session
        if TENSORFLOW_AVAILABLE:
            clear_session()

        # Force garbage collection
        for i in range(3):
            gc.collect(i)

        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Error clearing memory: {str(e)}")
        return False


def monitor_memory(threshold_gb: float = 16,
                   interval_seconds: int = 60,
                   log_file: Optional[str] = None,
                   in_background: bool = True) -> Optional[threading.Thread]:
    """Start memory monitoring with the given parameters.

    Args:
        threshold_gb: Memory threshold in GB to trigger cleanup
        interval_seconds: Interval in seconds between memory checks
        log_file: File to log memory usage
        in_background: Whether to run in background thread

    Returns:
        Thread object if running in background, None otherwise
    """
    # Create a monitor instance
    monitor = MemoryMonitor(
        threshold_gb=threshold_gb,
        log_interval=interval_seconds,
        check_interval=interval_seconds // 3,  # Check more frequently than log
        log_dir=os.path.dirname(log_file) if log_file else 'logs/memory'
    )

    # Start monitoring
    return monitor.start(in_background)


def memory_usage_decorator(logger: Optional[logging.Logger] = None,
                           threshold_gb: Optional[float] = None):
    """Decorator to log memory usage before and after function execution.

    Args:
        logger: Logger to use for logging memory usage
        threshold_gb: Optional threshold to warn about high memory usage

    Returns:
        Decorated function
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log memory before
            mem_before = log_memory_usage()
            logger.info(f"Memory before {func.__name__}: {mem_before:.2f}GB")

            # Call function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Log memory after
            mem_after = log_memory_usage()
            mem_diff = mem_after - mem_before
            logger.info(
                f"Memory after {func.__name__}: {mem_after:.2f}GB "
                f"(Δ{mem_diff:+.2f}GB, {execution_time:.2f}s)"
            )

            # Check threshold
            if threshold_gb is not None and mem_after > threshold_gb:
                logger.warning(
                    f"Memory usage after {func.__name__} ({mem_after:.2f}GB) "
                    f"exceeds threshold ({threshold_gb}GB)"
                )

            return result

        return wrapper

    return decorator


def find_memory_leaks(top_n: int = 10) -> List[Dict[str, Any]]:
    """Identify potential memory leaks by finding largest objects.

    Args:
        top_n: Number of top memory-consuming objects to return

    Returns:
        List of dictionaries with object info
    """
    # Get all objects
    objects = gc.get_objects()

    # Track size and type of large objects
    object_info = []

    for obj in objects:
        try:
            # Skip small objects
            size = sys.getsizeof(obj)
            if size < 1024 * 1024:  # Skip objects smaller than 1MB
                continue

            # Get object type and size
            obj_type = type(obj).__name__

            # Get additional info based on type
            extra_info = {}

            if isinstance(obj, pd.DataFrame):
                extra_info['shape'] = obj.shape
                extra_info['memory_usage'] = obj.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            elif isinstance(obj, np.ndarray):
                extra_info['shape'] = obj.shape
                extra_info['dtype'] = str(obj.dtype)
            elif isinstance(obj, (list, tuple, set, dict)):
                extra_info['length'] = len(obj)

            # Add to results
            object_info.append({
                'type': obj_type,
                'size_mb': size / (1024 * 1024),
                'info': extra_info
            })
        except Exception:
            # Skip problematic objects
            continue

    # Sort by size
    object_info.sort(key=lambda x: x['size_mb'], reverse=True)

    # Return top N
    return object_info[:top_n]