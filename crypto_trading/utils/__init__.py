"""
Utility functions and classes for the crypto trading system.

This package provides various utilities used throughout the system
including logging, monitoring, error handling, and helper functions.
"""

from .logging_utils import LoggerFactory, exception_handler, RetryWithBackoff
from .memory_monitor import log_memory_usage, clear_memory, monitor_memory, MemoryMonitor
from .temperature_monitor import TemperatureMonitor, setup_temperature_monitoring

__all__ = [
    'LoggerFactory',
    'exception_handler',
    'RetryWithBackoff',
    'log_memory_usage',
    'clear_memory',
    'monitor_memory',
    'MemoryMonitor',
    'TemperatureMonitor',
    'setup_temperature_monitoring'
]


def cleanup_old_files(max_age_days: int = 30) -> None:
    """Clean up old logs and results files.

    Args:
        max_age_days: Maximum age in days to keep files
    """
    path_manager = get_path_manager()

    # Clean up old log directories
    path_manager.cleanup_old_directories('logs', max_age_days)

    # Clean up old backtest results
    path_manager.cleanup_old_directories('results_backtest', max_age_days)

    # Clean up old model checkpoints (keep fewer of these)
    path_manager.cleanup_old_directories('models_checkpoints', max_age_days // 2)

    # Clean up excess files in some directories
    path_manager.ensure_clean_directory('logs_memory', max_files=100)
    path_manager.ensure_clean_directory('logs_temperature', max_files=100)