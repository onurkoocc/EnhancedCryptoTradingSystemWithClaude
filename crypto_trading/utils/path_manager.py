"""
Path management utility for the crypto trading system.

This module provides centralized path management to ensure consistency
across the application and maintain organized directories.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict


class PathManager:
    """Centralized path management for the crypto trading system."""

    def __init__(self, base_dir: Optional[str] = None, create_dirs: bool = True):
        """Initialize the path manager.

        Args:
            base_dir: Base directory for the application (optional)
            create_dirs: Whether to create directories automatically
        """
        # Use current directory as base if not specified
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.create_dirs = create_dirs

        # Define standard directory structure
        self.dirs = {
            # Data directories
            'data': self.base_dir / 'data',
            'data_raw': self.base_dir / 'data' / 'raw',
            'data_processed': self.base_dir / 'data' / 'processed',
            'data_market': self.base_dir / 'data' / 'market_data',
            'data_cache': self.base_dir / 'data' / 'processed' / 'cache',

            # Log directories
            'logs': self.base_dir / 'logs',
            'logs_memory': self.base_dir / 'logs' / 'memory',
            'logs_temperature': self.base_dir / 'logs' / 'temperature',
            'logs_tensorboard': self.base_dir / 'logs' / 'tensorboard',

            # Model directories
            'models': self.base_dir / 'models',
            'models_saved': self.base_dir / 'models' / 'saved',
            'models_checkpoints': self.base_dir / 'models' / 'checkpoints',
            'models_tuning': self.base_dir / 'models' / 'tuning',

            # Results directories
            'results': self.base_dir / 'results',
            'results_backtest': self.base_dir / 'results' / 'backtest',
            'results_live': self.base_dir / 'results' / 'live',
            'results_reports': self.base_dir / 'results' / 'reports'
        }

        # Create all standard directories if requested
        if create_dirs:
            self.create_standard_dirs()

    def create_standard_dirs(self) -> None:
        """Create all standard directories."""
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str) -> Path:
        """Get a standard path by key.

        Args:
            key: Key for the standard path

        Returns:
            Path object for the requested directory
        """
        if key not in self.dirs:
            raise ValueError(f"Unknown path key: {key}")

        path = self.dirs[key]
        if self.create_dirs and not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path

    def get_date_dir(self, base_key: str, date: Optional[datetime] = None) -> Path:
        """Get a date-based subdirectory.

        Args:
            base_key: Key for the base directory
            date: Date for subdirectory (default: current date)

        Returns:
            Path object for the date directory
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime('%Y-%m-%d')
        base_dir = self.get_path(base_key)
        date_dir = base_dir / date_str

        if self.create_dirs:
            date_dir.mkdir(parents=True, exist_ok=True)

        return date_dir

    def get_timestamped_dir(self, base_key: str, prefix: str = '') -> Path:
        """Get a timestamped subdirectory.

        Args:
            base_key: Key for the base directory
            prefix: Optional prefix for the directory name

        Returns:
            Path object for the timestamped directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f"{prefix}_{timestamp}" if prefix else timestamp
        
        base_dir = self.get_path(base_key)
        timestamped_dir = base_dir / dir_name

        if self.create_dirs:
            timestamped_dir.mkdir(parents=True, exist_ok=True)

        return timestamped_dir

    def get_strategy_dir(self, strategy_name: str) -> Path:
        """Get a directory for a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Path object for the strategy directory
        """
        strategy_dir = self.get_path('results_backtest') / 'strategies' / strategy_name

        if self.create_dirs:
            strategy_dir.mkdir(parents=True, exist_ok=True)

        return strategy_dir

    def get_model_dir(self, model_name: str, ensemble: bool = False) -> Path:
        """Get directory for a specific model.

        Args:
            model_name: Name of the model
            ensemble: Whether this is an ensemble model

        Returns:
            Path object for the model directory
        """
        if ensemble:
            model_dir = self.get_path('models_saved') / f"{model_name}_ensemble"
        else:
            model_dir = self.get_path('models_saved')

        if self.create_dirs:
            model_dir.mkdir(parents=True, exist_ok=True)

        return model_dir

    def get_log_file(self, name: str, use_date_dir: bool = True) -> Path:
        """Get path for a log file.

        Args:
            name: Name of the log file
            use_date_dir: Whether to store in date-based directory

        Returns:
            Path object for the log file
        """
        if not name.endswith('.log'):
            name = f"{name}.log"

        if use_date_dir:
            log_dir = self.get_date_dir('logs')
        else:
            log_dir = self.get_path('logs')

        return log_dir / name

    def get_data_file(self, name: str, timeframe: str = '30m') -> Path:
        """Get path for a data file.

        Args:
            name: Name of the data file (e.g., 'btc')
            timeframe: Timeframe for the data ('30m', '4h', 'daily')

        Returns:
            Path object for the data file
        """
        if not name.endswith('.csv'):
            name = f"{name}_data_{timeframe}.csv"

        return self.get_path('data_market') / name

    def cleanup_old_directories(self, base_key: str, max_age_days: int = 30) -> List[str]:
        """Clean up old directories based on age.

        Args:
            base_key: Key for the base directory
            max_age_days: Maximum age in days to keep

        Returns:
            List of removed directory paths
        """
        base_dir = self.get_path(base_key)
        threshold = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        removed = []

        # Check all directories in the base directory
        for item in base_dir.iterdir():
            if item.is_dir():
                try:
                    # Get directory creation time
                    created_time = item.stat().st_ctime
                    if created_time < threshold:
                        # Remove directory if older than threshold
                        shutil.rmtree(item)
                        removed.append(str(item))
                except Exception as e:
                    print(f"Error cleaning up directory {item}: {e}")

        return removed

    def ensure_clean_directory(self, path_key: str, max_files: int = 100, 
                              max_age_days: Optional[int] = None) -> None:
        """Ensure a directory doesn't exceed a certain number of files.

        Args:
            path_key: Key for the directory to clean
            max_files: Maximum number of files to keep
            max_age_days: Optional maximum age in days to keep files
        """
        directory = self.get_path(path_key)
        
        # List all files with their creation times
        files = []
        for item in directory.iterdir():
            if item.is_file():
                created_time = item.stat().st_ctime
                files.append((item, created_time))
        
        # Remove files by age if specified
        if max_age_days is not None:
            threshold = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            for file_path, created_time in files:
                if created_time < threshold:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"Error removing old file {file_path}: {e}")
        
        # If we still have too many files, remove oldest ones
        if len(files) > max_files:
            # Sort by creation time (oldest first)
            files.sort(key=lambda x: x[1])
            
            # Remove oldest files that exceed the limit
            for file_path, _ in files[:(len(files) - max_files)]:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Error removing excess file {file_path}: {e}")


# Create a singleton instance
_path_manager = None

def get_path_manager(base_dir: Optional[str] = None, create_dirs: bool = True) -> PathManager:
    """Get the singleton PathManager instance.

    Args:
        base_dir: Base directory for the application (optional)
        create_dirs: Whether to create directories automatically

    Returns:
        PathManager instance
    """
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager(base_dir, create_dirs)
    return _path_manager


if __name__ == "__main__":
    # Example usage
    pm = get_path_manager()
    
    # Print standard paths
    print("Standard Paths:")
    for key, path in pm.dirs.items():
        print(f"{key:20}: {path}")
    
    # Get date-based directory
    log_dir = pm.get_date_dir('logs')
    print(f"\nToday's log directory: {log_dir}")
    
    # Get timestamped directory
    run_dir = pm.get_timestamped_dir('results_backtest', 'backtest_run')
    print(f"\nTimestamped run directory: {run_dir}")
