"""
Centralized logging and error handling utilities.

This module provides a consistent logging interface and error handling
decorators to be used throughout the application.
"""

import logging
import functools
import traceback
import sys
import os
import time
from datetime import datetime
from typing import Optional, Type, Callable, Any, Union, Tuple, List, Dict


class LoggerFactory:
    """Factory class for creating standardized loggers."""

    _instance = None
    _loggers = {}

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(LoggerFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_level=logging.INFO, log_dir='logs',
                 console_logging=True, file_logging=True,
                 log_format=None):
        """Initialize the logger factory.

        Args:
            log_level: Default logging level
            log_dir: Directory for log files
            console_logging: Whether to log to console
            file_logging: Whether to log to file
            log_format: Optional custom log format string
        """
        # Only initialize once
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.log_level = log_level
        self.log_dir = log_dir
        self.console_logging = console_logging
        self.file_logging = file_logging
        self.log_format = log_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # Ensure log directory exists
        if self.file_logging and self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        # Set up root logger
        self._setup_root_logger()

        self._initialized = True

    def _setup_root_logger(self):
        """Set up the root logger with basic configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter(self.log_format)

        # Console handler with proper encoding
        if self.console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            # Ensure encoding is properly set for Windows
            console_handler.setStream(sys.stdout)
            root_logger.addHandler(console_handler)

        # File handler for root logger
        if self.file_logging and self.log_dir:
            file_path = os.path.join(self.log_dir, f'system_{datetime.now():%Y%m%d_%H%M%S}.log')
            # Use encoding='utf-8' to handle Unicode characters
            file_handler = logging.FileHandler(file_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    def get_logger(self, name, log_level=None, module_specific_file=True) -> logging.Logger:
        """Get a logger with the specified name.

        Args:
            name: Name of the logger
            log_level: Optional specific log level for this logger
            module_specific_file: Whether to create a separate log file for this module

        Returns:
            Logger instance
        """
        # Return existing logger if already created
        if name in self._loggers:
            return self._loggers[name]

        # Create new logger
        logger = logging.getLogger(name)

        # Set level if specified
        if log_level is not None:
            logger.setLevel(log_level)

        # Add module-specific file handler if requested
        if self.file_logging and module_specific_file and self.log_dir:
            # Create formatter
            formatter = logging.Formatter(self.log_format)

            # Create file handler with module name
            safe_name = name.replace('.', '_')
            file_path = os.path.join(self.log_dir, f'{safe_name}_{datetime.now():%Y%m%d_%H%M%S}.log')
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Store logger
        self._loggers[name] = logger

        return logger


def exception_handler(logger=None, reraise=True, fallback_value=None,
                      log_args=False):
    """Decorator to handle exceptions and log them.

    Args:
        logger: Logger to use for logging exceptions
               (if None, creates a logger based on function name)
        reraise: Whether to re-raise the exception after logging
        fallback_value: Value to return if an exception occurs and reraise is False
        log_args: Whether to log function arguments

    Returns:
        Decorated function
    """

    def decorator(func):
        # Get function module and name for logging
        module_name = func.__module__
        func_name = func.__qualname__

        # Get or create logger
        nonlocal logger
        if logger is None:
            logger = LoggerFactory().get_logger(module_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the exception with traceback
                error_msg = f"Exception in {func_name}: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")

                # Log arguments if requested
                if log_args:
                    # Limit argument logging for large objects
                    safe_args = [_truncate_arg(arg) for arg in args]
                    safe_kwargs = {k: _truncate_arg(v) for k, v in kwargs.items()}
                    logger.error(f"Function arguments - args: {safe_args}, kwargs: {safe_kwargs}")

                # Re-raise or return fallback
                if reraise:
                    raise
                return fallback_value

        return wrapper

    return decorator


def _truncate_arg(arg, max_length=1000):
    """Truncate string representation of argument for logging."""
    if isinstance(arg, (str, bytes)):
        s = str(arg)
        if len(s) > max_length:
            return s[:max_length] + "... [truncated]"
    return arg


class RetryWithBackoff:
    """Decorator to retry a function with exponential backoff.

    This decorator will retry a function if it raises specified exceptions,
    with an exponential backoff between retries.
    """

    def __init__(self, max_retries=3, initial_backoff=1, backoff_multiplier=2,
                 exceptions=(Exception,), logger=None, retry_msg=None,
                 failure_msg=None):
        """Initialize the retry decorator.

        Args:
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff in seconds
            backoff_multiplier: Multiplier for backoff on subsequent retries
            exceptions: Tuple of exceptions to catch and retry on
            logger: Logger to use for logging retries
            retry_msg: Custom message template for retry attempts
            failure_msg: Custom message template for final failure
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_multiplier = backoff_multiplier
        self.exceptions = exceptions
        self.logger = logger
        self.retry_msg = retry_msg or "Retry {retry}/{max_retries} for {func_name} after error: {error}. Waiting {backoff}s"
        self.failure_msg = failure_msg or "Maximum retries ({max_retries}) reached for {func_name}. Last error: {error}"

    def __call__(self, func):
        # Get function module and name for logging
        module_name = func.__module__
        func_name = func.__qualname__

        # Get or create logger
        if self.logger is None:
            self.logger = LoggerFactory().get_logger(module_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            backoff = self.initial_backoff
            last_exception = None

            for retry in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e
                    if retry < self.max_retries:
                        # Format retry message
                        msg = self.retry_msg.format(
                            retry=retry + 1,
                            max_retries=self.max_retries,
                            func_name=func_name,
                            error=str(e),
                            backoff=backoff
                        )
                        self.logger.warning(msg)

                        # Wait before retrying
                        time.sleep(backoff)
                        backoff *= self.backoff_multiplier
                    else:
                        # Format failure message
                        msg = self.failure_msg.format(
                            max_retries=self.max_retries,
                            func_name=func_name,
                            error=str(e)
                        )
                        self.logger.error(msg)

            # If we get here, all retries have failed
            raise last_exception

        return wrapper


def log_execution_time(logger=None, level=logging.INFO):
    """Decorator to log function execution time.

    Args:
        logger: Logger to use (if None, creates logger based on function module)
        level: Logging level to use

    Returns:
        Decorated function
    """

    def decorator(func):
        # Get function module and name for logging
        module_name = func.__module__
        func_name = func.__qualname__

        # Get or create logger
        nonlocal logger
        if logger is None:
            logger = LoggerFactory().get_logger(module_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Log execution time
            logger.log(level, f"Function {func_name} executed in {execution_time:.3f} seconds")

            return result

        return wrapper

    return decorator