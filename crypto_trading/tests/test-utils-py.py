"""
Unit tests for the utility modules.

This module provides tests for the utility functions and classes used throughout
the application, including logging, error handling, and memory monitoring.
"""

import pytest
import logging
import os
import time
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock

from crypto_trading.utils.logging_utils import LoggerFactory, exception_handler, RetryWithBackoff
from crypto_trading.utils.memory_monitor import log_memory_usage, clear_memory, MemoryMonitor
from crypto_trading.utils.temperature_monitor import get_current_temperatures


class TestLoggerFactory:
    """Tests for LoggerFactory class."""
    
    def test_singleton_pattern(self):
        """Test that LoggerFactory follows the singleton pattern."""
        factory1 = LoggerFactory()
        factory2 = LoggerFactory()
        
        # Should be the same instance
        assert factory1 is factory2
    
    def test_get_logger(self):
        """Test getting a logger from the factory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize factory with temporary directory
            factory = LoggerFactory(log_level=logging.INFO, log_dir=temp_dir)
            
            # Get logger
            logger = factory.get_logger('TestLogger')
            
            # Assert
            assert logger is not None
            assert logger.name == 'TestLogger'
            assert logger.level == logging.INFO
            
            # Get the same logger again
            logger2 = factory.get_logger('TestLogger')
            
            # Should be the same instance
            assert logger is logger2
            
            # Check if log file was created
            log_files = [f for f in os.listdir(temp_dir) if f.startswith('TestLogger_')]
            assert len(log_files) > 0


class TestExceptionHandler:
    """Tests for exception_handler decorator."""
    
    def test_exception_handling_reraise(self):
        """Test exception_handler with reraise=True."""
        # Define a function that raises an exception
        @exception_handler(reraise=True)
        def failing_function():
            raise ValueError("Test error")
        
        # Should reraise the exception
        with pytest.raises(ValueError):
            failing_function()
    
    def test_exception_handling_no_reraise(self):
        """Test exception_handler with reraise=False."""
        # Define a function that raises an exception
        @exception_handler(reraise=False, fallback_value="Fallback")
        def failing_function():
            raise ValueError("Test error")
        
        # Should return fallback value
        result = failing_function()
        assert result == "Fallback"
    
    def test_no_exception(self):
        """Test exception_handler with no exception."""
        # Define a function that does not raise an exception
        @exception_handler(reraise=True)
        def normal_function():
            return "Success"
        
        # Should work normally
        result = normal_function()
        assert result == "Success"


class TestRetryWithBackoff:
    """Tests for RetryWithBackoff decorator."""
    
    def test_successful_retry(self):
        """Test successful retry after failures."""
        # Mock counter to track attempts
        attempts = {'count': 0}
        
        # Define a function that fails first then succeeds
        @RetryWithBackoff(max_retries=3, initial_backoff=0.01, exceptions=(ValueError,))
        def flaky_function():
            attempts['count'] += 1
            if attempts['count'] < 3:
                raise ValueError("Temporary error")
            return "Success"
        
        # Should retry and eventually succeed
        result = flaky_function()
        assert result == "Success"
        assert attempts['count'] == 3
    
    def test_max_retries_exceeded(self):
        """Test case where max retries are exceeded."""
        # Mock counter to track attempts
        attempts = {'count': 0}
        
        # Define a function that always fails
        @RetryWithBackoff(max_retries=2, initial_backoff=0.01, exceptions=(ValueError,))
        def always_fails():
            attempts['count'] += 1
            raise ValueError("Always fails")
        
        # Should retry but eventually raise the exception
        with pytest.raises(ValueError):
            always_fails()
        
        # Should have attempted 3 times total (initial + 2 retries)
        assert attempts['count'] == 3
    
    def test_non_matching_exception(self):
        """Test case with an exception not in the retry list."""
        # Mock counter to track attempts
        attempts = {'count': 0}
        
        # Define a function that raises a different exception
        @RetryWithBackoff(max_retries=3, initial_backoff=0.01, exceptions=(ValueError,))
        def different_exception():
            attempts['count'] += 1
            raise KeyError("Not a ValueError")
        
        # Should not retry for this exception
        with pytest.raises(KeyError):
            different_exception()
        
        # Should have attempted only once
        assert attempts['count'] == 1


class TestMemoryMonitor:
    """Tests for memory monitoring utilities."""
    
    def test_log_memory_usage(self):
        """Test logging memory usage."""
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            # Log memory usage to file
            memory_gb = log_memory_usage(temp_file.name)
            
            # Check results
            assert memory_gb > 0
            
            # File should contain memory log
            temp_file.seek(0)
            content = temp_file.read()
            assert 'memory' in content.lower() or ',' in content
    
    @patch('crypto_trading.utils.memory_monitor.clear_session')
    @patch('gc.collect')
    def test_clear_memory(self, mock_gc_collect, mock_clear_session):
        """Test memory cleanup function."""
        # Run memory cleanup
        result = clear_memory()
        
        # Check function calls
        assert mock_clear_session.called
        assert mock_gc_collect.called
        assert result is True
    
    @patch('psutil.Process')
    def test_memory_monitor_initialization(self, mock_process):
        """Test MemoryMonitor initialization."""
        # Setup mocks
        mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 1024  # 1 GB
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize monitor
            monitor = MemoryMonitor(
                threshold_gb=2,
                log_dir=temp_dir,
                log_interval=1,
                check_interval=1
            )
            
            # Check initialization
            assert monitor is not None
            assert monitor.threshold_gb == 2
            assert os.path.exists(temp_dir)
            
            # Check log files
            log_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
            assert len(log_files) > 0


@pytest.mark.skipif(not os.path.exists('/usr/bin/sensors'), reason="Temperature monitoring requires lm-sensors")
class TestTemperatureMonitor:
    """Tests for temperature monitoring utilities."""
    
    def test_get_current_temperatures(self):
        """Test getting current temperature readings."""
        # Get temperatures
        temps = get_current_temperatures()
        
        # Check structure
        assert 'cpu_temp' in temps
        assert 'gpu_available' in temps
        
        # Values should be reasonable or zero
        assert 0 <= temps['cpu_temp'] < 120  # In celsius
        assert isinstance(temps['gpu_available'], bool)


class TestExceptHandler:
    """Additional tests for exception_handler with various scenarios."""
    
    def test_log_args(self):
        """Test the log_args parameter."""
        mock_logger = MagicMock()
        
        # Define a function that raises an exception
        @exception_handler(logger=mock_logger, reraise=False, log_args=True)
        def function_with_args(arg1, arg2, kwarg1=None):
            raise ValueError("Test error")
        
        # Call the function
        function_with_args("test1", "test2", kwarg1="test3")
        
        # Check logging
        for call in mock_logger.error.call_args_list:
            args, _ = call
            if len(args) > 0 and "Function arguments" in args[0]:
                assert "test1" in args[0]
                assert "test2" in args[0]
                assert "test3" in args[0]
                break
        else:
            pytest.fail("Arguments were not logged")
    
    def test_truncate_large_args(self):
        """Test truncation of large arguments."""
        mock_logger = MagicMock()
        
        # Define a function that raises an exception
        @exception_handler(logger=mock_logger, reraise=False, log_args=True)
        def function_with_large_arg(large_arg):
            raise ValueError("Test error")
        
        # Call with a large argument
        large_string = "x" * 2000
        function_with_large_arg(large_string)
        
        # Check logging
        truncated = False
        for call in mock_logger.error.call_args_list:
            args, _ = call
            if len(args) > 0 and "Function arguments" in args[0]:
                if "[truncated]" in args[0]:
                    truncated = True
                break
        
        assert truncated, "Large argument should be truncated"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
