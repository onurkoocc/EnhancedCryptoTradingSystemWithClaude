"""
GPU and CPU temperature monitoring utilities.

This module provides tools for monitoring GPU and CPU temperatures
and automatically applying throttling to prevent overheating.
"""

import os
import time
import subprocess
import threading
import psutil
import signal
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union, Callable


class TemperatureMonitor:
    """Monitors GPU and CPU temperatures and applies throttling if needed."""

    def __init__(self,
                 gpu_temp_threshold: int = 75,  # °C - conservative max for modern GPUs
                 cpu_temp_threshold: int = 85,  # °C - conservative max for modern CPUs
                 check_interval: int = 10,  # seconds between checks
                 log_interval: int = 60,  # seconds between regular logs
                 power_limit_percent: int = 70,  # % of max GPU power
                 log_dir: str = 'logs/temperature',
                 logger: Optional[logging.Logger] = None):
        """Initialize the temperature monitor.

        Args:
            gpu_temp_threshold: Temperature threshold in °C for GPU
            cpu_temp_threshold: Temperature threshold in °C for CPU
            check_interval: Interval in seconds between temperature checks
            log_interval: Interval in seconds between regular temperature logs
            power_limit_percent: Percentage of maximum GPU power to use
            log_dir: Directory for log files
            logger: Logger to use
        """
        self.gpu_temp_threshold = gpu_temp_threshold
        self.cpu_temp_threshold = cpu_temp_threshold
        self.check_interval = check_interval
        self.log_interval = log_interval
        self.power_limit_percent = power_limit_percent
        self.log_dir = log_dir
        self.logger = logger or logging.getLogger('TemperatureMonitor')

        self.is_running = False
        self.last_log_time = 0
        self.throttling_active = False
        self.monitor_thread = None

        # Detect available hardware
        self.gpu_available = self._check_nvidia_gpu()
        self.has_temp_sensors = self._check_temp_sensors()

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Create log files
        self.log_file = os.path.join(log_dir, f'temperature_log_{datetime.now():%Y%m%d_%H%M%S}.csv')
        self.warning_file = os.path.join(log_dir, f'temperature_warnings_{datetime.now():%Y%m%d_%H%M%S}.log')

        # Create log file with header if it does not exist
        with open(self.log_file, "w") as f:
            f.write("timestamp,gpu_temp,cpu_temp,gpu_power,gpu_utilization,cpu_utilization\n")

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals.

        Args:
            sig: Signal number
            frame: Current stack frame
        """
        self.logger.info("Temperature monitor shutting down...")
        self.is_running = False

        # Restore GPU power limit if we modified it
        if self.throttling_active:
            try:
                self._reset_gpu_power_limit()
            except Exception as e:
                self.logger.error(f"Error resetting GPU power limit: {str(e)}")

    def _check_nvidia_gpu(self) -> bool:
        """Check if NVIDIA GPU is available.

        Returns:
            True if NVIDIA GPU is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _check_temp_sensors(self) -> bool:
        """Check if temperature sensors are available.

        Returns:
            True if temperature sensors are available, False otherwise
        """
        try:
            temps = psutil.sensors_temperatures()
            return bool(temps)
        except Exception:
            return False

    def _get_gpu_temp(self) -> int:
        """Get NVIDIA GPU temperature using nvidia-smi.

        Returns:
            GPU temperature in °C, or 0 if not available
        """
        if not self.gpu_available:
            return 0

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip())
        except Exception as e:
            self.logger.error(f"Error getting GPU temperature: {str(e)}")
            return 0

    def _get_gpu_power(self) -> float:
        """Get NVIDIA GPU power consumption in watts.

        Returns:
            GPU power consumption in watts, or 0 if not available
        """
        if not self.gpu_available:
            return 0

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            self.logger.error(f"Error getting GPU power: {str(e)}")
            return 0

    def _get_gpu_utilization(self) -> int:
        """Get NVIDIA GPU utilization percentage.

        Returns:
            GPU utilization percentage, or 0 if not available
        """
        if not self.gpu_available:
            return 0

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip())
        except Exception as e:
            self.logger.error(f"Error getting GPU utilization: {str(e)}")
            return 0

    def _get_cpu_temp(self) -> float:
        """Get CPU temperature using psutil.

        Returns:
            CPU temperature in °C, or 0 if not available
        """
        if not self.has_temp_sensors:
            return 0

        try:
            # If psutil cannot get temperatures, try sensors command
            temps = psutil.sensors_temperatures()
            if not temps:
                raise Exception("No temperature data from psutil")

            # Try common sensor names
            for name in ["coretemp", "k10temp", "cpu_thermal", "acpitz", "soc"]:
                if name in temps:
                    return max(sensor.current for sensor in temps[name])

            # If we cannot find a specific one, return max of all
            return max(sensor.current for sensors in temps.values() for sensor in sensors)
        except Exception as e:
            # Fall back to lm-sensors through subprocess
            try:
                result = subprocess.run(
                    ["sensors", "-j"],
                    capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    import json
                    data = json.loads(result.stdout)
                    # Parse the json output to find CPU temperature
                    max_temp = 0
                    for device, sensors in data.items():
                        if "cpu" in device.lower() or "core" in device.lower():
                            for sensor_name, values in sensors.items():
                                if "temp" in sensor_name.lower() or "tdie" in sensor_name.lower():
                                    if isinstance(values, dict) and "input" in values:
                                        max_temp = max(max_temp, values["input"])
                    return max_temp
            except Exception as nested_e:
                self.logger.error(f"Error getting CPU temperature: {e} -> {nested_e}")

            return 0

    def _get_cpu_utilization(self) -> float:
        """Get CPU utilization percentage.

        Returns:
            CPU utilization percentage
        """
        return psutil.cpu_percent(interval=0.1)

    def _set_gpu_power_limit(self, percent: int) -> bool:
        """Set NVIDIA GPU power limit to percentage of maximum.

        Args:
            percent: Percentage of maximum power limit

        Returns:
            True if successful, False otherwise
        """
        if not self.gpu_available:
            return False

        try:
            # First get the maximum power limit
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.max_limit", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            max_power = float(result.stdout.strip())

            # Calculate new power limit
            new_limit = int(max_power * percent / 100)

            # Set the new power limit
            subprocess.run(
                ["nvidia-smi", f"--power-limit={new_limit}"],
                check=True
            )
            self.logger.info(f"GPU power limit set to {new_limit}W ({percent}% of max)")
            return True
        except Exception as e:
            self.logger.error(f"Error setting GPU power limit: {str(e)}")
            return False

    def _reset_gpu_power_limit(self) -> bool:
        """Reset GPU power limit to default.

        Returns:
            True if successful, False otherwise
        """
        if not self.gpu_available:
            return False

        try:
            subprocess.run(
                ["nvidia-smi", "--reset-gpu-application-clocks"],
                check=True
            )
            subprocess.run(
                ["nvidia-smi", "--reset-gpu-power-limit"],
                check=True
            )
            self.logger.info("GPU power limit reset to default")
            self.throttling_active = False
            return True
        except Exception as e:
            self.logger.error(f"Error resetting GPU power limit: {str(e)}")
            return False

    def _apply_throttling(self, temp_gpu: float, temp_cpu: float) -> None:
        """Apply throttling based on temperature.

        Args:
            temp_gpu: GPU temperature in °C
            temp_cpu: CPU temperature in °C
        """
        message = ""

        # GPU temperature throttling
        if self.gpu_available and temp_gpu > self.gpu_temp_threshold:
            # Calculate dynamic power limit - reduce more as temperature increases
            overshoot = temp_gpu - self.gpu_temp_threshold
            # Reduce by 5% for each degree over threshold, minimum 50%
            reduction = max(50, self.power_limit_percent - (overshoot * 5))

            if self._set_gpu_power_limit(reduction):
                self.throttling_active = True
                message += f"WARNING: GPU temperature {temp_gpu}°C exceeds threshold {self.gpu_temp_threshold}°C. "
                message += f"Power limit reduced to {reduction}%.\n"

        # CPU throttling through psutil - we cannot directly control CPU power,
        # but we can reduce process priority
        if self.has_temp_sensors and temp_cpu > self.cpu_temp_threshold:
            try:
                p = psutil.Process(os.getpid())
                p.nice(10)  # Lower priority (higher nice value)
                message += f"WARNING: CPU temperature {temp_cpu}°C exceeds threshold {self.cpu_temp_threshold}°C. "
                message += f"Process priority reduced.\n"
            except Exception as e:
                message += f"Failed to reduce CPU priority: {e}\n"

        # If temperatures are back to normal and throttling was active, reset
        if self.throttling_active and temp_gpu < (self.gpu_temp_threshold - 5) and temp_cpu < (
                self.cpu_temp_threshold - 5):
            if self._reset_gpu_power_limit():
                message += "Temperature returned to safe range. Reset power limits.\n"

        # Log warning message if any
        if message:
            with open(self.warning_file, "a") as f:
                f.write(f"{datetime.now()}: {message}")
            self.logger.warning(message)

    def log_temperatures(self, force: bool = False) -> Optional[Tuple[float, float, float, float, float]]:
        """Log temperature data to file.

        Args:
            force: Whether to force logging regardless of interval

        Returns:
            Tuple of (gpu_temp, cpu_temp, gpu_power, gpu_util, cpu_util) if logged,
            None if skipped due to interval
        """
        current_time = time.time()

        # Only log at specified intervals unless forced
        if not force and (current_time - self.last_log_time) < self.log_interval:
            return None

        self.last_log_time = current_time

        # Get temperature and utilization data
        temp_gpu = self._get_gpu_temp()
        temp_cpu = self._get_cpu_temp()
        power_gpu = self._get_gpu_power()
        util_gpu = self._get_gpu_utilization()
        util_cpu = self._get_cpu_utilization()

        # Log to file
        with open(self.log_file, "a") as f:
            f.write(f"{datetime.now()},{temp_gpu},{temp_cpu},{power_gpu},{util_gpu},{util_cpu}\n")

        return temp_gpu, temp_cpu, power_gpu, util_gpu, util_cpu

    def monitor_loop(self) -> None:
        """Main monitoring loop."""
        # Set initial power limit
        if self.gpu_available:
            self.logger.info(f"Setting initial GPU power limit to {self.power_limit_percent}%")
            self._set_gpu_power_limit(self.power_limit_percent)

        self.logger.info(
            f"Temperature monitor started. "
            f"GPU threshold: {self.gpu_temp_threshold}°C, "
            f"CPU threshold: {self.cpu_temp_threshold}°C"
        )

        self.is_running = True
        while self.is_running:
            try:
                # Get and log temperatures
                temps = self.log_temperatures()

                if temps:
                    temp_gpu, temp_cpu, power_gpu, util_gpu, util_cpu = temps

                    # Print status
                    status_msg = ""
                    if self.gpu_available:
                        status_msg += f"GPU: {temp_gpu}°C ({util_gpu}% util, {power_gpu:.1f}W)"
                    if self.has_temp_sensors:
                        if status_msg:
                            status_msg += ", "
                        status_msg += f"CPU: {temp_cpu}°C ({util_cpu}% util)"

                    if status_msg:
                        self.logger.info(status_msg)

                    # Apply throttling if needed
                    self._apply_throttling(temp_gpu, temp_cpu)

            except Exception as e:
                self.logger.error(f"Error in temperature monitor: {str(e)}")

            # Wait for next check
            time.sleep(self.check_interval)

    def start(self, in_background: bool = True) -> Optional[threading.Thread]:
        """Start temperature monitoring.

        Args:
            in_background: Whether to run in background thread

        Returns:
            Thread object if running in background, None otherwise
        """
        if not self.gpu_available and not self.has_temp_sensors:
            self.logger.warning(
                "No GPU or temperature sensors detected. Temperature monitoring disabled."
            )
            return None

        if in_background:
            self.monitor_thread = threading.Thread(
                target=self.monitor_loop,
                daemon=True
            )
            self.monitor_thread.start()
            return self.monitor_thread
        else:
            self.monitor_loop()
            return None

    def stop(self) -> None:
        """Stop temperature monitoring."""
        self.is_running = False

        # Reset power limit if active
        if self.throttling_active:
            self._reset_gpu_power_limit()

        # Wait for thread to terminate
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            if self.monitor_thread.is_alive():
                self.logger.warning("Temperature monitor thread did not terminate properly")

        self.logger.info("Temperature monitoring stopped")


def setup_temperature_monitoring(gpu_threshold: int = 75,
                                 cpu_threshold: int = 85,
                                 power_limit: int = 70,
                                 log_dir: str = 'logs/temperature',
                                 in_background: bool = True,
                                 logger: Optional[logging.Logger] = None) -> Optional[TemperatureMonitor]:
    """Set up temperature monitoring with safe defaults.

    Args:
        gpu_threshold: GPU temperature threshold in °C
        cpu_threshold: CPU temperature threshold in °C
        power_limit: GPU power limit as percentage of maximum
        log_dir: Directory for log files
        in_background: Whether to run in background
        logger: Logger to use

    Returns:
        TemperatureMonitor instance if successful, None otherwise
    """
    try:
        # Create monitor instance
        monitor = TemperatureMonitor(
            gpu_temp_threshold=gpu_threshold,
            cpu_temp_threshold=cpu_threshold,
            power_limit_percent=power_limit,
            log_dir=log_dir,
            logger=logger or logging.getLogger('TemperatureMonitor')
        )

        # Start monitoring
        monitor.start(in_background)
        return monitor
    except Exception as e:
        if logger:
            logger.error(f"Failed to set up temperature monitoring: {str(e)}")
        else:
            print(f"Failed to set up temperature monitoring: {str(e)}")
        return None


def get_current_temperatures() -> Dict[str, float]:
    """Get current GPU and CPU temperatures.

    Returns:
        Dictionary with temperature information
    """
    # Create temporary monitor to get readings
    monitor = TemperatureMonitor(
        check_interval=1,
        log_interval=0,
        logger=logging.getLogger('TempCheck')
    )

    # Get temperature readings
    temp_gpu = monitor._get_gpu_temp()
    temp_cpu = monitor._get_cpu_temp()
    power_gpu = monitor._get_gpu_power()
    util_gpu = monitor._get_gpu_utilization()
    util_cpu = monitor._get_cpu_utilization()

    return {
        'gpu_temp': temp_gpu,
        'cpu_temp': temp_cpu,
        'gpu_power': power_gpu,
        'gpu_utilization': util_gpu,
        'cpu_utilization': util_cpu,
        'gpu_available': monitor.gpu_available,
        'temp_sensors_available': monitor.has_temp_sensors
    }