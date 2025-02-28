"""
Configuration manager for loading, validating, and accessing configuration.

This module provides a configuration manager that handles loading configuration
from files and environment variables, validating it, and providing access to
configuration settings throughout the application.
"""

import json
import logging
import os
from dataclasses import asdict
from typing import Dict, Any, Optional, TypeVar

import yaml

from .default_config import (
    DataConfig,
    FeatureEngineeringConfig,
    ModelConfig,
    SignalConfig,
    RiskConfig,
    BacktestConfig,
    SystemConfig,
    get_default_config
)
from ..utils.path_manager import get_path_manager

# Type variable for generic dataclass updates
T = TypeVar('T')


class ConfigurationManager:
    """Manages loading and validating configuration."""

    def __init__(self, config_path: Optional[str] = None,
                 env_prefix: str = 'CRYPTO_',
                 logger: Optional[logging.Logger] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to config file (YAML or JSON)
            env_prefix: Prefix for environment variables to override config
            logger: Logger to use for logging messages
        """
        self.config_path = config_path
        self.env_prefix = env_prefix
        self.logger = logger or logging.getLogger('ConfigManager')
        self.config = get_default_config()
        self.path_manager = get_path_manager()

    # Add this new method to the ConfigurationManager class:
    def _update_paths(self):
        """Update configuration paths to use path manager."""
        # Update data paths
        self.config.data.csv_30m = str(self.path_manager.get_data_file("btc", "30m"))
        self.config.data.csv_4h = str(self.path_manager.get_data_file("btc", "4h"))
        self.config.data.csv_daily = str(self.path_manager.get_data_file("btc", "daily"))
        self.config.data.csv_oi = str(self.path_manager.get_path('data_market') / "btc_open_interest.csv")
        self.config.data.csv_funding = str(self.path_manager.get_path('data_market') / "btc_funding_rates.csv")
        self.config.data.cache_directory = str(self.path_manager.get_path('data_cache'))

        # Update log paths
        self.config.log_dir = str(self.path_manager.get_path('logs'))

        # Update model paths
        model_dir = self.path_manager.get_model_dir(self.config.model.project_name)
        model_file = f"best_{self.config.model.project_name}.keras"
        self.config.model.model_save_path = str(model_dir / model_file)

        # Update backtest paths
        self.config.backtest.results_directory = str(self.path_manager.get_path('results_backtest'))

    # Modify the load_config method to call _update_paths:
    def load_config(self) -> SystemConfig:
        """Load configuration from file and environment variables.

        Returns:
            SystemConfig: Configured system configuration
        """
        # Start with default configuration
        config_dict = {}

        # Load from file if provided
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        config_dict = yaml.safe_load(f)
                    elif self.config_path.endswith('.json'):
                        config_dict = json.load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {self.config_path}")

                self.logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                self.logger.error(f"Error loading configuration from {self.config_path}: {e}")
                self.logger.info("Using default configuration")

        # Override with environment variables
        self._override_from_env(config_dict)

        # Update configuration objects
        self._update_config_from_dict(config_dict)

        # Update paths to use path manager
        self._update_paths()

        # Validate configuration
        self._validate_config()

        return self.config

    def _override_from_env(self, config_dict: Dict[str, Any]) -> None:
        """Override configuration from environment variables.

        Args:
            config_dict: Configuration dictionary to update
        """
        # Example: CRYPTO_DATA_CSV_30M -> config_dict['data']['csv_30m']
        for env_var, value in os.environ.items():
            if env_var.startswith(self.env_prefix):
                parts = env_var[len(self.env_prefix):].lower().split('_')
                if len(parts) >= 2:
                    section = parts[0]
                    key = '_'.join(parts[1:])

                    if section not in config_dict:
                        config_dict[section] = {}

                    # Convert value to appropriate type
                    if value.lower() in ('true', 'yes', '1'):
                        config_dict[section][key] = True
                    elif value.lower() in ('false', 'no', '0'):
                        config_dict[section][key] = False
                    elif value.isdigit():
                        config_dict[section][key] = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        config_dict[section][key] = float(value)
                    else:
                        # Try to parse as JSON for lists or dicts
                        try:
                            config_dict[section][key] = json.loads(value)
                        except json.JSONDecodeError:
                            config_dict[section][key] = value

    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration objects from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        # Update component configs
        section_mapping = {
            'data': (self.config.data, DataConfig),
            'features': (self.config.features, FeatureEngineeringConfig),
            'model': (self.config.model, ModelConfig),
            'signal': (self.config.signal, SignalConfig),
            'risk': (self.config.risk, RiskConfig),
            'backtest': (self.config.backtest, BacktestConfig)
        }

        for section_name, (section_obj, section_class) in section_mapping.items():
            if section_name in config_dict:
                self._update_dataclass(section_obj, config_dict[section_name])

        # Update top-level config
        for key, value in config_dict.items():
            if key not in section_mapping and hasattr(self.config, key):
                setattr(self.config, key, value)

    def _update_dataclass(self, dataclass_obj: T, data_dict: Dict[str, Any]) -> None:
        """Update dataclass object from dictionary.

        Args:
            dataclass_obj: Dataclass object to update
            data_dict: Dictionary with values
        """
        for key, value in data_dict.items():
            if hasattr(dataclass_obj, key):
                try:
                    # Get current value type
                    current_value = getattr(dataclass_obj, key)
                    # If the current value is a list but the new value is a scalar,
                    # try to convert the new value to a list
                    if isinstance(current_value, list) and not isinstance(value, list):
                        try:
                            value = [value]
                        except (ValueError, TypeError):
                            self.logger.warning(
                                f"Cannot convert {value} to list for attribute {key}"
                            )
                            continue

                    setattr(dataclass_obj, key, value)
                except (ValueError, TypeError) as e:
                    self.logger.warning(
                        f"Failed to set {key}={value} on {type(dataclass_obj).__name__}: {e}"
                    )

    def _validate_config(self) -> None:
        """Validate configuration settings."""
        # Validate data config
        if self.config.data.lookback_30m_candles < 100:
            self.logger.warning("lookback_30m_candles is very small, setting to minimum of 100")
            self.config.data.lookback_30m_candles = 100

        # Validate model config
        if self.config.model.sequence_length <= 0:
            self.logger.error("sequence_length must be positive, setting to default of 144")
            self.config.model.sequence_length = 144

        if self.config.model.horizon <= 0:
            self.logger.error("horizon must be positive, setting to default of 48")
            self.config.model.horizon = 48

        # Validate risk config
        if self.config.risk.max_risk_per_trade <= 0 or self.config.risk.max_risk_per_trade > 0.1:
            self.logger.warning(
                f"max_risk_per_trade of {self.config.risk.max_risk_per_trade} is outside "
                "recommended range (0-0.1), setting to 0.02"
            )
            self.config.risk.max_risk_per_trade = 0.02

    def save_config(self, output_path: str) -> None:
        """Save current configuration to file.

        Args:
            output_path: Path to save configuration
        """
        # Create config dictionary from dataclasses
        config_dict = {
            'data': asdict(self.config.data),
            'features': asdict(self.config.features),
            'model': asdict(self.config.model),
            'signal': asdict(self.config.signal),
            'risk': asdict(self.config.risk),
            'backtest': asdict(self.config.backtest),
            'log_level': self.config.log_level,
            'log_dir': self.config.log_dir,
            'mode': self.config.mode,
            'gpu_memory_limit': self.config.gpu_memory_limit,
            'monitor_memory': self.config.monitor_memory,
            'monitor_temperature': self.config.monitor_temperature,
            'memory_threshold_gb': self.config.memory_threshold_gb,
            'env': self.config.env
        }

        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save to file
        with open(output_path, 'w') as f:
            if output_path.endswith('.yaml') or output_path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False)
            elif output_path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {output_path}")

        self.logger.info(f"Configuration saved to {output_path}")

    def get_env_docs(self) -> str:
        """Generate documentation for environment variable overrides.

        Returns:
            str: Documentation string listing all possible environment variables
        """
        docs = [
            "Environment Variable Configuration",
            "=================================",
            "",
            "The following environment variables can be used to override configuration settings:",
            ""
        ]

        def process_section(section_name: str, section_obj: object) -> None:
            for key, value in asdict(section_obj).items():
                env_var = f"{self.env_prefix}{section_name.upper()}_{key.upper()}"
                type_info = type(value).__name__
                default = str(value)
                if isinstance(value, list) and value:
                    element_type = type(value[0]).__name__
                    type_info = f"list of {element_type}"
                    default = str(value)

                docs.append(f"* {env_var}")
                docs.append(f"  - Type: {type_info}")
                docs.append(f"  - Default: {default}")
                docs.append("")

        # Process each section
        process_section('data', self.config.data)
        process_section('features', self.config.features)
        process_section('model', self.config.model)
        process_section('signal', self.config.signal)
        process_section('risk', self.config.risk)
        process_section('backtest', self.config.backtest)

        # Process top-level config
        for key, value in {
            'log_level': self.config.log_level,
            'log_dir': self.config.log_dir,
            'mode': self.config.mode,
            'gpu_memory_limit': self.config.gpu_memory_limit,
            'monitor_memory': self.config.monitor_memory,
            'monitor_temperature': self.config.monitor_temperature,
            'memory_threshold_gb': self.config.memory_threshold_gb,
            'env': self.config.env
        }.items():
            env_var = f"{self.env_prefix}{key.upper()}"
            type_info = type(value).__name__

            docs.append(f"* {env_var}")
            docs.append(f"  - Type: {type_info}")
            docs.append(f"  - Default: {value}")
            docs.append("")

        return "\n".join(docs)