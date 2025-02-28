"""
Configuration management module for the crypto trading system.

This package provides tools for loading, validating, and accessing
configuration settings throughout the application.
"""

from .config_manager import ConfigurationManager
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

__all__ = [
    'ConfigurationManager',
    'DataConfig',
    'FeatureEngineeringConfig',
    'ModelConfig',
    'SignalConfig',
    'RiskConfig',
    'BacktestConfig',
    'SystemConfig',
    'get_default_config'
]