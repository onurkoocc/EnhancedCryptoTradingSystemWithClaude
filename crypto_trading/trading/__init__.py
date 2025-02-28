"""
Trading components for cryptocurrency trading system.

This package provides signal generation, risk management, and backtesting
functionality for cryptocurrency trading systems.
"""

from .signal_generator import EnhancedSignalProducer
from .risk_manager import AdvancedRiskManager
from .backtester import EnhancedStrategyBacktester

__all__ = [
    'EnhancedSignalProducer',
    'AdvancedRiskManager',
    'EnhancedStrategyBacktester'
]