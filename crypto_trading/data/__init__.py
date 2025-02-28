"""
Data handling module for the crypto trading system.

This package provides components for fetching, processing, and engineering
features from cryptocurrency market data.
"""

from .binance_client import BitcoinData
from .data_processor import CryptoDataPreparer
from .feature_engineering import EnhancedCryptoFeatureEngineer

__all__ = [
    'BitcoinData',
    'CryptoDataPreparer',
    'EnhancedCryptoFeatureEngineer'
]
