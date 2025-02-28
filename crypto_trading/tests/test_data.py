"""
Unit tests for the data module.

This module provides tests for data loading, preprocessing, and
feature engineering components.
"""

import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock

from crypto_trading.data.binance_client import BitcoinData
from crypto_trading.data.data_processor import CryptoDataPreparer
from crypto_trading.data.feature_engineering import EnhancedCryptoFeatureEngineer


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    # Create a basic DataFrame with OHLCV data
    index = pd.date_range('2023-01-01', periods=100, freq='30min')
    data = {
        'open': np.random.uniform(20000, 30000, 100),
        'high': np.random.uniform(20000, 30000, 100),
        'low': np.random.uniform(20000, 30000, 100),
        'close': np.random.uniform(20000, 30000, 100),
        'volume': np.random.uniform(1, 100, 100),
        'turnover': np.random.uniform(20000, 100000, 100)
    }

    # Ensure high >= open, close, low and low <= open, close
    for i in range(100):
        data['high'][i] = max(data['open'][i], data['close'][i], data['high'][i])
        data['low'][i] = min(data['open'][i], data['close'][i], data['low'][i])

    df = pd.DataFrame(data, index=index)
    return df


class TestBitcoinData:
    """Tests for BitcoinData class."""

    @patch('crypto_trading.data.binance_client.UMFutures')
    def test_initialization(self, mock_umfutures):
        """Test BitcoinData initialization."""
        # Setup mock
        mock_umfutures.return_value = MagicMock()

        # Initialize BitcoinData
        client = BitcoinData(
            csv_30m='test_30m.csv',
            csv_4h='test_4h.csv',
            csv_daily='test_daily.csv',
            csv_oi='test_oi.csv',
            csv_funding='test_funding.csv'
        )

        # Assert client was initialized
        assert client is not None
        assert client.symbol == 'BTCUSDT'
        assert mock_umfutures.called

    def test_derive_4h_data(self, sample_ohlcv_data):
        """Test 4h data derivation from 30m data."""
        # Setup
        client = BitcoinData(
            csv_30m='test_30m.csv',
            csv_4h='test_4h.csv',
            csv_daily='test_daily.csv'
        )

        # Override UMFutures client
        client.client = MagicMock()

        # Derive 4h data
        df_4h = client.derive_4h_data(sample_ohlcv_data)

        # Assert
        assert df_4h is not None
        assert len(df_4h) < len(sample_ohlcv_data)
        assert all(col in df_4h.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_derive_daily_data(self, sample_ohlcv_data):
        """Test daily data derivation from 30m data."""
        # Setup
        client = BitcoinData(
            csv_30m='test_30m.csv',
            csv_4h='test_4h.csv',
            csv_daily='test_daily.csv'
        )

        # Override UMFutures client
        client.client = MagicMock()

        # Derive daily data
        df_daily = client.derive_daily_data(sample_ohlcv_data)

        # Assert
        assert df_daily is not None
        assert len(df_daily) < len(sample_ohlcv_data)
        assert all(col in df_daily.columns for col in ['open', 'high', 'low', 'close', 'volume'])


class TestCryptoDataPreparer:
    """Tests for CryptoDataPreparer class."""

    def test_initialization(self):
        """Test CryptoDataPreparer initialization."""
        preparer = CryptoDataPreparer(
            sequence_length=144,
            horizon=48,
            normalize_method='zscore'
        )

        assert preparer is not None
        assert preparer.sequence_length == 144
        assert preparer.horizon == 48
        assert preparer.normalize_method == 'zscore'

    def test_create_labels(self, sample_ohlcv_data):
        """Test label creation for training."""
        # Setup
        preparer = CryptoDataPreparer(
            sequence_length=10,
            horizon=5,
            normalize_method='zscore'
        )

        # Create labels
        df_labeled, labels, fwd_returns = preparer._create_labels(sample_ohlcv_data)

        # Assert
        assert df_labeled is not None
        assert len(df_labeled) == len(sample_ohlcv_data) - preparer.horizon
        assert len(labels) == len(sample_ohlcv_data) - preparer.horizon
        assert all(0 <= label <= 4 for label in labels)
        assert len(fwd_returns) == len(sample_ohlcv_data) - preparer.horizon

    def test_build_sequences(self, sample_ohlcv_data):
        """Test sequence building for model input."""
        # Setup
        preparer = CryptoDataPreparer(
            sequence_length=10,
            horizon=5,
            normalize_method='zscore'
        )

        # Create labels
        df_labeled, labels, fwd_returns = preparer._create_labels(sample_ohlcv_data)

        # Convert to numpy array
        data_array = df_labeled.values.astype(np.float32)

        # Build sequences
        X, y, seq_fwd_returns = preparer._build_sequences(data_array, labels, fwd_returns)

        # Assert
        assert X is not None
        assert y is not None
        assert seq_fwd_returns is not None

        # Check shapes
        assert X.shape[1] == preparer.sequence_length  # Check sequence length
        assert X.shape[2] == data_array.shape[1]  # Check feature dimension
        assert y.shape[1] == 5  # Check one-hot encoding (5 classes)
        assert len(seq_fwd_returns) == len(X)  # Check forward returns length


class TestEnhancedCryptoFeatureEngineer:
    """Tests for EnhancedCryptoFeatureEngineer class."""

    def test_initialization(self):
        """Test EnhancedCryptoFeatureEngineer initialization."""
        engineer = EnhancedCryptoFeatureEngineer(
            feature_scaling=True
        )

        assert engineer is not None
        assert engineer.feature_scaling is True

    def test_compute_indicators_30m(self, sample_ohlcv_data):
        """Test 30-minute indicators computation."""
        # Setup
        engineer = EnhancedCryptoFeatureEngineer(
            feature_scaling=False
        )

        # Compute indicators
        indicators = engineer._compute_indicators_30m(sample_ohlcv_data)

        # Assert
        assert indicators is not None
        assert len(indicators) == len(sample_ohlcv_data)

        # Check if basic indicators are calculated
        assert 'BB_middle' in indicators.columns
        assert 'BB_upper' in indicators.columns
        assert 'BB_lower' in indicators.columns
        assert 'hist_vol_20' in indicators.columns

    def test_compute_indicators_4h(self, sample_ohlcv_data):
        """Test 4-hour indicators computation."""
        # Setup
        engineer = EnhancedCryptoFeatureEngineer(
            feature_scaling=False
        )

        # Compute indicators
        indicators = engineer._compute_indicators_4h(sample_ohlcv_data)

        # Assert
        assert indicators is not None
        assert len(indicators) == len(sample_ohlcv_data)

        # Check if basic indicators are calculated
        assert any(col.startswith('SMA_') for col in indicators.columns)
        assert any(col.startswith('EMA_') for col in indicators.columns)
        assert any(col.startswith('RSI_') for col in indicators.columns)

    def test_compute_market_regime(self, sample_ohlcv_data):
        """Test market regime detection."""
        # Setup
        engineer = EnhancedCryptoFeatureEngineer(
            feature_scaling=False
        )

        # Add some required columns for market regime calculation
        sample_ohlcv_data['h4_ADX'] = np.random.uniform(10, 40, len(sample_ohlcv_data))
        sample_ohlcv_data['h4_SMA_20'] = np.cumsum(np.random.uniform(-100, 100, len(sample_ohlcv_data)))
        sample_ohlcv_data['h4_SMA_50'] = np.cumsum(np.random.uniform(-50, 50, len(sample_ohlcv_data)))

        # Compute market regime
        regime = engineer._compute_market_regime(sample_ohlcv_data)

        # Assert
        assert regime is not None
        assert len(regime) == len(sample_ohlcv_data)
        assert all(value in [-1, 0, 1] for value in np.unique(regime))


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])