"""
Unit tests for the trading module.

This module provides tests for the signal generation, risk management,
and backtesting components.
"""

import pytest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from crypto_trading.trading.signal_generator import EnhancedSignalProducer
from crypto_trading.trading.risk_manager import AdvancedRiskManager
from crypto_trading.trading.backtester import EnhancedStrategyBacktester


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    # Create a basic DataFrame with OHLCV data and indicators
    index = pd.date_range('2023-01-01', periods=100, freq='30min')
    data = {
        'open': np.random.uniform(20000, 30000, 100),
        'high': np.random.uniform(20000, 30000, 100),
        'low': np.random.uniform(20000, 30000, 100),
        'close': np.random.uniform(20000, 30000, 100),
        'volume': np.random.uniform(1, 100, 100),
        'h4_SMA_20': np.random.uniform(20000, 30000, 100),
        'h4_SMA_50': np.random.uniform(20000, 30000, 100),
        'h4_SMA_200': np.random.uniform(20000, 30000, 100),
        'h4_ADX': np.random.uniform(10, 40, 100),
        'd1_RSI_14': np.random.uniform(20, 80, 100),
        'h4_RSI_14': np.random.uniform(20, 80, 100),
        'd1_ATR_14': np.random.uniform(100, 1000, 100),
        'market_regime': np.random.choice([-1, 0, 1], 100),
        'volatility_regime': np.random.choice([-1, 0, 1], 100),
        'trend_strength': np.random.uniform(-1, 1, 100),
        'hist_vol_20': np.random.uniform(0.2, 0.8, 100)
    }

    # Ensure high >= open, close, low and low <= open, close
    for i in range(100):
        data['high'][i] = max(data['open'][i], data['close'][i], data['high'][i])
        data['low'][i] = min(data['open'][i], data['close'][i], data['low'][i])

    df = pd.DataFrame(data, index=index)
    return df


@pytest.fixture
def model_predictions():
    """Create sample model predictions for testing."""
    # 5-class prediction probabilities
    return np.array([0.05, 0.10, 0.10, 0.25, 0.50])  # Strong bullish signal


class TestEnhancedSignalProducer:
    """Tests for EnhancedSignalProducer class."""

    def test_initialization(self):
        """Test EnhancedSignalProducer initialization."""
        signal_producer = EnhancedSignalProducer(
            confidence_threshold=0.4,
            strong_signal_threshold=0.7,
            atr_multiplier_sl=1.5
        )

        assert signal_producer is not None
        assert signal_producer.confidence_threshold == 0.4
        assert signal_producer.strong_signal_threshold == 0.7
        assert signal_producer.atr_multiplier_sl == 1.5

    def test_get_signal_bullish(self, sample_market_data, model_predictions):
        """Test signal generation for bullish prediction."""
        # Setup
        signal_producer = EnhancedSignalProducer(
            confidence_threshold=0.4,
            strong_signal_threshold=0.7
        )

        # Get signal
        signal = signal_producer.get_signal(
            model_probs=model_predictions,
            df=sample_market_data
        )

        # Assert
        assert signal is not None
        assert 'signal_type' in signal
        assert 'Buy' in signal['signal_type'] or 'NoTrade' in signal['signal_type']

        # If it's a buy signal, it should have a stop loss
        if 'Buy' in signal['signal_type']:
            assert 'stop_loss' in signal
            assert signal['stop_loss'] < sample_market_data['close'].iloc[-1]

    def test_get_signal_bearish(self, sample_market_data):
        """Test signal generation for bearish prediction."""
        # Setup
        signal_producer = EnhancedSignalProducer(
            confidence_threshold=0.4,
            strong_signal_threshold=0.7
        )

        # Create bearish prediction (first two classes)
        bearish_prediction = np.array([0.40, 0.45, 0.05, 0.05, 0.05])

        # Get signal
        signal = signal_producer.get_signal(
            model_probs=bearish_prediction,
            df=sample_market_data
        )

        # Assert
        assert signal is not None
        assert 'signal_type' in signal
        assert 'Sell' in signal['signal_type'] or 'NoTrade' in signal['signal_type']

        # If it's a sell signal, it should have a stop loss
        if 'Sell' in signal['signal_type']:
            assert 'stop_loss' in signal
            assert signal['stop_loss'] > sample_market_data['close'].iloc[-1]

    def test_compute_atr(self, sample_market_data):
        """Test ATR computation."""
        # Setup
        signal_producer = EnhancedSignalProducer()

        # Compute ATR
        atr = signal_producer._compute_atr(sample_market_data)

        # Assert
        assert atr is not None
        assert len(atr) == len(sample_market_data)
        assert not atr.isna().all()

    def test_should_close_position(self, sample_market_data, model_predictions):
        """Test position closing logic."""
        # Setup
        signal_producer = EnhancedSignalProducer()

        # Create a test position
        current_price = sample_market_data['close'].iloc[-1]
        position = {
            'direction': 'long',
            'entry_price': current_price * 0.95,  # 5% below current price (in profit)
            'stop_loss': current_price * 0.90,
            'take_profit': current_price * 1.10
        }

        # Check if we should close
        close_decision = signal_producer.should_close_position(
            position=position,
            current_data=sample_market_data,
            model_probs=model_predictions
        )

        # Assert
        assert close_decision is not None
        assert 'should_close' in close_decision

        # Repeat with a position at stop loss
        position['stop_loss'] = current_price * 1.01  # Just above current price
        close_decision = signal_producer.should_close_position(
            position=position,
            current_data=sample_market_data,
            model_probs=model_predictions
        )

        # Should close if stop loss hit
        assert close_decision.get('should_close', False)
        assert close_decision.get('reason') == 'StopLoss'


class TestAdvancedRiskManager:
    """Tests for AdvancedRiskManager class."""

    def test_initialization(self):
        """Test AdvancedRiskManager initialization."""
        risk_manager = AdvancedRiskManager(
            initial_capital=10000.0,
            max_risk_per_trade=0.02,
            max_correlated_exposure=0.06
        )

        assert risk_manager is not None
        assert risk_manager.initial_capital == 10000.0
        assert risk_manager.current_capital == 10000.0
        assert risk_manager.max_risk_per_trade == 0.02
        assert risk_manager.max_correlated_exposure == 0.06

    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Setup
        risk_manager = AdvancedRiskManager(
            initial_capital=10000.0,
            max_risk_per_trade=0.02
        )

        # Calculate position size
        signal = {
            'confidence': 0.8,
            'signal_type': 'StrongBuy'
        }
        entry_price = 25000.0
        stop_loss = 24000.0  # $1000 distance to stop loss

        position_size = risk_manager.calculate_position_size(
            signal=signal,
            entry_price=entry_price,
            stop_loss=stop_loss
        )

        # Assert
        assert position_size > 0

        # Risk amount should be approximately 2% of capital
        risk_amount = position_size * abs(entry_price - stop_loss)
        assert 190 < risk_amount < 210  # Should be around $200 (2% of $10,000)

    def test_check_correlation_risk(self):
        """Test correlation risk check."""
        # Setup
        risk_manager = AdvancedRiskManager(
            initial_capital=10000.0,
            max_risk_per_trade=0.02,
            max_correlated_exposure=0.06
        )

        # Check with no existing positions
        allowed, adjusted_risk = risk_manager.check_correlation_risk({
            'direction': 'long',
            'risk_amount': 200.0
        })

        # Assert
        assert allowed is True
        assert adjusted_risk > 0

        # Add a position and check again
        risk_manager.open_positions.append({
            'direction': 'long',
            'risk_amount': 400.0
        })

        allowed, adjusted_risk = risk_manager.check_correlation_risk({
            'direction': 'long',
            'risk_amount': 300.0
        })

        # Should still be allowed but with potentially reduced risk
        assert allowed is True

        # Add more positions to exceed limit
        risk_manager.open_positions.append({
            'direction': 'long',
            'risk_amount': 300.0
        })

        allowed, adjusted_risk = risk_manager.check_correlation_risk({
            'direction': 'long',
            'risk_amount': 300.0
        })

        # Should not be allowed
        assert allowed is False

    def test_add_position(self):
        """Test adding a position."""
        # Setup
        risk_manager = AdvancedRiskManager(
            initial_capital=10000.0
        )

        # Add position
        position = {
            'entry_price': 25000.0,
            'stop_loss': 24000.0,
            'quantity': 0.01,
            'direction': 'long'
        }

        position_id = risk_manager.add_position(position)

        # Assert
        assert position_id > 0
        assert len(risk_manager.open_positions) == 1
        assert risk_manager.open_positions[0]['id'] == position_id

    def test_close_position(self):
        """Test closing a position."""
        # Setup
        risk_manager = AdvancedRiskManager(
            initial_capital=10000.0
        )

        # Add position
        position = {
            'entry_price': 25000.0,
            'stop_loss': 24000.0,
            'quantity': 0.01,
            'direction': 'long',
            'entry_time': datetime.now()
        }

        position_id = risk_manager.add_position(position)

        # Close position with profit
        exit_price = 26000.0  # $1000 profit
        closed_position = risk_manager.close_position(
            position_id=position_id,
            exit_price=exit_price,
            exit_time=datetime.now() + timedelta(hours=1),
            exit_reason="Test"
        )

        # Assert
        assert closed_position is not None
        assert 'pnl' in closed_position
        assert closed_position['pnl'] > 0
        assert len(risk_manager.open_positions) == 0
        assert len(risk_manager.trade_history) == 1

        # Check capital update
        assert risk_manager.current_capital > 10000.0


class TestEnhancedStrategyBacktester:
    """Tests for EnhancedStrategyBacktester class."""

    def test_initialization(self, sample_market_data):
        """Test EnhancedStrategyBacktester initialization."""
        # Create mock objects
        preparer = MagicMock()
        modeler = MagicMock()
        signal_producer = MagicMock()
        risk_manager = MagicMock()

        # Initialize backtester
        backtester = EnhancedStrategyBacktester(
            data_df=sample_market_data,
            preparer=preparer,
            modeler=modeler,
            signal_producer=signal_producer,
            risk_manager=risk_manager,
            train_window_size=50,
            test_window_size=20
        )

        assert backtester is not None
        assert backtester.data_df is sample_market_data
        assert backtester.train_window_size == 50
        assert backtester.test_window_size == 20

    def test_detect_regime(self, sample_market_data):
        """Test market regime detection."""
        # Create mock objects
        preparer = MagicMock()
        modeler = MagicMock()
        signal_producer = MagicMock()
        risk_manager = MagicMock()

        # Initialize backtester
        backtester = EnhancedStrategyBacktester(
            data_df=sample_market_data,
            preparer=preparer,
            modeler=modeler,
            signal_producer=signal_producer,
            risk_manager=risk_manager
        )

        # Detect regime
        regime = backtester._detect_regime(sample_market_data)

        # Assert
        assert regime is not None
        assert regime in ['trending', 'ranging', 'volatile', 'unknown']

    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create mock objects
        preparer = MagicMock()
        modeler = MagicMock()
        signal_producer = MagicMock()
        risk_manager = MagicMock()

        # Initialize backtester
        backtester = EnhancedStrategyBacktester(
            data_df=pd.DataFrame(),
            preparer=preparer,
            modeler=modeler,
            signal_producer=signal_producer,
            risk_manager=risk_manager
        )

        # Create sample trades
        trades = [
            {
                'entry_time': '2023-01-01 00:00:00',
                'exit_time': '2023-01-01 12:00:00',
                'direction': 'long',
                'entry_price': 25000.0,
                'exit_price': 26000.0,
                'quantity': 0.1,
                'pnl': 100.0,
                'pnl_percent': 4.0,
                'entry_signal': 'Buy',
                'exit_reason': 'TakeProfit'
            },
            {
                'entry_time': '2023-01-02 00:00:00',
                'exit_time': '2023-01-02 12:00:00',
                'direction': 'short',
                'entry_price': 26000.0,
                'exit_price': 25500.0,
                'quantity': 0.1,
                'pnl': 50.0,
                'pnl_percent': 1.9,
                'entry_signal': 'Sell',
                'exit_reason': 'TakeProfit'
            },
            {
                'entry_time': '2023-01-03 00:00:00',
                'exit_time': '2023-01-03 12:00:00',
                'direction': 'long',
                'entry_price': 25000.0,
                'exit_price': 24000.0,
                'quantity': 0.1,
                'pnl': -100.0,
                'pnl_percent': -4.0,
                'entry_signal': 'Buy',
                'exit_reason': 'StopLoss'
            }
        ]

        # Calculate metrics
        metrics = backtester._calculate_performance_metrics(trades, 10050.0)

        # Assert
        assert metrics is not None
        assert 'total_trades' in metrics
        assert metrics['total_trades'] == 3
        assert 'win_rate' in metrics
        assert metrics['win_rate'] == 2 / 3
        assert 'profit_factor' in metrics
        assert metrics['profit_factor'] > 1


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])