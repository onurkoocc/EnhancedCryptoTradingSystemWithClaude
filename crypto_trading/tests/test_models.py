"""
Unit tests for the models module.

This module provides tests for model creation, training, and inference.
"""

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from crypto_trading.models.callbacks import RiskAdjustedTradeMetric
from crypto_trading.models.crypto_model import EnhancedCryptoModel
from crypto_trading.models.metrics import TradingMetrics


@pytest.fixture
def sample_model_data():
    """Create sample data for model testing."""
    # Create sample sequential data
    sequence_length = 10
    features = 5
    num_samples = 100

    # Create X (input sequences)
    X = np.random.randn(num_samples, sequence_length, features).astype(np.float32)

    # Create y (one-hot encoded labels for 5 classes)
    y_indices = np.random.randint(0, 5, num_samples)
    y = np.zeros((num_samples, 5))
    for i, idx in enumerate(y_indices):
        y[i, idx] = 1

    # Create forward returns
    fwd_returns = np.random.uniform(-0.05, 0.05, num_samples)

    # Create validation DataFrame
    dates = pd.date_range('2023-01-01', periods=num_samples)
    df_val = pd.DataFrame({
        'close': np.random.uniform(20000, 30000, num_samples),
        'open': np.random.uniform(20000, 30000, num_samples),
        'high': np.random.uniform(20000, 30000, num_samples),
        'low': np.random.uniform(20000, 30000, num_samples),
        'volume': np.random.uniform(1, 100, num_samples)
    }, index=dates)

    return X, y, fwd_returns, df_val


class TestEnhancedCryptoModel:
    """Tests for EnhancedCryptoModel class."""

    def test_initialization(self):
        """Test model initialization."""
        model = EnhancedCryptoModel(
            project_name="test_model",
            max_trials=10,
            ensemble_size=3,
            use_mixed_precision=False
        )

        assert model is not None
        assert model.project_name == "test_model"
        assert model.max_trials == 10
        assert model.ensemble_size == 3

    def test_build_model(self):
        """Test model building functionality."""

        # Create a mock hyperparameters object
        class MockHP:
            def Int(self, name, min_value, max_value, step=None):
                return 64

            def Float(self, name, min_value, max_value, step=None, sampling=None):
                return 0.1

            def Choice(self, name, values):
                return values[0]

            def Boolean(self, name):
                return True

        # Initialize model
        tf_model = EnhancedCryptoModel(
            project_name="test_model",
            use_mixed_precision=False
        )

        # Build model
        hp = MockHP()
        input_shape = (10, 5)  # sequence_length, features
        model = tf_model._build_model(hp, input_shape, total_steps=1000)

        # Assert
        assert model is not None
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape[1:] == input_shape
        assert model.output_shape[1] == 5  # 5 classes

    def test_transformer_block(self):
        """Test transformer block implementation."""
        # Initialize model
        tf_model = EnhancedCryptoModel(
            project_name="test_model",
            use_mixed_precision=False
        )

        # Create input tensor
        x = tf.random.normal((32, 10, 64))  # batch_size, sequence_length, features

        # Apply transformer block
        output = tf_model._transformer_block(
            x=x,
            units=64,
            num_heads=4,
            dropout_rate=0.1,
            use_layer_norm=True
        )

        # Assert
        assert output is not None
        assert output.shape == x.shape  # Transformer maintains shape

    def test_predict_signals(self, sample_model_data):
        """Test model prediction function."""
        X, _, _, _ = sample_model_data

        # Create a mock model for prediction
        input_shape = (X.shape[1], X.shape[2])
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.LSTM(32, return_sequences=False)(inputs)
        outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
        mock_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Initialize model
        crypto_model = EnhancedCryptoModel(
            project_name="test_model",
            use_mixed_precision=False
        )

        # Set the mock model as the best model
        crypto_model.best_model = mock_model

        # Make predictions
        predictions = crypto_model.predict_signals(X)

        # Assert
        assert predictions is not None
        assert predictions.shape == (X.shape[0], 5)  # Batch size, 5 classes
        assert np.allclose(np.sum(predictions, axis=1), 1.0)  # Softmax sums to 1


class TestTradingMetrics:
    """Tests for TradingMetrics class."""

    def test_initialization(self):
        """Test TradingMetrics initialization."""
        metrics = TradingMetrics(
            num_classes=5,
            classes_to_monitor=[0, 1, 2, 3, 4]
        )

        assert metrics is not None

    def test_get_metrics(self):
        """Test getting metrics from TradingMetrics."""
        metrics = TradingMetrics(
            num_classes=5,
            classes_to_monitor=[0, 1, 2, 3, 4]
        )

        metrics_list = metrics.get_metrics()

        # Assert
        assert metrics_list is not None
        assert len(metrics_list) > 0

        # Check if we have precision, recall, and F1 for each class
        class_metrics = ['precision_class', 'recall_class', 'f1_class']
        for cls_id in range(5):
            for metric in class_metrics:
                found = False
                for m in metrics_list:
                    if hasattr(m, '__name__') and f"{metric}_{cls_id}" == m.__name__:
                        found = True
                        break
                assert found, f"Missing {metric}_{cls_id} in metrics list"


class TestRiskAdjustedTradeMetric:
    """Tests for RiskAdjustedTradeMetric callback."""

    def test_initialization(self, sample_model_data):
        """Test RiskAdjustedTradeMetric initialization."""
        X, y, fwd_returns, df_val = sample_model_data

        # Initialize callback
        callback = RiskAdjustedTradeMetric(
            X_val=X,
            y_val=y,
            fwd_returns_val=fwd_returns,
            df_val=df_val
        )

        assert callback is not None

    def test_calculate_performance_metrics(self, sample_model_data):
        """Test performance metrics calculation."""
        _, _, _, df_val = sample_model_data

        # Initialize callback
        callback = RiskAdjustedTradeMetric(
            X_val=np.zeros((10, 10, 5)),
            y_val=np.zeros((10, 5)),
            fwd_returns_val=np.zeros(10),
            df_val=df_val
        )

        # Create sample trades
        trades = [
            {'pnl': 100.0},
            {'pnl': -50.0},
            {'pnl': 75.0},
            {'pnl': 25.0},
            {'pnl': -25.0}
        ]

        trade_returns = [t['pnl'] for t in trades]
        equity_curve = [10000]
        for r in trade_returns:
            equity_curve.append(equity_curve[-1] + r)

        # Calculate metrics
        metrics = callback._calculate_performance_metrics(trades, trade_returns, equity_curve)

        # Assert
        assert metrics is not None
        assert 'trade_count' in metrics
        assert metrics['trade_count'] == 5
        assert 'win_rate' in metrics
        assert metrics['win_rate'] == 0.6  # 3 wins out of 5
        assert 'profit_factor' in metrics
        assert metrics['profit_factor'] > 0  # Should be 200/75 = 2.67


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])