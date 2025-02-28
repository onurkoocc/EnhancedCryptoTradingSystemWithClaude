"""
Custom callbacks for cryptocurrency trading models.

This module provides custom callbacks for training cryptocurrency
trading models, including risk-adjusted metrics and performance monitoring.
"""

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import Callback
from typing import Dict, List, Tuple, Optional, Union, Any
import gc

from ..utils.memory_monitor import log_memory_usage, clear_memory


class RiskAdjustedTradeMetric(Callback):
    """Callback to calculate risk-adjusted trading metrics during training.

    This callback simulates trading using model predictions to calculate
    risk-adjusted performance metrics.
    """

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, fwd_returns_val: np.ndarray,
                 df_val: pd.DataFrame, initial_balance: float = 10000,
                 kelly_fraction: float = 0.5, reward_risk_ratio: float = 2.5,
                 partial_close_ratio: float = 0.5, atr_period: int = 14,
                 atr_multiplier_sl: float = 1.5, logger: Optional[logging.Logger] = None):
        """Initialize the callback.

        Args:
            X_val: Validation features
            y_val: Validation labels
            fwd_returns_val: Validation forward returns
            df_val: Validation DataFrame
            initial_balance: Initial balance for simulated trading
            kelly_fraction: Fraction of Kelly criterion to use
            reward_risk_ratio: Reward/risk ratio for position sizing
            partial_close_ratio: Ratio for partial position closes
            atr_period: Period for ATR calculation
            atr_multiplier_sl: Multiplier for ATR to set stop loss
            logger: Logger to use
        """
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.fwd_returns_val = fwd_returns_val
        self.df_val = df_val
        self.initial_balance = initial_balance
        self.kelly_fraction = kelly_fraction
        self.reward_risk_ratio = reward_risk_ratio
        self.partial_close_ratio = partial_close_ratio
        self.atr_period = atr_period
        self.atr_multiplier_sl = atr_multiplier_sl
        self.logger = logger or logging.getLogger('RiskAdjTradeMetric')

        # Keep history of performance
        self.performance_history = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Calculate metrics at the end of each epoch.

        Args:
            epoch: Current epoch
            logs: Logs dictionary to update
        """
        logs = logs or {}

        try:
            # Get predictions
            y_pred_probs = self.model.predict(self.X_val, verbose=0)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)

            # Simulate trading
            final_balance, trades, metrics = self._simulate_trading(y_pred_probs, y_pred_classes)

            # Update logs with metrics
            logs['val_final_balance'] = final_balance
            logs['val_roi_percent'] = (final_balance / self.initial_balance - 1) * 100
            logs['val_trade_count'] = metrics['trade_count']
            logs['val_win_rate'] = metrics['win_rate']
            logs['val_profit_factor'] = metrics['profit_factor']
            logs['val_avg_risk_adj_return'] = metrics['avg_trade']
            logs['val_max_drawdown'] = metrics['max_drawdown']
            logs['val_sharpe_ratio'] = metrics['sharpe_ratio']

            # Store performance for this epoch
            self.performance_history.append({
                'epoch': epoch + 1,
                'final_balance': final_balance,
                'roi_percent': logs['val_roi_percent'],
                'trade_count': metrics['trade_count'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'avg_trade': metrics['avg_trade'],
                'max_drawdown': metrics['max_drawdown'],
                'sharpe_ratio': metrics['sharpe_ratio']
            })

            # Log results
            self.logger.info(
                f"Epoch {epoch + 1}: Balance={final_balance:.2f}, "
                f"ROI={logs['val_roi_percent']:.2f}%, "
                f"Win Rate={metrics['win_rate']:.2f}, "
                f"Trades={metrics['trade_count']}"
            )

        except Exception as e:
            self.logger.error(f"Error in RiskAdjustedTradeMetric: {str(e)}")
            # Provide default values to avoid training interruption
            logs['val_final_balance'] = self.initial_balance
            logs['val_roi_percent'] = 0.0
            logs['val_avg_risk_adj_return'] = 0.0

    def _simulate_trading(self, y_pred_probs: np.ndarray, y_pred_classes: np.ndarray) -> Tuple[float, List[Dict], Dict]:
        """Simulate trading based on model predictions.

        Args:
            y_pred_probs: Predicted probabilities
            y_pred_classes: Predicted class indices

        Returns:
            Tuple of (final balance, trades list, metrics dictionary)
        """
        current_balance = self.initial_balance
        trades = []
        trade_returns = []
        equity_curve = [current_balance]

        for i in range(len(y_pred_classes)):
            pred_class = y_pred_classes[i]
            confidence = y_pred_probs[i][pred_class]
            actual_return = self.fwd_returns_val[i]

            # Determine trade direction
            if pred_class in [3, 4]:  # Bullish classes
                direction = 'long'
            elif pred_class in [0, 1]:  # Bearish classes
                direction = 'short'
            else:
                # No trade for neutral class
                trade_returns.append(0)
                equity_curve.append(current_balance)
                continue

            # Calculate ATR for position sizing
            atr = self._compute_atr(self.df_val, self.atr_period).iloc[i]
            if np.isnan(atr) or atr <= 0:
                # Skip if ATR is invalid
                trade_returns.append(0)
                equity_curve.append(current_balance)
                continue

            # Calculate risk distance based on ATR
            distance = self.atr_multiplier_sl * atr

            # Calculate Kelly position size
            # b = reward/risk ratio
            # p = probability of winning (use confidence)
            # f = (b*p - q) / b where q = 1-p
            b = self.reward_risk_ratio
            p = confidence
            q = 1 - p

            # Only trade if expected value is positive
            if b * p <= q:
                trade_returns.append(0)
                equity_curve.append(current_balance)
                continue

            # Calculate Kelly fraction
            f = min(max((b * p - q) / b, 0), 1) * self.kelly_fraction

            # Calculate position size
            risk_amount = current_balance * f
            entry_price = self.df_val['close'].iloc[i]
            quantity = risk_amount / distance

            # Set stop loss and take profit
            if direction == 'long':
                stop_loss = entry_price - distance
                take_profit = entry_price + (self.reward_risk_ratio * distance)

                # Estimate exit price based on actual return
                if actual_return >= 0:
                    # Winning trade - exit at take profit or actual return (whichever is less)
                    exit_price = min(entry_price * (1 + actual_return), take_profit)
                else:
                    # Losing trade - exit at stop loss or actual return (whichever is more)
                    exit_price = max(entry_price * (1 + actual_return), stop_loss)
            else:
                stop_loss = entry_price + distance
                take_profit = entry_price - (self.reward_risk_ratio * distance)

                # Estimate exit price based on actual return
                if actual_return <= 0:
                    # Winning trade - exit at take profit or actual return (whichever is more negative)
                    exit_price = max(entry_price * (1 + actual_return), take_profit)
                else:
                    # Losing trade - exit at stop loss or actual return (whichever is less)
                    exit_price = min(entry_price * (1 + actual_return), stop_loss)

            # Calculate PnL
            if direction == 'long':
                pnl = quantity * (exit_price - entry_price)
            else:
                pnl = quantity * (entry_price - exit_price)

            # Update balance
            current_balance += pnl

            # Record trade
            trades.append({
                'index': i,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'confidence': confidence,
                'predicted_class': pred_class,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pnl': pnl
            })

            trade_returns.append(pnl)
            equity_curve.append(current_balance)

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(trades, trade_returns, equity_curve)

        return current_balance, trades, metrics

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range.

        Args:
            df: DataFrame with OHLCV data
            period: Period for ATR calculation

        Returns:
            Series with ATR values
        """
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift(1)).abs()
        low_close_prev = (df['low'] - df['close'].shift(1)).abs()

        # True Range is the maximum of the three
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # ATR is the moving average of True Range
        return tr.rolling(window=period).mean()

    def _calculate_performance_metrics(self, trades: List[Dict], trade_returns: List[float],
                                       equity_curve: List[float]) -> Dict[str, float]:
        """Calculate performance metrics from trades.

        Args:
            trades: List of trade dictionaries
            trade_returns: List of trade returns
            equity_curve: List of account equity values

        Returns:
            Dictionary with performance metrics
        """
        # Basic metrics
        if len(trades) > 0:
            win_trades = [t for t in trades if t['pnl'] > 0]
            loss_trades = [t for t in trades if t['pnl'] <= 0]

            win_rate = len(win_trades) / len(trades) if trades else 0
            avg_win = np.mean([t['pnl'] for t in win_trades]) if win_trades else 0
            avg_loss = np.mean([t['pnl'] for t in loss_trades]) if loss_trades else 0

            total_profit = sum(t['pnl'] for t in win_trades) if win_trades else 0
            total_loss = abs(sum(t['pnl'] for t in loss_trades)) if loss_trades else 1e-10
            profit_factor = total_profit / total_loss

            avg_trade = np.mean(trade_returns) if trade_returns else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade = 0

        # Calculate drawdown
        max_equity = 0
        drawdowns = []

        for equity in equity_curve:
            if equity > max_equity:
                max_equity = equity

            # Calculate drawdown percentage
            if max_equity > 0:
                drawdown = (max_equity - equity) / max_equity
                drawdowns.append(drawdown)
            else:
                drawdowns.append(0)

        max_drawdown = max(drawdowns) if drawdowns else 0

        # Calculate Sharpe ratio
        if len(equity_curve) > 1:
            returns = []
            for i in range(1, len(equity_curve)):
                ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                returns.append(ret)

            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 1e-10
            sharpe_ratio = avg_return / std_return * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0

        return {
            'trade_count': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }


class SaveBestModelCallback(Callback):
    """Callback to save the best model based on a specified metric.

    This callback provides more flexibility than ModelCheckpoint by
    allowing complex metrics and custom saving logic.
    """

    def __init__(self, filepath: str, monitor: str = 'val_avg_risk_adj_return',
                 save_weights_only: bool = False, mode: str = 'max',
                 save_best_only: bool = True, verbose: int = 1,
                 logger: Optional[logging.Logger] = None):
        """Initialize the callback.

        Args:
            filepath: Path to save the model
            monitor: Metric to monitor
            save_weights_only: Whether to save only weights
            mode: 'min' or 'max'
            save_best_only: Whether to save only if improved
            verbose: Verbosity level
            logger: Logger to use
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.logger = logger or logging.getLogger('SaveBestModel')

        # Initialize best value
        if mode == 'min':
            self.best = float('inf')
            self.monitor_op = lambda x, y: x < y
        else:
            self.best = float('-inf')
            self.monitor_op = lambda x, y: x > y

        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Check and save model at the end of each epoch.

        Args:
            epoch: Current epoch
            logs: Logs dictionary
        """
        logs = logs or {}

        # Get current monitored value
        if self.monitor in logs:
            current = logs[self.monitor]

            # Check if improved
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    self.logger.info(
                        f"Epoch {epoch + 1}: {self.monitor} improved from "
                        f"{self.best:.6f} to {current:.6f}, saving model to {self.filepath}"
                    )

                # Save the model
                if self.save_weights_only:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)

                # Update best value
                self.best = current
                self.best_epoch = epoch + 1
            else:
                if self.verbose > 0:
                    self.logger.info(
                        f"Epoch {epoch + 1}: {self.monitor} did not improve from {self.best:.6f}"
                    )
        else:
            if self.verbose > 0:
                self.logger.warning(f"Metric '{self.monitor}' not found in logs!")


class MemoryCheckpoint(Callback):
    """Callback to monitor memory usage during training.

    This callback logs memory usage during training and performs cleanup if needed.
    """

    def __init__(self, threshold_gb: float = 14, log_interval: int = 1,
                 cleanup_threshold_gb: float = 16, logger: Optional[logging.Logger] = None):
        """Initialize the callback.

        Args:
            threshold_gb: Memory threshold for logging in GB
            log_interval: Interval (in epochs) for logging memory usage
            cleanup_threshold_gb: Memory threshold for cleanup in GB
            logger: Logger to use
        """
        super().__init__()
        self.threshold_gb = threshold_gb
        self.log_interval = log_interval
        self.cleanup_threshold_gb = cleanup_threshold_gb
        self.logger = logger or logging.getLogger('MemoryCheckpoint')

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Log memory usage at the beginning of each epoch.

        Args:
            epoch: Current epoch
            logs: Logs dictionary
        """
        if epoch % self.log_interval == 0:
            memory_gb = log_memory_usage()

            if memory_gb > self.threshold_gb:
                self.logger.warning(
                    f"Memory usage at epoch start: {memory_gb:.2f}GB "
                    f"(threshold: {self.threshold_gb}GB)"
                )
            else:
                self.logger.info(f"Memory usage at epoch start: {memory_gb:.2f}GB")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Check memory usage at the end of each epoch.

        Args:
            epoch: Current epoch
            logs: Logs dictionary
        """
        if epoch % self.log_interval == 0:
            memory_gb = log_memory_usage()

            if memory_gb > self.cleanup_threshold_gb:
                self.logger.warning(
                    f"Memory usage ({memory_gb:.2f}GB) exceeded cleanup threshold "
                    f"({self.cleanup_threshold_gb}GB). Performing cleanup..."
                )
                clear_memory()

                # Log memory after cleanup
                memory_gb = log_memory_usage()
                self.logger.info(f"Memory after cleanup: {memory_gb:.2f}GB")
            elif memory_gb > self.threshold_gb:
                self.logger.warning(
                    f"Memory usage at epoch end: {memory_gb:.2f}GB "
                    f"(threshold: {self.threshold_gb}GB)"
                )
            else:
                self.logger.info(f"Memory usage at epoch end: {memory_gb:.2f}GB")

    def on_train_end(self, logs: Optional[Dict[str, float]] = None):
        """Log memory usage at the end of training.

        Args:
            logs: Logs dictionary
        """
        memory_gb = log_memory_usage()
        self.logger.info(f"Memory usage at end of training: {memory_gb:.2f}GB")

        # Perform cleanup
        clear_memory()
        memory_gb = log_memory_usage()
        self.logger.info(f"Memory after final cleanup: {memory_gb:.2f}GB")