"""
Backtesting framework for cryptocurrency trading.

This module provides a walk-forward backtesting framework for
evaluating cryptocurrency trading strategies.
"""

import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import json
import concurrent.futures
import gc
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import linregress

from ..utils.logging_utils import exception_handler, log_execution_time
from ..utils.memory_monitor import memory_usage_decorator, log_memory_usage, clear_memory


class EnhancedStrategyBacktester:
    """Walk-forward backtesting framework for cryptocurrency trading strategies."""

    def __init__(self, data_df: pd.DataFrame, preparer: Any,
                 modeler: Any, signal_producer: Any, risk_manager: Any,
                 train_window_size: int = 5000, test_window_size: int = 1000,
                 fixed_cost: float = 0.001, variable_cost: float = 0.0005,
                 slippage: float = 0.0005, walk_forward_steps: int = 4,
                 monte_carlo_sims: int = 100, use_parallel: bool = True,
                 max_workers: Optional[int] = None, results_dir: str = 'results',
                 logger: Optional[logging.Logger] = None):
        """Initialize backtester.

        Args:
            data_df: DataFrame with feature data
            preparer: Data preparer instance
            modeler: Model instance
            signal_producer: Signal producer instance
            risk_manager: Risk manager instance
            train_window_size: Size of training window
            test_window_size: Size of testing window
            fixed_cost: Fixed cost per trade
            variable_cost: Variable cost as percentage of trade value
            slippage: Slippage as percentage of price
            walk_forward_steps: Number of steps to move window
            monte_carlo_sims: Number of Monte Carlo simulations
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of worker processes
            results_dir: Directory for results
            logger: Logger to use
        """
        self.data_df = data_df
        self.preparer = preparer
        self.modeler = modeler
        self.signal_producer = signal_producer
        self.risk_manager = risk_manager
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.fixed_cost = fixed_cost
        self.variable_cost = variable_cost
        self.slippage = slippage
        self.walk_forward_steps = walk_forward_steps
        self.monte_carlo_sims = monte_carlo_sims
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        self.results_dir = results_dir
        self.logger = logger or logging.getLogger('Backtester')

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Storage for results
        self.results = []

    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the backtester.

        Returns:
            Configured logger
        """
        logger = logging.getLogger("EnhancedBacktester")
        logger.setLevel(logging.INFO)

        # Create handlers
        log_path = os.path.join(self.results_dir, f"backtest_log_{datetime.now():%Y%m%d_%H%M%S}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)

        return logger

    @exception_handler(reraise=True)
    @log_execution_time()
    @memory_usage_decorator(threshold_gb=14)
    def walk_forward_backtest(self) -> pd.DataFrame:
        """Perform walk-forward backtesting.

        Returns:
            DataFrame with backtest results
        """
        start_idx = 0
        df_len = len(self.data_df)
        iteration = 0

        # Ensure results directory exists
        trades_dir = os.path.join(self.results_dir, "trades")
        os.makedirs(trades_dir, exist_ok=True)

        # Create CSV file for trades
        trades_path = os.path.join(trades_dir, f"trades_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(trades_path, 'w') as f:
            f.write("iteration,entry_time,exit_time,direction,entry_price,exit_price,quantity,"
                    "pnl,pnl_percent,entry_signal,exit_reason,regime,stop_loss,take_profit\n")

        # Storage for results
        all_results = []
        performance_by_iteration = []

        # Calculate step size for window movement
        step_size = self.test_window_size // 2  # 50% overlap by default

        # Use the walk_forward_steps parameter if it's not 0
        if self.walk_forward_steps > 0:
            step_size = self.test_window_size // self.walk_forward_steps

        # Ensure step size is at least 1
        step_size = max(1, step_size)

        self.logger.info(f"Starting walk-forward backtest with step size {step_size}")

        # Run walk-forward backtest
        if self.use_parallel:
            # Prepare batches for parallel processing
            batches = []

            while start_idx + self.train_window_size + self.test_window_size <= df_len:
                iteration += 1
                train_end = start_idx + self.train_window_size
                test_end = train_end + self.test_window_size

                batches.append({
                    'iteration': iteration,
                    'start_idx': start_idx,
                    'train_end': train_end,
                    'test_end': test_end
                })

                start_idx += step_size

            # Process batches in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                batch_results = list(executor.map(self._process_batch, batches))

            # Combine results
            for result, trades, performance in batch_results:
                # Save trades to CSV
                self._save_trades_to_file(trades, trades_path)

                # Add result to list
                all_results.append(result)
                performance_by_iteration.append(performance)
        else:
            # Process sequentially
            while start_idx + self.train_window_size + self.test_window_size <= df_len:
                iteration += 1
                self.logger.info(f"Starting iteration {iteration} of walk-forward backtest")

                # Log memory at start of each iteration
                log_memory_usage()

                train_end = start_idx + self.train_window_size
                test_end = train_end + self.test_window_size

                # Get training and testing data
                df_train = self.data_df.iloc[start_idx:train_end].copy()
                df_test = self.data_df.iloc[train_end:test_end].copy()

                # Detect market regime in this period
                regime = self._detect_regime(df_train)
                self.logger.info(f"Detected market regime: {regime}")

                # Process iteration
                result, trades, performance = self._process_iteration(
                    iteration, df_train, df_test, regime
                )

                # Save trades to CSV
                self._save_trades_to_file(trades, trades_path)

                # Add result to lists
                all_results.append(result)
                performance_by_iteration.append(performance)

                # Save checkpoint
                self._save_checkpoint(iteration, result)

                # Force memory cleanup
                clear_memory()

                # Move to next window
                start_idx += step_size

        # Analyze performance across different regimes
        self._analyze_period_performance(performance_by_iteration)

        # Run Monte Carlo simulation
        self._run_monte_carlo_analysis(trades_path)

        # Save overall results
        self._save_overall_results(all_results, performance_by_iteration)

        # Return results DataFrame
        return pd.DataFrame(all_results)

    def _process_batch(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """Process a batch for parallel execution.

        Args:
            batch: Dictionary with batch parameters

        Returns:
            Tuple of (result dictionary, trades list, performance dictionary)
        """
        iteration = batch['iteration']
        start_idx = batch['start_idx']
        train_end = batch['train_end']
        test_end = batch['test_end']

        # Get training and testing data
        df_train = self.data_df.iloc[start_idx:train_end].copy()
        df_test = self.data_df.iloc[train_end:test_end].copy()

        # Detect market regime in this period
        regime = self._detect_regime(df_train)

        # Process iteration
        return self._process_iteration(iteration, df_train, df_test, regime)

    def _process_iteration(self, iteration: int, df_train: pd.DataFrame,
                           df_test: pd.DataFrame, regime: str) -> Tuple[
        Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """Process a single iteration of the walk-forward test.

        Args:
            iteration: Iteration number
            df_train: Training data
            df_test: Testing data
            regime: Market regime

        Returns:
            Tuple of (result dictionary, trades list, performance dictionary)
        """
        # Prepare data
        X_train, y_train, X_val, y_val, df_val, fwd_returns_val = self.preparer.prepare_data(df_train)

        if len(X_train) == 0:
            self.logger.warning(f"Insufficient training data in iteration {iteration}")
            return {
                "iteration": iteration,
                "train_start": df_train.index[0],
                "train_end": df_train.index[-1],
                "test_end": df_test.index[-1],
                "final_equity": self.risk_manager.initial_capital,
                "regime": regime
            }, [], {
                "iteration": iteration,
                "train_start": df_train.index[0],
                "train_end": df_train.index[-1],
                "test_end": df_test.index[-1],
                "final_equity": self.risk_manager.initial_capital,
                "regime": regime,
                "metrics": {}
            }

        # Compute class weights with emphasis on extreme classes
        y_train_flat = np.argmax(y_train, axis=1)
        class_weights = self._compute_class_weights(y_train_flat, regime)

        # Train model or ensemble
        if hasattr(self.modeler, 'build_ensemble'):
            self.logger.info(f"Training ensemble models for iteration {iteration}")
            self.modeler.build_ensemble(
                X_train, y_train, X_val, y_val, df_val, fwd_returns_val,
                epochs=32, batch_size=256, class_weight=class_weights
            )
        else:
            self.logger.info(f"Training single model for iteration {iteration}")
            self.modeler.tune_and_train(
                iteration, X_train, y_train, X_val, y_val, df_val, fwd_returns_val,
                epochs=32, batch_size=256, class_weight=class_weights
            )

        # Force memory cleanup after training
        clear_memory()

        # Evaluate model performance
        if len(X_val) > 0:
            self.logger.info(f"Evaluating model for iteration {iteration}")
            self.modeler.evaluate(X_val, y_val)

        # Clean up to save memory
        del X_train, y_train, X_val, y_val, df_val, fwd_returns_val
        gc.collect()

        # Backtest on test period
        self.logger.info(f"Simulating trading for iteration {iteration}")
        test_equity, test_trades = self._simulate_test(df_test, iteration, regime)

        # Calculate performance metrics
        perf_metrics = self._calculate_performance_metrics(test_trades, test_equity)

        # Create result dictionary
        result = {
            "iteration": iteration,
            "train_start": df_train.index[0],
            "train_end": df_train.index[-1],
            "test_end": df_test.index[-1],
            "final_equity": test_equity,
            "regime": regime
        }

        # Create performance dictionary
        performance = {
            "iteration": iteration,
            "train_start": df_train.index[0],
            "train_end": df_train.index[-1],
            "test_end": df_test.index[-1],
            "final_equity": test_equity,
            "regime": regime,
            "metrics": perf_metrics
        }

        return result, test_trades, performance

    def _compute_class_weights(self, y_train_flat: np.ndarray, regime: str) -> Dict[int, float]:
        """Compute class weights with adjustments for market regime.

        Args:
            y_train_flat: Flattened training labels
            regime: Market regime

        Returns:
            Dictionary with class weights
        """
        from sklearn.utils import compute_class_weight

        # Compute balanced class weights
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train_flat), y=y_train_flat
        )

        # Convert to dictionary
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        # Adjust weights based on regime
        if regime == "trending":
            # In trending regimes, emphasize strong directional moves
            class_weight_dict[0] *= 1.75  # Strongly bearish
            class_weight_dict[4] *= 1.75  # Strongly bullish
        elif regime == "ranging":
            # In ranging regimes, emphasize mean reversion
            class_weight_dict[1] *= 1.5  # Moderately bearish
            class_weight_dict[3] *= 1.5  # Moderately bullish
        elif regime == "volatile":
            # In volatile regimes, all signals matter
            for i in range(5):
                class_weight_dict[i] *= 1.0

        return class_weight_dict

    def _save_trades_to_file(self, trades: List[Dict[str, Any]], filepath: str) -> None:
        """Save trades to CSV file.

        Args:
            trades: List of trade dictionaries
            filepath: Path to save trades
        """
        if not trades:
            return

        # Append trades to file
        with open(filepath, 'a') as f:
            for trade in trades:
                # Format trade details for CSV
                line = (
                    f"{trade.get('iteration', 0)},"
                    f"{trade.get('entry_time', '')},"
                    f"{trade.get('exit_time', '')},"
                    f"{trade.get('direction', '')},"
                    f"{trade.get('entry_price', 0)},"
                    f"{trade.get('exit_price', 0)},"
                    f"{trade.get('quantity', 0)},"
                    f"{trade.get('pnl', 0)},"
                    f"{trade.get('pnl_percent', 0)},"
                    f"{trade.get('entry_signal', '').replace(',', ';')},"
                    f"{trade.get('exit_reason', '').replace(',', ';')},"
                    f"{trade.get('regime', '').replace(',', ';')},"
                    f"{trade.get('stop_loss', 0)},"
                    f"{trade.get('take_profit', 0)}\n"
                )
                f.write(line)

        # Log trade count
        self.logger.info(f"Saved {len(trades)} trades to {filepath}")

    def _save_checkpoint(self, iteration: int, result: Dict[str, Any]) -> None:
        """Save checkpoint for an iteration.

        Args:
            iteration: Iteration number
            result: Result dictionary
        """
        checkpoint_path = os.path.join(self.results_dir, f"checkpoint_iter_{iteration}.json")

        with open(checkpoint_path, 'w') as f:
            # Convert dates to strings for JSON serialization
            json_result = {}
            for key, value in result.items():
                if isinstance(value, pd.Timestamp):
                    json_result[key] = value.isoformat()
                else:
                    json_result[key] = value

            json.dump(json_result, f, indent=2)

    def _save_overall_results(self, all_results: List[Dict[str, Any]],
                              performance_by_iteration: List[Dict[str, Any]]) -> None:
        """Save overall backtest results.

        Args:
            all_results: List of result dictionaries
            performance_by_iteration: List of performance dictionaries
        """
        # Save summary results
        summary_path = os.path.join(self.results_dir, f"results_summary_{datetime.now():%Y%m%d_%H%M%S}.json")

        with open(summary_path, 'w') as f:
            # Convert all results to JSON serializable format
            json_results = []

            for result in all_results:
                json_result = {}
                for key, value in result.items():
                    if isinstance(value, pd.Timestamp):
                        json_result[key] = value.isoformat()
                    else:
                        json_result[key] = value

                json_results.append(json_result)

            json.dump(json_results, f, indent=2)

        # Save performance metrics
        perf_path = os.path.join(self.results_dir, f"performance_metrics_{datetime.now():%Y%m%d_%H%M%S}.json")

        with open(perf_path, 'w') as f:
            # Extract key metrics and convert to JSON serializable format
            json_performance = []

            for perf in performance_by_iteration:
                json_perf = {
                    'iteration': perf['iteration'],
                    'regime': perf['regime']
                }

                # Add timestamps
                for key in ['train_start', 'train_end', 'test_end']:
                    if key in perf and isinstance(perf[key], pd.Timestamp):
                        json_perf[key] = perf[key].isoformat()
                    else:
                        json_perf[key] = str(perf.get(key, ''))

                # Add metrics
                metrics = perf.get('metrics', {})
                for key, value in metrics.items():
                    # Handle non-serializable types
                    if isinstance(value, (np.int64, np.float64)):
                        json_perf[key] = float(value)
                    elif isinstance(value, (datetime, pd.Timestamp)):
                        json_perf[key] = value.isoformat()
                    else:
                        json_perf[key] = value

                json_performance.append(json_perf)

            json.dump(json_performance, f, indent=2)

        # Create summary plots
        self._create_summary_plots(all_results, performance_by_iteration)

    def _create_summary_plots(self, all_results: List[Dict[str, Any]],
                              performance_by_iteration: List[Dict[str, Any]]) -> None:
        """Create summary plots for backtest results.

        Args:
            all_results: List of result dictionaries
            performance_by_iteration: List of performance dictionaries
        """
        # Create plots directory
        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        try:
            # 1. Equity curve
            plt.figure(figsize=(12, 6))

            # Extract equity values
            initial_capital = self.risk_manager.initial_capital
            equity_values = [initial_capital]

            for result in all_results:
                equity_values.append(result['final_equity'])

            # Plot equity curve
            plt.plot(range(len(equity_values)), equity_values)
            plt.title('Equity Curve')
            plt.xlabel('Iteration')
            plt.ylabel('Equity')
            plt.grid(True)

            # Save plot
            plt.savefig(os.path.join(plots_dir, 'equity_curve.png'))
            plt.close()

            # 2. Performance by regime
            regimes = set(p['regime'] for p in performance_by_iteration)

            plt.figure(figsize=(12, 6))

            for regime in regimes:
                # Extract equity values for this regime
                regime_equity = [initial_capital]

                for p in performance_by_iteration:
                    if p['regime'] == regime:
                        regime_equity.append(p['final_equity'])

                if len(regime_equity) > 1:
                    plt.plot(range(len(regime_equity)), regime_equity, label=regime)

            plt.title('Equity Curve by Market Regime')
            plt.xlabel('Iteration')
            plt.ylabel('Equity')
            plt.legend()
            plt.grid(True)

            # Save plot
            plt.savefig(os.path.join(plots_dir, 'equity_by_regime.png'))
            plt.close()

            # 3. Win rate by regime
            win_rates = []
            regime_labels = []

            for regime in regimes:
                regime_metrics = [p['metrics'] for p in performance_by_iteration if p['regime'] == regime]

                if regime_metrics:
                    avg_win_rate = np.mean([m.get('win_rate', 0) for m in regime_metrics])
                    win_rates.append(avg_win_rate)
                    regime_labels.append(regime)

            if win_rates:
                plt.figure(figsize=(10, 6))
                plt.bar(regime_labels, win_rates)
                plt.title('Average Win Rate by Market Regime')
                plt.xlabel('Market Regime')
                plt.ylabel('Win Rate')
                plt.ylim(0, 1)

                # Add percentage labels
                for i, v in enumerate(win_rates):
                    plt.text(i, v + 0.02, f"{v:.1%}", ha='center')

                # Save plot
                plt.savefig(os.path.join(plots_dir, 'win_rate_by_regime.png'))
                plt.close()

        except Exception as e:
            self.logger.error(f"Error creating summary plots: {str(e)}")

    def _detect_regime(self, df: pd.DataFrame) -> str:
        """Detect market regime in the given dataframe.

        Args:
            df: DataFrame with market data

        Returns:
            Market regime string
        """
        if len(df) < 100:
            return "unknown"

        # Check if we have regime columns from feature engineering
        if 'market_regime' in df.columns and 'volatility_regime' in df.columns:
            # Use the pre-calculated regime features
            market_regime = df['market_regime'].iloc[-20:].mean()
            volatility_regime = df['volatility_regime'].iloc[-20:].mean()

            if volatility_regime > 0.5:
                return "volatile"
            elif abs(market_regime) > 0.5:
                return "trending"
            else:
                return "ranging"

        # Otherwise, calculate regime based on price action
        close = df['close'].values

        # Calculate directional movement
        returns = np.diff(close) / close[:-1]

        # Volatility
        volatility = np.std(returns[-50:]) * np.sqrt(252)  # Annualized

        # Trend - use linear regression slope
        x = np.arange(min(50, len(close)))
        y = close[-min(50, len(close)):]

        slope, _, r_value, _, _ = linregress(x, y)

        trend_strength = abs(r_value)
        normalized_slope = slope / close[-1] * 100  # Normalize by current price

        # Determine regime
        if volatility > 0.8:  # High volatility threshold
            return "volatile"
        elif trend_strength > 0.7 and abs(normalized_slope) > 0.1:
            return "trending"
        else:
            return "ranging"

    def _calculate_performance_metrics(self, trades: List[Dict[str, Any]],
                                       final_equity: float) -> Dict[str, Any]:
        """Calculate performance metrics from trades.

        Args:
            trades: List of trade dictionaries
            final_equity: Final equity value

        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_trade': 0,
                'return': 0
            }

        # Basic metrics
        win_trades = [t for t in trades if t['pnl'] > 0]
        loss_trades = [t for t in trades if t['pnl'] <= 0]

        total_trades = len(trades)
        win_rate = len(win_trades) / total_trades if total_trades > 0 else 0

        total_profit = sum(t['pnl'] for t in win_trades)
        total_loss = abs(sum(t['pnl'] for t in loss_trades)) if loss_trades else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Build equity curve
        initial_capital = self.risk_manager.initial_capital
        equity_curve = [initial_capital]

        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['pnl'])

        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            returns.append(ret)

        # Sharpe ratio
        sharpe = 0
        if len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0

        # Maximum drawdown
        peak = equity_curve[0]
        drawdowns = []

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            drawdowns.append(dd)

        max_drawdown = max(drawdowns)

        # Average trade and return
        avg_trade = sum(t['pnl'] for t in trades) / len(trades) if trades else 0
        total_return = (final_equity - initial_capital) / initial_capital * 100

        # Additional metrics
        avg_win = sum(t['pnl'] for t in win_trades) / len(win_trades) if win_trades else 0
        avg_loss = sum(t['pnl'] for t in loss_trades) / len(loss_trades) if loss_trades else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Calculate average holding time
        holding_times = []
        for trade in trades:
            try:
                entry = pd.to_datetime(trade['entry_time'])
                exit = pd.to_datetime(trade['exit_time'])
                holding_time = (exit - entry).total_seconds() / 3600  # in hours
                holding_times.append(holding_time)
            except (TypeError, ValueError):
                continue

        avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'return': total_return,
            'avg_holding_time': avg_holding_time
        }

    def _analyze_period_performance(self, performance_by_iteration: List[Dict[str, Any]]) -> Dict[
        str, Dict[str, float]]:
        """Analyze performance across different regimes.

        Args:
            performance_by_iteration: List of performance dictionaries

        Returns:
            Dictionary with performance metrics by regime
        """
        if not performance_by_iteration:
            return {}

        # Group by regime
        regime_groups = {}
        for perf in performance_by_iteration:
            regime = perf['regime']
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(perf)

        # Calculate average metrics by regime
        regime_metrics = {}

        for regime, performances in regime_groups.items():
            # Calculate average metrics
            avg_metrics = {}
            for metric in performances[0]['metrics']:
                values = [p['metrics'][metric] for p in performances if metric in p['metrics']]
                if values:
                    avg_metrics[metric] = sum(values) / len(values)

            regime_metrics[regime] = avg_metrics

        # Log the analysis
        self.logger.info("\n=== Regime Performance Analysis ===")
        for regime, metrics in regime_metrics.items():
            self.logger.info(f"\nRegime: {regime}")
            for metric, value in metrics.items():
                self.logger.info(f"Avg {metric}: {value}")

        # Save regime analysis to file
        analysis_path = os.path.join(self.results_dir, f"regime_analysis_{datetime.now():%Y%m%d_%H%M%S}.json")

        with open(analysis_path, 'w') as f:
            # Convert regime metrics to JSON serializable format
            json_metrics = {}

            for regime, metrics in regime_metrics.items():
                json_metrics[regime] = {}
                for metric, value in metrics.items():
                    if isinstance(value, (np.int64, np.float64)):
                        json_metrics[regime][metric] = float(value)
                    else:
                        json_metrics[regime][metric] = value

            json.dump(json_metrics, f, indent=2)

        return regime_metrics

    def _run_monte_carlo_analysis(self, trades_path: str) -> Dict[str, Any]:
        """Run Monte Carlo simulations to assess strategy robustness.

        Args:
            trades_path: Path to trades CSV file

        Returns:
            Dictionary with Monte Carlo results
        """
        try:
            # Load trades
            trades_df = pd.read_csv(trades_path)

            if trades_df.empty:
                self.logger.warning("No trades found for Monte Carlo analysis")
                return {}

            # Extract PnL from trades
            pnl_values = trades_df['pnl'].values
            initial_capital = self.risk_manager.initial_capital

            # Arrays to store results
            final_equities = np.zeros(self.monte_carlo_sims)
            max_drawdowns = np.zeros(self.monte_carlo_sims)
            sharpe_ratios = np.zeros(self.monte_carlo_sims)

            for i in range(self.monte_carlo_sims):
                # Shuffle the trade sequence
                np.random.shuffle(pnl_values)

                # Build equity curve
                equity = np.zeros(len(pnl_values) + 1)
                equity[0] = initial_capital

                for j in range(len(pnl_values)):
                    equity[j + 1] = equity[j] + pnl_values[j]

                # Calculate metrics
                final_equities[i] = equity[-1]

                # Max drawdown
                peak = np.maximum.accumulate(equity)
                drawdown = (peak - equity) / peak
                max_drawdowns[i] = drawdown.max()

                # Returns for Sharpe
                returns = np.diff(equity) / equity[:-1]
                sharpe_ratios[i] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            # Calculate percentiles
            results = {
                'monte_carlo_runs': self.monte_carlo_sims,
                'final_equity_median': np.median(final_equities),
                'final_equity_95pct': np.percentile(final_equities, 95),
                'final_equity_5pct': np.percentile(final_equities, 5),
                'max_drawdown_median': np.median(max_drawdowns),
                'max_drawdown_95pct': np.percentile(max_drawdowns, 95),
                'sharpe_ratio_median': np.median(sharpe_ratios),
                'probability_profit': np.mean(final_equities > initial_capital),
                'probability_10pct_return': np.mean((final_equities - initial_capital) / initial_capital > 0.1),
                'probability_20pct_drawdown': np.mean(max_drawdowns > 0.2)
            }

            # Save Monte Carlo results
            mc_path = os.path.join(self.results_dir, f"monte_carlo_{datetime.now():%Y%m%d_%H%M%S}.json")

            with open(mc_path, 'w') as f:
                # Convert to JSON serializable format
                json_results = {}
                for key, value in results.items():
                    if isinstance(value, (np.int64, np.float64, np.ndarray)):
                        json_results[key] = float(value)
                    else:
                        json_results[key] = value

                json.dump(json_results, f, indent=2)

            # Create Monte Carlo visualization
            self._create_monte_carlo_plot(final_equities, max_drawdowns)

            # Log results
            self.logger.info("\n=== Monte Carlo Analysis ===")
            self.logger.info(f"Number of simulations: {self.monte_carlo_sims}")
            self.logger.info(f"Median final equity: ${results['final_equity_median']:.2f}")
            self.logger.info(
                f"5th-95th percentile range: ${results['final_equity_5pct']:.2f} - ${results['final_equity_95pct']:.2f}")
            self.logger.info(f"Median max drawdown: {results['max_drawdown_median']:.2%}")
            self.logger.info(f"Probability of profit: {results['probability_profit']:.2%}")
            self.logger.info(f"Probability of >10% return: {results['probability_10pct_return']:.2%}")
            self.logger.info(f"Probability of >20% drawdown: {results['probability_20pct_drawdown']:.2%}")

            return results

        except Exception as e:
            self.logger.error(f"Error in Monte Carlo analysis: {str(e)}")
            return {}

    def _create_monte_carlo_plot(self, final_equities: np.ndarray, max_drawdowns: np.ndarray) -> None:
        """Create Monte Carlo visualization.

        Args:
            final_equities: Array of final equity values
            max_drawdowns: Array of maximum drawdown values
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.results_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # 1. Final equity distribution
            plt.figure(figsize=(10, 6))
            plt.hist(final_equities, bins=20)
            plt.axvline(self.risk_manager.initial_capital, color='r', linestyle='--', label='Initial Capital')
            plt.axvline(np.median(final_equities), color='g', linestyle='--', label='Median')
            plt.axvline(np.percentile(final_equities, 5), color='orange', linestyle='--', label='5th Percentile')
            plt.axvline(np.percentile(final_equities, 95), color='orange', linestyle='--', label='95th Percentile')

            plt.title('Monte Carlo: Final Equity Distribution')
            plt.xlabel('Final Equity')
            plt.ylabel('Frequency')
            plt.legend()

            # Save plot
            plt.savefig(os.path.join(plots_dir, 'monte_carlo_equity.png'))
            plt.close()

            # 2. Drawdown distribution
            plt.figure(figsize=(10, 6))
            plt.hist(max_drawdowns * 100, bins=20)  # Convert to percentage
            plt.axvline(np.median(max_drawdowns) * 100, color='g', linestyle='--', label='Median')
            plt.axvline(np.percentile(max_drawdowns, 95) * 100, color='r', linestyle='--', label='95th Percentile')

            plt.title('Monte Carlo: Maximum Drawdown Distribution')
            plt.xlabel('Maximum Drawdown (%)')
            plt.ylabel('Frequency')
            plt.legend()

            # Save plot
            plt.savefig(os.path.join(plots_dir, 'monte_carlo_drawdown.png'))
            plt.close()

            # 3. Scatterplot of final equity vs max drawdown
            plt.figure(figsize=(10, 6))
            plt.scatter(max_drawdowns * 100, final_equities, alpha=0.5)

            plt.title('Monte Carlo: Final Equity vs Maximum Drawdown')
            plt.xlabel('Maximum Drawdown (%)')
            plt.ylabel('Final Equity')
            plt.grid(True)

            # Save plot
            plt.savefig(os.path.join(plots_dir, 'monte_carlo_scatter.png'))
            plt.close()

        except Exception as e:
            self.logger.error(f"Error creating Monte Carlo plots: {str(e)}")

    @exception_handler(reraise=True)
    def _simulate_test(self, df_test: pd.DataFrame, iteration: int,
                       regime: str = "unknown") -> Tuple[float, List[Dict[str, Any]]]:
        """Simulate trading on test data.

        Args:
            df_test: Test data
            iteration: Iteration number
            regime: Market regime

        Returns:
            Tuple of (final equity, trades list)
        """
        # Reset risk manager state for new test
        self.risk_manager.current_capital = self.risk_manager.initial_capital
        self.risk_manager.open_positions = []
        self.risk_manager.trade_history = []

        # Prepare test data
        X_test, y_test, df_labeled, _ = self.preparer.prepare_test_data(df_test)

        if len(X_test) == 0:
            self.logger.warning(f"Insufficient test data in iteration {iteration}")
            return self.risk_manager.initial_capital, []

        # Get predictions
        if hasattr(self.modeler, 'predict_with_ensemble'):
            preds, uncertainties = self.modeler.predict_with_ensemble(X_test)
        else:
            preds = self.modeler.predict_signals(X_test)
            uncertainties = None

        # Initialize tracking variables
        trades = []
        position = 0
        position_id = None
        trade_entry = None
        sequence_length = self.preparer.sequence_length

        # Simulate trading
        for i in range(len(preds)):
            current_row = i + sequence_length - 1

            if current_row >= len(df_test):
                break

            current_time = df_test.index[current_row]
            current_price = df_test['close'].iloc[current_row]

            # Get signal
            model_probs = preds[i]

            # Get uncertainty if available
            signal_uncertainty = 0
            if uncertainties is not None:
                signal_uncertainty = uncertainties[i].max()

            # Adapt signal producer to current market conditions
            self.signal_producer.adapt_to_market_conditions(df_test.iloc[:current_row])

            # Get signal with current parameters
            signal = self.signal_producer.get_signal(model_probs, df_test.iloc[:current_row + 1])

            # Log signal
            self.logger.debug(
                f"{current_time} - Signal: {signal.get('signal_type', 'N/A')}, "
                f"Confidence: {signal.get('confidence', 'N/A')}, "
                f"Uncertainty: {signal_uncertainty:.4f}"
            )

            # Position management
            if position != 0 and trade_entry is not None:
                # Check for closing existing position
                if position_id is not None:
                    # Get current ATR for dynamic exit management
                    atr_series = self.signal_producer._compute_atr(df_test)
                    current_atr = atr_series.iloc[current_row] if current_row < len(atr_series) else 0

                    # Check exit conditions
                    exit_decision = self.signal_producer.should_close_position(
                        trade_entry, df_test.iloc[:current_row + 1], model_probs
                    )

                    # Handle stop loss or take profit hit
                    if exit_decision.get('should_close', False):
                        # Get exit price
                        exit_price = exit_decision.get('price', current_price)

                        # Add slippage for market orders
                        if exit_decision.get('reason') in ['StopLoss', 'TakeProfit']:
                            # Use exact price for stop loss or take profit
                            pass
                        else:
                            # Add slippage for market orders
                            slippage_amount = current_price * self.slippage
                            if position > 0:  # Long position
                                exit_price = current_price - slippage_amount
                            else:  # Short position
                                exit_price = current_price + slippage_amount

                        # Close the position
                        closed_position = self.risk_manager.close_position(
                            position_id=position_id,
                            exit_price=exit_price,
                            exit_time=current_time,
                            exit_reason=exit_decision.get('reason', 'Unknown')
                        )

                        # Add to trades list
                        trades.append(closed_position)

                        # Reset position tracking
                        position = 0
                        position_id = None
                        trade_entry = None

                    # Handle trailing stop adjustment
                    elif exit_decision.get('should_update_stop', False):
                        new_stop = exit_decision.get('new_stop')

                        # Update the position
                        updated_position = self.risk_manager.update_position(
                            position_id=position_id,
                            updates={'stop_loss': new_stop}
                        )

                        # Update trade entry
                        if updated_position:
                            trade_entry = updated_position

                    # Handle partial exit
                    elif exit_decision.get('partial_exit', False):
                        # Calculate exit ratio
                        exit_ratio = exit_decision.get('exit_ratio', 0.5)
                        exit_reason = exit_decision.get('reason', 'PartialExit')

                        # Calculate quantity to close
                        full_quantity = trade_entry['quantity']
                        close_quantity = full_quantity * exit_ratio

                        # Calculate exit price with slippage
                        slippage_amount = current_price * self.slippage
                        if position > 0:  # Long position
                            exit_price = current_price - slippage_amount
                        else:  # Short position
                            exit_price = current_price + slippage_amount

                        # Calculate P&L for partial exit
                        entry_price = trade_entry['entry_price']
                        direction = trade_entry['direction']

                        if direction == 'long':
                            pnl = close_quantity * (exit_price - entry_price)
                        else:
                            pnl = close_quantity * (entry_price - exit_price)

                        # Add partial trade to trade history
                        partial_trade = trade_entry.copy()
                        partial_trade.update({
                            'exit_price': exit_price,
                            'exit_time': current_time,
                            'exit_reason': exit_reason,
                            'pnl': pnl,
                            'pnl_percent': pnl / (entry_price * close_quantity) * 100,
                            'quantity': close_quantity
                        })

                        # Add to trades list
                        trades.append(partial_trade)

                        # Update remaining position
                        if exit_ratio < 1.0:
                            remaining_quantity = full_quantity - close_quantity

                            # Mark partial exit in position
                            update_data = {
                                'quantity': remaining_quantity,
                                'partial_exit_1': True if exit_reason == 'FirstTarget' else trade_entry.get(
                                    'partial_exit_1', False),
                                'partial_exit_2': True if exit_reason == 'SecondTarget' else trade_entry.get(
                                    'partial_exit_2', False)
                            }

                            # Update stop loss if specified
                            if exit_decision.get('update_stop', False):
                                update_data['stop_loss'] = exit_decision.get('new_stop')

                            # Update the position
                            updated_position = self.risk_manager.update_position(
                                position_id=position_id,
                                updates=update_data
                            )

                            # Update trade entry
                            if updated_position:
                                trade_entry = updated_position

                                # Update the risk manager's capital
                                self.risk_manager.current_capital += pnl
                        else:
                            # Full exit
                            position = 0
                            position_id = None
                            trade_entry = None

            # Check for entry conditions if no position
            if position == 0 and ("Buy" in signal['signal_type'] or "Sell" in signal['signal_type']):
                direction = 'long' if "Buy" in signal['signal_type'] else 'short'
                confidence = signal['confidence']

                # Apply confidence adjustment based on uncertainty
                if uncertainties is not None:
                    uncertainty = signal_uncertainty
                    if uncertainty > 0.3:  # High uncertainty threshold
                        confidence *= (1 - uncertainty)

                    # Skip if confidence now too low
                    if confidence < self.signal_producer.confidence_threshold:
                        continue

                # Check correlation risk for the new position
                risk_check = self.risk_manager.check_correlation_risk({
                    'direction': direction,
                    'risk_amount': self.risk_manager.current_capital * self.risk_manager.max_risk_per_trade
                })

                can_add_position, adjusted_risk = risk_check

                if not can_add_position:
                    continue

                # Determine stop loss price
                stop_loss = signal['stop_loss']

                # Calculate position size
                volatility_regime = signal.get('volatility_regime', 0)
                quantity = self.risk_manager.calculate_position_size(
                    signal=signal,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    volatility_regime=volatility_regime
                )

                if quantity <= 0:
                    continue

                # Add slippage to entry price
                slippage_amount = current_price * self.slippage
                if direction == 'long':
                    entry_price = current_price + slippage_amount
                else:
                    entry_price = current_price - slippage_amount

                # Set take profit level
                take_profit_dist = abs(entry_price - stop_loss) * self.risk_manager.reward_risk_ratio
                if direction == 'long':
                    take_profit = entry_price + take_profit_dist
                else:
                    take_profit = entry_price - take_profit_dist

                # Create position entry
                position_data = {
                    'iteration': iteration,
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'direction': direction,
                    'quantity': quantity,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_signal': signal['signal_type'],
                    'confidence': confidence,
                    'partial_exit_1': False,
                    'partial_exit_2': False,
                    'regime': regime
                }

                # Add position through risk manager
                position_id = self.risk_manager.add_position(position_data)

                if position_id > 0:
                    # Get the full position data
                    for pos in self.risk_manager.open_positions:
                        if pos.get('id') == position_id:
                            trade_entry = pos
                            break

                    position = 1 if direction == 'long' else -1

                    self.logger.info(
                        f"{current_time} - Opening {direction} position at {entry_price:.2f}, "
                        f"qty: {quantity:.8f}, signal: {signal['signal_type']}, "
                        f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}"
                    )

        # Close any open position at the end of test
        if position != 0 and trade_entry is not None and position_id is not None:
            # Close the position
            closed_position = self.risk_manager.close_position(
                position_id=position_id,
                exit_price=df_test['close'].iloc[-1],
                exit_time=df_test.index[-1],
                exit_reason="EndOfTest"
            )

            # Add to trades list
            trades.append(closed_position)

        # Return final equity and trades
        return self.risk_manager.current_capital, trades