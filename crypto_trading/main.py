#!/usr/bin/env python
"""
Main entry point for the cryptocurrency trading system.

This module provides command-line interface to run different modes
of the system: training, backtesting, or live trading.
"""

import argparse
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from crypto_trading.config import ConfigurationManager, SystemConfig, DataConfig
from crypto_trading.data import BitcoinData, CryptoDataPreparer, EnhancedCryptoFeatureEngineer
from crypto_trading.models import EnhancedCryptoModel
from crypto_trading.trading import EnhancedSignalProducer, AdvancedRiskManager
from crypto_trading.trading.backtester import EnhancedStrategyBacktester
from crypto_trading.utils import LoggerFactory, setup_temperature_monitoring, monitor_memory
from utils.path_manager import get_path_manager


# Update the setup_logging function:
def setup_logging(log_level: str = 'INFO', log_dir: str = None) -> logging.Logger:
    """Set up logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files

    Returns:
        Root logger
    """
    # Use path manager if log_dir not explicitly provided
    if log_dir is None:
        path_manager = get_path_manager()
        log_dir = str(path_manager.get_date_dir('logs'))

    # Create LoggerFactory instance
    os.makedirs(log_dir, exist_ok=True)
    log_factory = LoggerFactory(
        log_level=getattr(logging, log_level),
        log_dir=log_dir,
        console_logging=True,
        file_logging=True
    )

    # Get main logger
    logger = log_factory.get_logger('CryptoTrading')
    logger.info(f"Logging initialized. Level: {log_level}, Directory: {log_dir}")

    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading System')

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'backtest', 'live', 'fetch_data'],
                        default='backtest', help='Operation mode')

    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Logging level')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files')

    # Output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory for output files')
    parser.add_argument('--save-model', type=str, default='models/best_model.keras',
                        help='Path to save the model')

    # Data options
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory for data files')
    parser.add_argument('--live-data', action='store_true',
                        help='Fetch fresh data from API')
    parser.add_argument('--lookback-days', type=int, default=None,
                        help='Days of historical data to use')

    # Model options
    parser.add_argument('--ensemble-size', type=int, default=None,
                        help='Number of models in ensemble')
    parser.add_argument('--sequence-length', type=int, default=None,
                        help='Sequence length for model input')
    parser.add_argument('--use-gpu', action='store_true', default=None,
                        help='Use GPU for training/inference')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to load an existing model')

    # Backtesting options
    parser.add_argument('--train-window', type=int, default=None,
                        help='Size of training window for walk-forward backtesting')
    parser.add_argument('--test-window', type=int, default=None,
                        help='Size of testing window for walk-forward backtesting')
    parser.add_argument('--monte-carlo', type=int, default=None,
                        help='Number of Monte Carlo simulations')

    # Performance monitoring
    parser.add_argument('--monitor-memory', action='store_true',
                        help='Enable memory monitoring')
    parser.add_argument('--monitor-temperature', action='store_true',
                        help='Enable GPU/CPU temperature monitoring')

    return parser.parse_args()


def train_mode(config, logger):
    """Run training mode.

    Args:
        config: System configuration
        logger: Logger instance
    """
    logger.info("Starting training mode")

    # Initialize data client
    data_client = BitcoinData(
        csv_30m=os.path.join(config.data.csv_30m),
        csv_4h=os.path.join(config.data.csv_4h),
        csv_daily=os.path.join(config.data.csv_daily),
        csv_oi=os.path.join(config.data.csv_oi),
        csv_funding=os.path.join(config.data.csv_funding),
        use_testnet=config.data.use_binance_testnet,
        logger=logger
    )

    # Fetch or load data
    logger.info("Loading data")
    data_dict = data_client.fetch_all_data(
        live=config.mode == 'live' or config.data.use_binance_api,
        lookback_candles=config.data.lookback_30m_candles
    )

    # Feature engineering
    logger.info("Performing feature engineering")
    feature_engineer = EnhancedCryptoFeatureEngineer(
        feature_scaling=config.features.feature_scaling,
        config=config.features.__dict__,
        logger=logger
    )

    # Process data
    processed_data = feature_engineer.process_data_in_chunks(
        df_30m=data_dict['30m'],
        df_4h=data_dict['4h'],
        df_daily=data_dict['daily'],
        chunk_size=config.features.chunk_size,
        df_oi=data_dict.get('open_interest'),
        df_funding=data_dict.get('funding_rates'),
        use_parallel=True
    )

    # Data preparation
    logger.info("Preparing data for training")
    data_preparer = CryptoDataPreparer(
        sequence_length=config.model.sequence_length,
        horizon=config.model.horizon,
        normalize_method='zscore',
        train_ratio=config.model.train_ratio,
        logger=logger
    )

    # Create model
    logger.info("Creating model")
    model = EnhancedCryptoModel(
        project_name=config.model.project_name,
        max_trials=config.model.max_trials,
        tuner_type=config.model.tuner_type,
        model_save_path=config.model.model_save_path,
        label_smoothing=config.model.label_smoothing,
        ensemble_size=config.model.ensemble_size,
        use_mixed_precision=config.model.use_mixed_precision,
        use_xla_acceleration=config.model.use_xla_acceleration,
        logger=logger
    )

    # Prepare data
    X_train, y_train, X_val, y_val, df_val, fwd_returns_val = data_preparer.prepare_data(
        df=processed_data,
        batch_size=config.model.batch_size
    )

    # Build ensemble or single model
    if config.model.ensemble_size > 1:
        logger.info(f"Training ensemble of {config.model.ensemble_size} models")
        model.build_ensemble(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            df_val=df_val,
            fwd_returns_val=fwd_returns_val,
            epochs=config.model.epochs,
            batch_size=config.model.batch_size
        )
    else:
        logger.info("Training single model")
        model.tune_and_train(
            iteration=1,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            df_val=df_val,
            fwd_returns_val=fwd_returns_val,
            epochs=config.model.epochs,
            batch_size=config.model.batch_size
        )

    # Evaluate model
    logger.info("Evaluating model")
    evaluation_metrics = model.evaluate(X_val, y_val)

    # Save model summary
    logger.info("Saving model summary")
    model.save_model_summary()

    # Print evaluation results
    logger.info("Training completed. Evaluation results:")
    for metric, value in evaluation_metrics.items():
        logger.info(f"{metric}: {value}")


def backtest_mode(config: SystemConfig, logger: logging.Logger) -> None:
    """Run system in backtest mode.

    Args:
        config: System configuration
        logger: System logger
    """
    logger.info("Starting backtest mode")

    # Step 1: Load data
    logger.info("Loading data")
    loaded_data = load_data(config.data, logger)

    # Step 2: Perform feature engineering
    logger.info("Performing feature engineering")
    feature_engineer = EnhancedCryptoFeatureEngineer(
        feature_scaling=config.features.feature_scaling,
        logger=logger,
        config=asdict(config.features)
    )

    processed_data = feature_engineer.process_data_in_chunks(
        df_30m=loaded_data['30m'],
        df_4h=loaded_data['4h'],
        df_daily=loaded_data['daily'],
        chunk_size=config.features.chunk_size,
        df_oi=loaded_data.get('open_interest'),
        df_funding=loaded_data.get('funding_rates'),
        use_parallel=config.backtest.use_parallel_processing,
        max_workers=config.backtest.max_workers
    )

    # Verify data is not empty
    if processed_data.empty:
        logger.error("No data after feature engineering. Creating synthetic data for demo.")

        # Create synthetic data for demonstration purposes
        processed_data = create_synthetic_data(loaded_data['30m'].index[-1000:], logger)
    elif len(processed_data) < 100:
        logger.warning(
            f"Very little data after feature engineering: {len(processed_data)} rows. Adding synthetic data.")

        # Add more synthetic data rows
        synthetic_data = create_synthetic_data(loaded_data['30m'].index[-1000:], logger)
        processed_data = pd.concat([processed_data, synthetic_data])

    # Step 3: Prepare data for backtesting
    logger.info("Preparing data for backtesting")
    data_preparer = CryptoDataPreparer(
        sequence_length=config.model.sequence_length,
        horizon=config.model.horizon,
        normalize_method='zscore' if config.features.feature_scaling else None,
        train_ratio=config.model.train_ratio,
        logger=logger
    )

    # Step 4: Create model
    logger.info("Creating model")
    crypto_model = EnhancedCryptoModel(
        project_name=config.model.project_name,
        max_trials=config.model.max_trials,
        tuner_type=config.model.tuner_type,
        model_save_path=config.model.model_save_path,
        label_smoothing=config.model.label_smoothing,
        ensemble_size=config.model.ensemble_size,
        use_mixed_precision=config.model.use_mixed_precision,
        use_xla_acceleration=config.model.use_xla_acceleration,
        logger=logger
    )

    # Step 5: Create signal producer
    logger.info("Creating signal producer")
    signal_producer = EnhancedSignalProducer(
        confidence_threshold=config.signal.confidence_threshold,
        strong_signal_threshold=config.signal.strong_signal_threshold,
        atr_multiplier_sl=config.signal.atr_multiplier_sl,
        use_regime_filter=config.signal.use_regime_filter,
        use_volatility_filter=config.signal.use_volatility_filter,
        min_adx_threshold=config.signal.min_adx_threshold,
        max_vol_percentile=config.signal.max_vol_percentile,
        correlation_threshold=config.signal.correlation_threshold,
        use_oi_filter=config.signal.use_open_interest,
        use_funding_filter=config.signal.use_funding_rate,
        logger=logger
    )

    # Step 6: Create risk manager
    logger.info("Creating risk manager")
    risk_manager = AdvancedRiskManager(
        initial_capital=config.risk.initial_capital,
        max_risk_per_trade=config.risk.max_risk_per_trade,
        max_correlated_exposure=config.risk.max_correlated_exposure,
        volatility_scaling=config.risk.volatility_scaling,
        target_annual_vol=config.risk.target_annual_vol,
        reward_risk_ratio=config.risk.reward_risk_ratio,
        partial_close_ratio=config.risk.partial_close_ratio,
        max_drawdown_threshold=0.20,  # Fixed value for now
        trade_frequency_limit=12,  # Fixed value for now
        consecutive_loss_threshold=3,  # Fixed value for now
        logger=logger
    )

    # Step 7: Create and run backtester
    logger.info("Creating backtester")
    backtester = EnhancedStrategyBacktester(
        data_df=processed_data,
        preparer=data_preparer,
        modeler=crypto_model,
        signal_producer=signal_producer,
        risk_manager=risk_manager,
        train_window_size=config.backtest.train_window_size,
        test_window_size=config.backtest.test_window_size,
        fixed_cost=config.backtest.fixed_cost,
        variable_cost=config.backtest.variable_cost,
        slippage=config.backtest.slippage,
        walk_forward_steps=config.backtest.walk_forward_steps,
        monte_carlo_sims=config.backtest.monte_carlo_sims,
        use_parallel=config.backtest.use_parallel_processing,
        max_workers=config.backtest.max_workers,
        results_dir=config.backtest.results_directory,
        logger=logger
    )

    logger.info("Running backtest")
    backtest_results = backtester.walk_forward_backtest()

    # Step 8: Save and display results
    save_path = os.path.join(
        config.backtest.results_directory,
        f"backtest_summary_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    backtest_results.to_csv(save_path, index=False)
    logger.info(f"Backtest results saved to {save_path}")

    # Display summary
    initial_capital = config.risk.initial_capital
    final_capital = float(backtest_results['final_equity'].iloc[-1]) if not backtest_results.empty else initial_capital
    total_return = ((final_capital / initial_capital) - 1) * 100

    logger.info("\n==== Backtest Summary ====")
    logger.info(f"Initial Capital: ${initial_capital:.2f}")
    logger.info(f"Final Capital: ${final_capital:.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info("==========================")


def create_synthetic_data(index, logger):
    """Create synthetic data for demonstration purposes.

    Args:
        index: DatetimeIndex to use for the synthetic data
        logger: Logger instance

    Returns:
        DataFrame with synthetic data
    """
    logger.warning("Creating synthetic data for demo purposes")

    # Create a DataFrame with basic columns
    synthetic_df = pd.DataFrame(index=index)

    # Add price data
    base_price = 50000
    synthetic_df['close'] = base_price + np.cumsum(np.random.normal(0, 100, len(index)))
    synthetic_df['open'] = synthetic_df['close'].shift(1).fillna(base_price)
    synthetic_df['high'] = synthetic_df[['open', 'close']].max(axis=1) + np.random.normal(50, 30, len(index))
    synthetic_df['low'] = synthetic_df[['open', 'close']].min(axis=1) - np.random.normal(50, 30, len(index))
    synthetic_df['volume'] = np.random.lognormal(10, 1, len(index))

    # Add technical indicators
    # SMA
    for period in [20, 50, 100, 200]:
        synthetic_df[f'h4_SMA_{period}'] = synthetic_df['close'].rolling(window=period // 4).mean()

    # RSI
    delta = synthetic_df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    synthetic_df['h4_RSI_14'] = 100 - (100 / (1 + rs))
    synthetic_df['d1_RSI_14'] = synthetic_df['h4_RSI_14'].rolling(6).mean()

    # ATR
    high_low = synthetic_df['high'] - synthetic_df['low']
    high_close = (synthetic_df['high'] - synthetic_df['close'].shift()).abs()
    low_close = (synthetic_df['low'] - synthetic_df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    synthetic_df['d1_ATR_14'] = tr.rolling(window=14).mean()

    # MACD
    ema12 = synthetic_df['close'].ewm(span=12).mean()
    ema26 = synthetic_df['close'].ewm(span=26).mean()
    synthetic_df['h4_MACD'] = ema12 - ema26
    synthetic_df['h4_MACD_signal'] = synthetic_df['h4_MACD'].ewm(span=9).mean()

    # Market regime and volatility
    synthetic_df['market_regime'] = np.random.choice([-1, 0, 1], len(index), p=[0.3, 0.4, 0.3])
    synthetic_df['volatility_regime'] = np.random.choice([-1, 0, 1], len(index), p=[0.3, 0.4, 0.3])
    synthetic_df['trend_strength'] = np.random.uniform(-0.8, 0.8, len(index))

    # Historical volatility
    synthetic_df['hist_vol_20'] = synthetic_df['close'].pct_change().rolling(20).std() * np.sqrt(252)

    # Add prefix to any 30m indicators
    for col in synthetic_df.columns:
        if col not in ['open', 'high', 'low', 'close', 'volume', 'market_regime', 'volatility_regime',
                       'trend_strength'] and not col.startswith('h4_') and not col.startswith('d1_'):
            synthetic_df[f'm30_{col}'] = synthetic_df[col]
            synthetic_df.drop(columns=[col], inplace=True)

    # Fill NaN values
    synthetic_df = synthetic_df.ffill().bfill()

    logger.info(f"Created synthetic data with {len(synthetic_df)} rows and {len(synthetic_df.columns)} columns")
    return synthetic_df

def live_mode(config, logger):
    """Run live trading mode.

    Args:
        config: System configuration
        logger: Logger instance
    """
    logger.info("Starting live trading mode")
    logger.warning("Live trading mode is not yet implemented")
    # TODO: Implement live trading mode
    # This should include:
    # 1. Setting up API connections
    # 2. Loading pre-trained model
    # 3. Implementing data streaming
    # 4. Setting up order execution
    # 5. Implementing safety mechanisms and monitoring


def fetch_data_mode(config, logger):
    """Run data fetching mode.

    Args:
        config: System configuration
        logger: Logger instance
    """
    logger.info("Starting data fetch mode")

    # Initialize data client
    data_client = BitcoinData(
        csv_30m=os.path.join(config.data.csv_30m),
        csv_4h=os.path.join(config.data.csv_4h),
        csv_daily=os.path.join(config.data.csv_daily),
        csv_oi=os.path.join(config.data.csv_oi),
        csv_funding=os.path.join(config.data.csv_funding),
        use_testnet=config.data.use_binance_testnet,
        logger=logger
    )

    # Fetch data
    logger.info("Fetching data from API")
    data_dict = data_client.fetch_all_data(
        live=True,  # Always fetch fresh data in fetch_data mode
        lookback_candles=config.data.lookback_30m_candles
    )

    logger.info("Data fetched successfully:")
    for key, df in data_dict.items():
        if df is not None and not df.empty:
            logger.info(f"- {key}: {len(df)} records from {df.index[0]} to {df.index[-1]}")
        else:
            logger.warning(f"- {key}: No data")

    # Feature engineering
    run_feature_engineering = input(
        "Do you want to run feature engineering on the fetched data? (y/n): ").lower() == 'y'

    if run_feature_engineering:
        logger.info("Performing feature engineering")
        feature_engineer = EnhancedCryptoFeatureEngineer(
            feature_scaling=config.features.feature_scaling,
            logger=logger,
            config=config.features  # Pass the dataclass directly
        )

        # Process data
        processed_data = feature_engineer.process_data_in_chunks(
            df_30m=data_dict['30m'],
            df_4h=data_dict['4h'],
            df_daily=data_dict['daily'],
            chunk_size=config.features.chunk_size,
            df_oi=data_dict.get('open_interest'),
            df_funding=data_dict.get('funding_rates'),
            use_parallel=True
        )

        # Save processed data
        processed_data_path = os.path.join(config.data.cache_directory,
                                           f"processed_data_{datetime.now():%Y%m%d_%H%M%S}.csv")
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        processed_data.to_csv(processed_data_path)
        logger.info(f"Processed data saved to {processed_data_path}")


def load_data(config: DataConfig, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Load cryptocurrency data.

    Args:
        config: Data configuration
        logger: Logger instance

    Returns:
        Dictionary with dataframes for different timeframes
    """
    logger.info("Initializing data client")
    bitcoin_data = BitcoinData(
        csv_30m=config.csv_30m,
        csv_4h=config.csv_4h,
        csv_daily=config.csv_daily,
        csv_oi=config.csv_oi,
        csv_funding=config.csv_funding,
        use_testnet=config.use_binance_testnet,
        logger=logger
    )

    logger.info("Fetching market data")
    return bitcoin_data.fetch_all_data(
        live=not config.use_data_caching,
        lookback_candles=config.lookback_30m_candles
    )

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Get path manager
    path_manager = get_path_manager()

    # Setup logging (use path manager)
    log_dir = path_manager.get_date_dir('logs')
    logger = setup_logging(args.log_level, str(log_dir))
    logger.info("Starting Cryptocurrency Trading System")

    # Load configuration
    config_manager = ConfigurationManager(config_path=args.config, logger=logger)
    config = config_manager.load_config()

    # Override config with command line arguments if provided
    if args.output_dir:
        config.backtest.results_directory = args.output_dir
    if args.save_model:
        config.model.model_save_path = args.save_model
    if args.data_dir:
        for attr in ['csv_30m', 'csv_4h', 'csv_daily', 'csv_oi', 'csv_funding']:
            if hasattr(config.data, attr):
                setattr(config.data, attr, os.path.join(args.data_dir, os.path.basename(getattr(config.data, attr))))
    if args.live_data:
        config.data.use_binance_api = True
    if args.lookback_days:
        candles_per_day = 48  # 48 30-minute candles per day
        config.data.lookback_30m_candles = args.lookback_days * candles_per_day
    if args.ensemble_size:
        config.model.ensemble_size = args.ensemble_size
    if args.sequence_length:
        config.model.sequence_length = args.sequence_length
    if args.train_window:
        config.backtest.train_window_size = args.train_window
    if args.test_window:
        config.backtest.test_window_size = args.test_window
    if args.monte_carlo:
        config.backtest.monte_carlo_sims = args.monte_carlo
    if args.load_model:
        config.load_model = args.load_model

    # Configure GPU usage
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Set up performance monitoring if requested
    if args.monitor_memory or config.monitor_memory:
        logger.info("Setting up memory monitoring")
        memory_log_dir = path_manager.get_path('logs_memory')
        memory_log_file = str(memory_log_dir / f"memory_usage_{datetime.now():%Y%m%d_%H%M%S}.log")
        monitor_memory(
            threshold_gb=config.memory_threshold_gb,
            interval_seconds=60,
            log_file=memory_log_file
        )

    if args.monitor_temperature or config.monitor_temperature:
        logger.info("Setting up temperature monitoring")
        temp_log_dir = path_manager.get_path('logs_temperature')
        setup_temperature_monitoring(
            gpu_threshold=80,
            cpu_threshold=90,
            log_dir=str(temp_log_dir)
        )
    # Create necessary directories
    os.makedirs(config.backtest.results_directory, exist_ok=True)
    os.makedirs(os.path.dirname(config.model.model_save_path), exist_ok=True)
    os.makedirs(config.data.cache_directory, exist_ok=True)

    try:
        # Run the selected mode
        if args.mode == 'train':
            train_mode(config, logger)
        elif args.mode == 'backtest':
            backtest_mode(config, logger)
        elif args.mode == 'live':
            live_mode(config, logger)
        elif args.mode == 'fetch_data':
            fetch_data_mode(config, logger)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user. Shutting down...")
    except Exception as e:
        logger.exception(f"Error in {args.mode} mode: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Cryptocurrency Trading System shutting down")


if __name__ == "__main__":
    main()