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
from datetime import datetime
from pathlib import Path
import json

from crypto_trading.config import ConfigurationManager, get_default_config
from crypto_trading.utils import LoggerFactory, setup_temperature_monitoring, monitor_memory
from crypto_trading.data import BitcoinData, CryptoDataPreparer, EnhancedCryptoFeatureEngineer
from crypto_trading.models import EnhancedCryptoModel
from crypto_trading.trading import EnhancedSignalProducer, AdvancedRiskManager
from crypto_trading.trading.backtester import EnhancedStrategyBacktester


def setup_logging(log_level: str = 'INFO', log_dir: str = 'logs') -> logging.Logger:
    """Set up logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files

    Returns:
        Root logger
    """
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


def backtest_mode(config, logger):
    """Run backtest mode.

    Args:
        config: System configuration
        logger: Logger instance
    """
    logger.info("Starting backtest mode")

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
    logger.info("Preparing data for backtesting")
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

    # Load existing model if specified
    if hasattr(config, 'load_model') and config.load_model:
        if os.path.exists(config.load_model):
            logger.info(f"Loading model from {config.load_model}")
            model.model_save_path = config.load_model
            model.load_best_model()
        else:
            logger.warning(f"Model file not found: {config.load_model}. Will train new model.")

    # Create signal producer
    logger.info("Creating signal producer")
    signal_producer = EnhancedSignalProducer(
        confidence_threshold=config.signal.confidence_threshold,
        strong_signal_threshold=config.signal.strong_signal_threshold,
        atr_multiplier_sl=config.signal.atr_multiplier_sl,
        use_regime_filter=config.signal.use_regime_filter,
        use_volatility_filter=config.signal.use_volatility_filter,
        logger=logger
    )

    # Create risk manager
    logger.info("Creating risk manager")
    risk_manager = AdvancedRiskManager(
        initial_capital=config.risk.initial_capital,
        max_risk_per_trade=config.risk.max_risk_per_trade,
        max_correlated_exposure=config.risk.max_correlated_exposure,
        volatility_scaling=config.risk.volatility_scaling,
        target_annual_vol=config.risk.target_annual_vol,
        reward_risk_ratio=config.risk.reward_risk_ratio,
        logger=logger
    )

    # Create backtester
    logger.info("Creating backtester")
    backtester = EnhancedStrategyBacktester(
        data_df=processed_data,
        preparer=data_preparer,
        modeler=model,
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

    # Run backtest
    logger.info("Running backtest")
    backtest_results = backtester.walk_forward_backtest()

    # Save backtest results
    results_file = os.path.join(config.backtest.results_directory,
                                f"backtest_summary_{datetime.now():%Y%m%d_%H%M%S}.csv")
    backtest_results.to_csv(results_file)
    logger.info(f"Backtest results saved to {results_file}")

    # Display summary statistics
    logger.info("\n==== Backtest Summary ====")
    initial_capital = config.risk.initial_capital
    final_capital = backtest_results['final_equity'].iloc[-1] if not backtest_results.empty else initial_capital
    total_return = (final_capital - initial_capital) / initial_capital * 100

    logger.info(f"Initial Capital: ${initial_capital:.2f}")
    logger.info(f"Final Capital: ${final_capital:.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info("==========================")


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

        # Save processed data
        processed_data_path = os.path.join(config.data.cache_directory,
                                           f"processed_data_{datetime.now():%Y%m%d_%H%M%S}.csv")
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        processed_data.to_csv(processed_data_path)
        logger.info(f"Processed data saved to {processed_data_path}")


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()

    # Setup logging
    logger = setup_logging(args.log_level, args.log_dir)
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
        monitor_memory(threshold_gb=config.memory_threshold_gb, interval_seconds=60,
                       log_file=os.path.join(args.log_dir, 'memory_usage.log'))

    if args.monitor_temperature or config.monitor_temperature:
        logger.info("Setting up temperature monitoring")
        setup_temperature_monitoring(
            gpu_threshold=80,
            cpu_threshold=90,
            log_dir=os.path.join(args.log_dir, 'temperature')
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