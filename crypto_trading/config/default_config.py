"""
Default configuration settings for the crypto trading system.

This module defines the default configuration settings and dataclasses
for different components of the system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DataConfig:
    """Configuration for data sources and parameters."""
    # File paths
    csv_30m: str = 'data/btc_data_30m.csv'
    csv_4h: str = 'data/btc_data_4h.csv'
    csv_daily: str = 'data/btc_data_daily.csv'
    csv_oi: str = 'data/btc_open_interest.csv'
    csv_funding: str = 'data/btc_funding_rates.csv'

    # Data collection parameters
    lookback_30m_candles: int = 7000
    lookback_4h_candles: int = 1000
    lookback_daily_candles: int = 300

    # API settings
    use_binance_api: bool = True
    binance_base_url: str = 'https://fapi.binance.com'
    binance_timeout: int = 30

    # Data processing
    use_data_caching: bool = True
    cache_directory: str = 'data/cache'
    use_binance_testnet: bool = False


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""
    # General settings
    feature_scaling: bool = True
    chunk_size: int = 2000  # For chunked processing

    # Technical indicator parameters
    ma_periods_daily: List[int] = field(default_factory=lambda: [10, 20, 50])
    ma_periods_4h: List[int] = field(default_factory=lambda: [20, 50, 100, 200])
    rsi_period_daily: int = 14
    rsi_period_4h: int = 14
    macd_fast_daily: int = 12
    macd_slow_daily: int = 26
    macd_signal_daily: int = 9
    bb_period_daily: int = 20
    bb_stddev_daily: int = 2
    atr_period_daily: int = 14
    mfi_period_daily: int = 14
    cmf_period_daily: int = 21

    # Advanced feature parameters
    regime_window: int = 20
    volume_zones_lookback: int = 50
    swing_threshold: float = 0.5

    # Feature selection
    use_feature_selection: bool = True
    top_features_count: int = 50
    force_include_features: List[str] = field(default_factory=lambda: [
        'close', 'volume', 'market_regime', 'volatility_regime',
        'trend_strength', 'swing_high', 'swing_low'
    ])


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    # Model identification
    project_name: str = 'enhanced_crypto_model'
    model_save_path: str = 'models/saved/best_model.keras'

    # Hyperparameter tuning
    tuner_type: str = 'bayesian'  # 'bayesian' or 'hyperband'
    max_trials: int = 100
    executions_per_trial: int = 1

    # Model parameters
    ensemble_size: int = 3
    label_smoothing: float = 0.1
    focal_loss_gamma: float = 2.0
    learning_rate_min: float = 1e-4
    learning_rate_max: float = 1e-2

    # Sequence parameters
    sequence_length: int = 144  # 3 days of 30-minute data
    horizon: int = 48  # 1 day forecast

    # Training parameters
    train_ratio: float = 0.8
    batch_size: int = 256
    epochs: int = 32
    early_stopping_patience: int = 6

    # Performance optimization
    use_mixed_precision: bool = True
    use_xla_acceleration: bool = True
    tf_memory_growth: bool = True
    memory_limit_mb: int = 5120  # 5GB


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    # Signal thresholds
    confidence_threshold: float = 0.4
    strong_signal_threshold: float = 0.7

    # Risk management parameters
    atr_multiplier_sl: float = 1.5

    # Filters
    use_regime_filter: bool = True
    use_volatility_filter: bool = True
    min_adx_threshold: int = 20
    max_vol_percentile: int = 85

    # Multi-timeframe confirmation
    correlation_threshold: float = 0.6

    # Use additional data sources
    use_open_interest: bool = True
    use_funding_rate: bool = True
    use_liquidation_data: bool = True


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    # Capital allocation
    initial_capital: float = 10000.0
    max_risk_per_trade: float = 0.02
    max_correlated_exposure: float = 0.06

    # Trade management
    volatility_scaling: bool = True
    target_annual_vol: float = 0.2
    reward_risk_ratio: float = 2.5
    partial_close_ratio: float = 0.5

    # Stop loss configuration
    atr_period: int = 14
    atr_multiplier_sl: float = 1.5

    # Trailing stop parameters
    use_trailing_stops: bool = True
    trailing_stop_activation_pct: float = 0.02  # 2% profit to activate

    # Performance adjustment
    adjust_position_after_losses: bool = True
    max_consecutive_losses: int = 3
    position_reduction_factor: float = 0.5  # Reduce by 50% after max losses


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Window sizes
    train_window_size: int = 5000
    test_window_size: int = 1000
    step_size: int = 500  # Step between windows

    # Trade costs
    fixed_cost: float = 0.001
    variable_cost: float = 0.0005
    slippage: float = 0.0005

    # Backtesting parameters
    walk_forward_steps: int = 4
    monte_carlo_sims: int = 100

    # Parallel processing
    use_parallel_processing: bool = True
    max_workers: Optional[int] = None  # None = use CPU count

    # Results
    results_directory: str = "results/backtest"
    save_trades: bool = True
    plot_results: bool = True


@dataclass
class SystemConfig:
    """Overall system configuration."""
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # System parameters
    log_level: str = 'INFO'
    log_dir: str = 'logs'
    mode: str = 'backtest'  # 'backtest' or 'live'

    # Resource management
    gpu_memory_limit: int = 5120  # in MB
    monitor_memory: bool = True
    monitor_temperature: bool = True
    memory_threshold_gb: int = 14

    # Environment
    env: str = 'dev'  # 'dev', 'test', or 'prod'
    binance_api_key_env_var: str = "BINANCE_API_KEY"
    binance_api_secret_env_var: str = "BINANCE_API_SECRET"


def get_default_config() -> SystemConfig:
    """Returns the default system configuration.

    Returns:
        SystemConfig: Default configuration for the system
    """
    return SystemConfig()