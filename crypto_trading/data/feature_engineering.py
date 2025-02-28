"""
Feature engineering for cryptocurrency market data.

This module provides classes and functions for generating features from
raw market data for use in machine learning models.
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ProcessPoolExecutor
import functools
import gc

from ..utils.logging_utils import exception_handler
from ..utils.memory_monitor import memory_usage_decorator, log_memory_usage


class EnhancedCryptoFeatureEngineer:
    """Enhanced feature engineering for cryptocurrency market data."""

    def __init__(self, feature_scaling: bool = False,
                 logger: Optional[logging.Logger] = None,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineer.

        Args:
            feature_scaling: Whether to scale features
            logger: Logger to use
            config: Configuration dictionary with feature parameters
        """
        self.feature_scaling = feature_scaling
        self.logger = logger or logging.getLogger('FeatureEngineer')

        # Load configuration or use defaults
        self.config = config or {}

        # Daily parameters
        self.ma_periods_daily = self.config.get('ma_periods_daily', [10, 20, 50])
        self.rsi_period_daily = self.config.get('rsi_period_daily', 14)
        self.macd_fast_daily = self.config.get('macd_fast_daily', 12)
        self.macd_slow_daily = self.config.get('macd_slow_daily', 26)
        self.macd_signal_daily = self.config.get('macd_signal_daily', 9)
        self.bb_period_daily = self.config.get('bb_period_daily', 20)
        self.bb_stddev_daily = self.config.get('bb_stddev_daily', 2)
        self.atr_period_daily = self.config.get('atr_period_daily', 14)
        self.mfi_period_daily = self.config.get('mfi_period_daily', 14)
        self.cmf_period_daily = self.config.get('cmf_period_daily', 21)

        # 4H parameters
        self.ma_periods_4h = self.config.get('ma_periods_4h', [20, 50, 100, 200])
        self.rsi_period_4h = self.config.get('rsi_period_4h', 14)
        self.macd_fast_4h = self.config.get('macd_fast_4h', 12)
        self.macd_slow_4h = self.config.get('macd_slow_4h', 26)
        self.macd_signal_4h = self.config.get('macd_signal_4h', 9)
        self.mfi_period_4h = self.config.get('mfi_period_4h', 14)
        self.adx_period_4h = self.config.get('adx_period_4h', 14)

        # 30m parameters
        self.cmf_period_30m = self.config.get('cmf_period_30m', 20)
        self.obv_ma_period_30m = self.config.get('obv_ma_period_30m', 10)
        self.mfi_period_30m = self.config.get('mfi_period_30m', 14)
        self.force_ema_span_30m = self.config.get('force_ema_span_30m', 2)
        self.vwap_period_30m = self.config.get('vwap_period_30m', 20)

        # Enhanced parameters
        self.regime_window = self.config.get('regime_window', 20)
        self.volume_zones_lookback = self.config.get('volume_zones_lookback', 50)
        self.swing_threshold = self.config.get('swing_threshold', 0.5)

        # Feature selection settings
        self.use_feature_selection = self.config.get('use_feature_selection', True)
        self.top_features_count = self.config.get('top_features_count', 50)

    @exception_handler(reraise=True)
    @memory_usage_decorator(threshold_gb=12)
    def process_data_3way(self, df_30m: pd.DataFrame, df_4h: pd.DataFrame,
                          df_daily: pd.DataFrame, df_oi: Optional[pd.DataFrame] = None,
                          df_funding: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Process data with more aggressive memory management.

        Args:
            df_30m: DataFrame with 30-minute OHLCV data
            df_4h: DataFrame with 4-hour OHLCV data
            df_daily: DataFrame with daily OHLCV data
            df_oi: Optional DataFrame with open interest data
            df_funding: Optional DataFrame with funding rate data

        Returns:
            DataFrame with all computed features
        """
        # Log memory at start
        log_memory_usage()

        # Ensure dataframes have consistent dtypes
        # Convert DataFrames to float32 to reduce memory usage
        for df in [df_30m, df_4h, df_daily]:
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype(np.float32)

        # Ensure all dataframes have datetime index with no timezone for consistent alignment
        for df in [df_30m, df_4h, df_daily]:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            # Remove timezone info to avoid alignment issues
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

        # Get base features with reduced memory footprint
        try:
            self.logger.info("Computing 30m features")
            feat_30m = self._compute_indicators_30m(df_30m).add_prefix('m30_')
            feat_30m[['open', 'high', 'low', 'close', 'volume']] = df_30m[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            self.logger.error(f"Error computing 30m features: {str(e)}")
            feat_30m = pd.DataFrame(index=df_30m.index)
            # Add minimum required columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                feat_30m[col] = df_30m[col]

        # Add open interest features if available
        if df_oi is not None:
            try:
                self.logger.info("Adding open interest features")
                oi_features = self._compute_open_interest_features(df_oi, df_30m.index).add_prefix('oi_')
                feat_30m = pd.concat([feat_30m, oi_features], axis=1)
            except Exception as e:
                self.logger.error(f"Error adding open interest features: {str(e)}")
                # Continue without OI features

        # Add funding rate features if available
        if df_funding is not None:
            try:
                self.logger.info("Adding funding rate features")
                funding_features = self._compute_funding_features(df_funding, df_30m.index).add_prefix('funding_')
                feat_30m = pd.concat([feat_30m, funding_features], axis=1)
            except Exception as e:
                self.logger.error(f"Error adding funding rate features: {str(e)}")
                # Continue without funding features

        # Clear unused DataFrames
        del df_30m
        gc.collect()

        try:
            self.logger.info("Computing 4h features")
            feat_4h = self._compute_indicators_4h(df_4h).add_prefix('h4_')

            # Try simple reindex first
            try:
                feat_4h_ff = feat_4h.reindex(feat_30m.index, method='ffill')
            except Exception:
                # Fall back to custom alignment
                feat_4h_ff = self._align_timeframes(feat_4h, feat_30m.index)
        except Exception as e:
            self.logger.error(f"Error computing or aligning 4h features: {str(e)}")
            feat_4h_ff = pd.DataFrame(index=feat_30m.index)

        # Clear unused DataFrames
        del df_4h, feat_4h
        gc.collect()

        try:
            self.logger.info("Computing daily features")
            feat_daily = self._compute_indicators_daily(df_daily).add_prefix('d1_')

            # Try simple reindex first
            try:
                feat_daily_ff = feat_daily.reindex(feat_30m.index, method='ffill')
            except Exception:
                # Fall back to custom alignment
                feat_daily_ff = self._align_timeframes(feat_daily, feat_30m.index)
        except Exception as e:
            self.logger.error(f"Error computing or aligning daily features: {str(e)}")
            feat_daily_ff = pd.DataFrame(index=feat_30m.index)

        # Clear unused DataFrames
        del df_daily, feat_daily
        gc.collect()

        # Use a more memory-efficient concat
        self.logger.info("Combining features")

        # List of DataFrames to concatenate
        dfs_to_concat = [feat_30m]

        # Only add non-empty DataFrames
        if not feat_4h_ff.empty and len(feat_4h_ff.columns) > 0:
            dfs_to_concat.append(feat_4h_ff)
        if not feat_daily_ff.empty and len(feat_daily_ff.columns) > 0:
            dfs_to_concat.append(feat_daily_ff)

        # Concatenate available DataFrames
        combined = pd.concat(dfs_to_concat, axis=1, copy=False)

        # Drop NaN values if needed
        combined.dropna(inplace=True)

        # Clear unused DataFrames
        del feat_30m, feat_4h_ff, feat_daily_ff
        gc.collect()

        # Check memory usage
        log_memory_usage()

        self.logger.info(f"Feature engineering complete: {combined.shape} features created")

        return combined

    def _align_timeframes(self, higher_tf_data: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Align higher timeframe data to target timeframe index.

        Args:
            higher_tf_data: Higher timeframe DataFrame
            target_index: Target timeframe index

        Returns:
            DataFrame aligned to target timeframe
        """
        if higher_tf_data.empty or len(target_index) == 0:
            return pd.DataFrame(index=target_index)

        # Create a DataFrame with the target index
        aligned_data = pd.DataFrame(index=target_index)

        # For each higher timeframe row
        for i, (idx, row) in enumerate(higher_tf_data.iterrows()):
            # Handle the last candle
            if i == len(higher_tf_data) - 1:
                # For the last candle, include all remaining target indices
                mask = target_index >= idx
            else:
                # Get the next timestamp
                next_idx = higher_tf_data.index[i + 1]
                # Find target indices between this candle and the next
                # Use element-wise comparison with scalar values
                mask = (target_index >= idx) & (target_index < next_idx)

            # Skip if mask is empty
            if not any(mask):
                continue

            # Set values for all columns
            for col in higher_tf_data.columns:
                if col not in aligned_data:
                    aligned_data[col] = np.nan
                aligned_data.loc[mask, col] = row[col]

        # Fill any remaining NaN values with method='ffill'
        aligned_data = aligned_data.fillna(method='ffill')

        return aligned_data

    @exception_handler(reraise=True)
    def process_data_in_chunks(self, df_30m: pd.DataFrame, df_4h: pd.DataFrame,
                               df_daily: pd.DataFrame, chunk_size: int = 2000,
                               df_oi: Optional[pd.DataFrame] = None,
                               df_funding: Optional[pd.DataFrame] = None,
                               use_parallel: bool = False,
                               max_workers: Optional[int] = None) -> pd.DataFrame:
        """Process data in chunks with better handling of derived timeframes.

        Args:
            df_30m: DataFrame with 30-minute OHLCV data
            df_4h: DataFrame with 4-hour OHLCV data
            df_daily: DataFrame with daily OHLCV data
            chunk_size: Size of chunks to process
            df_oi: Optional DataFrame with open interest data
            df_funding: Optional DataFrame with funding rate data
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of worker processes

        Returns:
            DataFrame with all computed features
        """
        self.logger.info(f"Processing data in chunks of size {chunk_size}")

        # Get the total number of chunks
        num_chunks = (len(df_30m) + chunk_size - 1) // chunk_size
        self.logger.info(f"Total chunks: {num_chunks}")

        if use_parallel and num_chunks > 1:
            return self._process_chunks_parallel(
                df_30m, df_4h, df_daily, chunk_size, df_oi, df_funding, max_workers
            )
        else:
            return self._process_chunks_sequential(
                df_30m, df_4h, df_daily, chunk_size, df_oi, df_funding
            )

    def _process_chunks_sequential(self, df_30m: pd.DataFrame, df_4h: pd.DataFrame,
                                   df_daily: pd.DataFrame, chunk_size: int,
                                   df_oi: Optional[pd.DataFrame] = None,
                                   df_funding: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Process data chunks sequentially.

        Args:
            df_30m: DataFrame with 30-minute OHLCV data
            df_4h: DataFrame with 4-hour OHLCV data
            df_daily: DataFrame with daily OHLCV data
            chunk_size: Size of chunks to process
            df_oi: Optional DataFrame with open interest data
            df_funding: Optional DataFrame with funding rate data

        Returns:
            DataFrame with all computed features
        """
        results = []

        for i in range(0, len(df_30m), chunk_size):
            self.logger.info(f"Processing chunk {i // chunk_size + 1}")
            log_memory_usage()

            # Extract chunk
            end_idx = min(i + chunk_size, len(df_30m))
            chunk_30m = df_30m.iloc[i:end_idx]

            # Find corresponding indices in derived timeframes
            start_time = chunk_30m.index[0]
            end_time = chunk_30m.index[-1]

            # For derived timeframes, we need to be a bit more careful with the selection
            chunk_4h = df_4h[(df_4h.index <= end_time)]
            if not chunk_4h.empty:
                # Make sure we have at least one candle before start_time for proper alignment
                if chunk_4h.index[0] > start_time and len(df_4h) > len(chunk_4h):
                    # Find the last candle before our chunk starts
                    prev_candles = df_4h[df_4h.index < start_time]
                    if not prev_candles.empty:
                        prev_idx = prev_candles.index[-1]
                        # Add it to our chunk
                        chunk_4h = pd.concat([df_4h.loc[[prev_idx]], chunk_4h])

            chunk_daily = df_daily[(df_daily.index <= end_time)]
            if not chunk_daily.empty:
                # Same logic for daily data
                if chunk_daily.index[0] > start_time and len(df_daily) > len(chunk_daily):
                    prev_candles = df_daily[df_daily.index < start_time]
                    if not prev_candles.empty:
                        prev_idx = prev_candles.index[-1]
                        chunk_daily = pd.concat([df_daily.loc[[prev_idx]], chunk_daily])

            # Extract chunks of additional data if provided
            chunk_oi = None
            if df_oi is not None:
                chunk_oi = df_oi[(df_oi.index >= start_time) & (df_oi.index <= end_time)]

            chunk_funding = None
            if df_funding is not None:
                chunk_funding = df_funding[(df_funding.index >= start_time) & (df_funding.index <= end_time)]

            # Process chunk
            chunk_features = self.process_data_3way(
                chunk_30m, chunk_4h, chunk_daily, chunk_oi, chunk_funding
            )

            results.append(chunk_features)

            # Clear memory
            del chunk_30m, chunk_4h, chunk_daily, chunk_features
            if chunk_oi is not None:
                del chunk_oi
            if chunk_funding is not None:
                del chunk_funding
            gc.collect()

        # Combine results
        self.logger.info("Combining results from all chunks")
        log_memory_usage()
        combined = pd.concat(results, copy=False)

        del results
        gc.collect()

        return combined

    def _process_chunks_parallel(self, df_30m: pd.DataFrame, df_4h: pd.DataFrame,
                                 df_daily: pd.DataFrame, chunk_size: int,
                                 df_oi: Optional[pd.DataFrame] = None,
                                 df_funding: Optional[pd.DataFrame] = None,
                                 max_workers: Optional[int] = None) -> pd.DataFrame:
        """Process data chunks in parallel.

        Args:
            df_30m: DataFrame with 30-minute OHLCV data
            df_4h: DataFrame with 4-hour OHLCV data
            df_daily: DataFrame with daily OHLCV data
            chunk_size: Size of chunks to process
            df_oi: Optional DataFrame with open interest data
            df_funding: Optional DataFrame with funding rate data
            max_workers: Maximum number of worker processes

        Returns:
            DataFrame with all computed features
        """
        self.logger.info(f"Processing data in parallel chunks")

        # Prepare chunks
        chunks = []
        for i in range(0, len(df_30m), chunk_size):
            end_idx = min(i + chunk_size, len(df_30m))

            # Extract chunk
            chunk_30m = df_30m.iloc[i:end_idx]

            # Find corresponding indices in derived timeframes
            start_time = chunk_30m.index[0]
            end_time = chunk_30m.index[-1]

            # Prepare chunk data
            chunk_data = {
                'start_idx': i,
                'end_idx': end_idx,
                'start_time': start_time,
                'end_time': end_time
            }

            chunks.append(chunk_data)

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(
                self._process_chunk_wrapper,
                chunks,
                [df_30m] * len(chunks),
                [df_4h] * len(chunks),
                [df_daily] * len(chunks),
                [df_oi] * len(chunks),
                [df_funding] * len(chunks)
            ))

        # Filter out None results
        chunk_results = [result for result in chunk_results if result is not None]

        # Check if we have any valid results
        if not chunk_results:
            self.logger.error("All parallel chunks failed processing. Falling back to sequential processing.")
            return self._process_chunks_sequential(df_30m, df_4h, df_daily, chunk_size, df_oi, df_funding)

        # Combine results
        self.logger.info("Combining results from all parallel chunks")
        try:
            combined = pd.concat(chunk_results, copy=False)
            return combined
        except ValueError as e:
            self.logger.error(f"Error combining chunk results: {e}")
            # If concatenation failed but we have at least one valid result, return the first one
            if len(chunk_results) > 0:
                self.logger.warning("Returning result from first successful chunk as fallback")
                return chunk_results[0]
            else:
                # Last resort - return an empty DataFrame with the right columns
                self.logger.error("No valid chunks to combine. Returning empty DataFrame")
                return pd.DataFrame()

    def _process_chunk_wrapper(self, chunk_data: Dict[str, Any], df_30m: pd.DataFrame,
                               df_4h: pd.DataFrame, df_daily: pd.DataFrame,
                               df_oi: Optional[pd.DataFrame], df_funding: Optional[pd.DataFrame]) -> Optional[
        pd.DataFrame]:
        """Wrapper for processing a single chunk in parallel.

        Args:
            chunk_data: Dictionary with chunk information
            df_30m: DataFrame with 30-minute OHLCV data
            df_4h: DataFrame with 4-hour OHLCV data
            df_daily: DataFrame with daily OHLCV data
            df_oi: Optional DataFrame with open interest data
            df_funding: Optional DataFrame with funding rate data

        Returns:
            DataFrame with features for this chunk, or None if error
        """
        try:
            # Extract chunk info
            start_idx = chunk_data['start_idx']
            end_idx = chunk_data['end_idx']
            start_time = pd.Timestamp(chunk_data['start_time'])  # Convert to pandas Timestamp
            end_time = pd.Timestamp(chunk_data['end_time'])  # Convert to pandas Timestamp

            # Extract chunk data
            chunk_30m = df_30m.iloc[start_idx:end_idx]

            # Extract corresponding timeframes
            chunk_4h = df_4h[(df_4h.index <= end_time)]
            if not chunk_4h.empty:
                if chunk_4h.index[0] > start_time and len(df_4h) > len(chunk_4h):
                    prev_candles = df_4h[df_4h.index < start_time]
                    if not prev_candles.empty:
                        prev_idx = prev_candles.index[-1]
                        chunk_4h = pd.concat([df_4h.loc[[prev_idx]], chunk_4h])

            chunk_daily = df_daily[(df_daily.index <= end_time)]
            if not chunk_daily.empty:
                if chunk_daily.index[0] > start_time and len(df_daily) > len(chunk_daily):
                    prev_candles = df_daily[df_daily.index < start_time]
                    if not prev_candles.empty:
                        prev_idx = prev_candles.index[-1]
                        chunk_daily = pd.concat([df_daily.loc[[prev_idx]], chunk_daily])

            # Extract additional data
            chunk_oi = None
            if df_oi is not None:
                chunk_oi = df_oi[(df_oi.index >= start_time) & (df_oi.index <= end_time)]

            chunk_funding = None
            if df_funding is not None:
                chunk_funding = df_funding[(df_funding.index >= start_time) & (df_funding.index <= end_time)]

            # Process chunk
            chunk_features = self.process_data_3way(
                chunk_30m, chunk_4h, chunk_daily, chunk_oi, chunk_funding
            )

            return chunk_features

        except Exception as e:
            # Log error but don't crash the entire process
            print(f"Error processing chunk {start_idx}-{end_idx}: {str(e)}")
            return None

    def select_top_features(self, df: pd.DataFrame, n_top: int = 50) -> List[str]:
        """Select top features based on importance.

        Args:
            df: DataFrame with features
            n_top: Number of top features to select

        Returns:
            List of top feature names
        """
        self.logger.info(f"Selecting top {n_top} features")

        # Sample data for faster feature selection
        sample_size = min(5000, len(df))
        horizon = 48  # Must match the horizon in _create_labels_for_selection

        if sample_size <= horizon:
            self.logger.warning("Sample size too small for the given horizon.")
            # Return all features if sample too small
            return [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume',
                                                             'market_regime', 'volatility_regime', 'trend_strength',
                                                             'swing_high', 'swing_low']]

        df_sample = df.iloc[:sample_size]

        # Create feature matrix
        X = df_sample.iloc[:sample_size - horizon].drop(columns=['open', 'high', 'low', 'close', 'volume',
                                                                 'market_regime', 'volatility_regime', 'trend_strength',
                                                                 'swing_high', 'swing_low'],
                                                        errors='ignore')

        # Create labels for feature selection
        y = self._create_labels_for_selection(df_sample, horizon)

        # Train a Random Forest to get feature importances
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
        rf.fit(X, y)

        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Get top features
        top_features = X.columns[indices[:n_top]].tolist()

        self.logger.info(f"Top 10 features: {', '.join(top_features[:10])}")

        return top_features

    def _create_labels_for_selection(self, df: pd.DataFrame, horizon: int = 48) -> np.ndarray:
        """Create labels for feature selection.

        Args:
            df: DataFrame with OHLCV data
            horizon: Forecast horizon in candles

        Returns:
            NumPy array with labels
        """
        price = df['close']
        fwd_return = (price.shift(-horizon) / price - 1).iloc[:-horizon]

        # Use adaptive thresholds based on volatility
        if 'd1_ATR_14' in df.columns:
            atr_pct = (df['d1_ATR_14'] / df['close']).iloc[:-horizon]
        else:
            # Calculate ATR on the fly as fallback
            high_low = df['high'] - df['low']
            high_close_prev = (df['high'] - df['close'].shift(1)).abs()
            low_close_prev = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            atr_pct = (atr / df['close']).iloc[:-horizon]

        # Create thresholds
        thresholds = [-2 * atr_pct, -0.5 * atr_pct, 0.5 * atr_pct, 2 * atr_pct]

        # Digitize the forward returns into 5 classes
        labels = np.zeros(len(fwd_return), dtype=int)

        for i in range(len(fwd_return)):
            ret = fwd_return.iloc[i]
            thresh = [thresholds[j].iloc[i] for j in range(4)]

            if ret < thresh[0]:
                labels[i] = 0  # Strong bearish
            elif ret < thresh[1]:
                labels[i] = 1  # Moderate bearish
            elif ret < thresh[2]:
                labels[i] = 2  # Neutral
            elif ret < thresh[3]:
                labels[i] = 3  # Moderate bullish
            else:
                labels[i] = 4  # Strong bullish

        return labels

    def _compute_indicators_30m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators for 30-minute data.

        Args:
            df: DataFrame with 30-minute OHLCV data

        Returns:
            DataFrame with computed indicators
        """
        out = pd.DataFrame(index=df.index)

        # Bollinger Bands
        mid = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std(ddof=0)
        out['BB_middle'] = mid
        out['BB_upper'] = mid + 2 * std
        out['BB_lower'] = mid - 2 * std
        out['BB_width'] = (out['BB_upper'] - out['BB_lower']) / mid

        # Historical Volatility
        out['hist_vol_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252 * 48)  # Annualized

        # Chaikin Money Flow
        multiplier = np.where(df['high'] != df['low'],
                              ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']),
                              0).astype(np.float32)
        money_flow_volume = multiplier * df['volume']
        out[f'CMF_{self.cmf_period_30m}'] = (
                money_flow_volume.rolling(self.cmf_period_30m).sum() /
                df['volume'].rolling(self.cmf_period_30m).sum()
        ).astype(np.float32)

        # OBV
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        out['OBV'] = obv
        out[f'OBV_SMA_{self.obv_ma_period_30m}'] = obv.rolling(self.obv_ma_period_30m).mean()
        out['OBV_change'] = obv.pct_change(5)

        # MFI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        price_diff = typical_price.diff()

        # More vectorized approach
        pos_mask = price_diff > 0
        neg_mask = price_diff < 0
        pos_sum = pd.Series(np.zeros_like(raw_money_flow), index=df.index)
        neg_sum = pd.Series(np.zeros_like(raw_money_flow), index=df.index)

        pos_sum[pos_mask] = raw_money_flow[pos_mask]
        neg_sum[neg_mask] = raw_money_flow[neg_mask]

        pos_sum = pos_sum.rolling(self.mfi_period_30m).sum()
        neg_sum = neg_sum.rolling(self.mfi_period_30m).sum()

        money_flow_ratio = np.where(neg_sum != 0, pos_sum / neg_sum, 100)
        out[f'MFI_{self.mfi_period_30m}'] = (100 - (100 / (1 + money_flow_ratio))).astype(np.float32)

        # Force Index
        force_index_1 = (df['close'] - df['close'].shift(1)) * df['volume']
        out[f'ForceIndex_EMA{self.force_ema_span_30m}'] = force_index_1.ewm(
            span=self.force_ema_span_30m, adjust=False
        ).mean().astype(np.float32)

        # VWAP
        out[f'VWAP_{self.vwap_period_30m}'] = (
                (df['close'] * df['volume']).rolling(self.vwap_period_30m).sum() /
                df['volume'].rolling(self.vwap_period_30m).sum()
        ).astype(np.float32)

        # Relative position to VWAP
        out['VWAP_distance'] = (df['close'] / out[f'VWAP_{self.vwap_period_30m}'] - 1) * 100

        # Replace infinities with NaN
        out.replace([np.inf, -np.inf], np.nan, inplace=True)

        return out

    def _compute_indicators_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators for 4-hour data.

        Args:
            df: DataFrame with 4-hour OHLCV data

        Returns:
            DataFrame with computed indicators
        """
        out = pd.DataFrame(index=df.index)

        # Bollinger Bands
        mid = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std(ddof=0)
        out['BB_middle'] = mid
        out['BB_upper'] = mid + 2 * std
        out['BB_lower'] = mid - 2 * std
        out['BB_width'] = (out['BB_upper'] - out['BB_lower']) / mid

        # Historical Volatility
        out['hist_vol_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252 * 6)  # Annualized

        # MAs
        for period in self.ma_periods_4h:
            out[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
            out[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Calculate MA crossovers
        for fast_period, slow_period in [(20, 50), (50, 200)]:
            out[f'SMA_{fast_period}_{slow_period}_cross'] = np.where(
                out[f'SMA_{fast_period}'] > out[f'SMA_{slow_period}'], 1,
                np.where(out[f'SMA_{fast_period}'] < out[f'SMA_{slow_period}'], -1, 0)
            )

        # RSI
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(self.rsi_period_4h).mean()
        avg_loss = down.rolling(self.rsi_period_4h).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out[f'RSI_{self.rsi_period_4h}'] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = df['close'].ewm(span=self.macd_fast_4h, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow_4h, adjust=False).mean()
        macd_ = ema_fast - ema_slow
        macd_signal_ = macd_.ewm(span=self.macd_signal_4h, adjust=False).mean()
        out['MACD'] = macd_
        out['MACD_signal'] = macd_signal_
        out['MACD_hist'] = macd_ - macd_signal_
        out['MACD_hist_change'] = out['MACD_hist'].diff()

        # OBV
        out['OBV'] = (np.sign(df['close'].diff()) * df['volume']).cumsum().fillna(0)

        # MFI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        tp_diff = typical_price.diff()

        # More vectorized approach
        pos_mask = tp_diff > 0
        neg_mask = tp_diff < 0
        pos_flow = pd.Series(np.zeros_like(raw_money_flow), index=df.index)
        neg_flow = pd.Series(np.zeros_like(raw_money_flow), index=df.index)

        pos_flow[pos_mask] = raw_money_flow[pos_mask]
        neg_flow[neg_mask] = raw_money_flow[neg_mask]

        pos_sum = pos_flow.rolling(self.mfi_period_4h).sum()
        neg_sum = neg_flow.rolling(self.mfi_period_4h).sum()

        mfi_ratio = np.where(neg_sum != 0, pos_sum / neg_sum, 100)
        out[f'MFI_{self.mfi_period_4h}'] = (100 - (100 / (1 + mfi_ratio)))

        # ADX
        out['ADX'] = self._compute_adx(df, self.adx_period_4h)

        # Return and momentum measures
        out['return_pct'] = df['close'].pct_change() * 100
        out['momentum_1d'] = df['close'].pct_change(6) * 100  # 1 day (6 4h candles)
        out['momentum_3d'] = df['close'].pct_change(18) * 100  # 3 days
        out['momentum_5d'] = df['close'].pct_change(30) * 100  # 5 days

        # Replace infinities with NaN
        out.replace([np.inf, -np.inf], np.nan, inplace=True)

        return out

    def _compute_indicators_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators for daily data.

        Args:
            df: DataFrame with daily OHLCV data

        Returns:
            DataFrame with computed indicators
        """
        out = pd.DataFrame(index=df.index)

        # Bollinger Bands
        mid = df['close'].rolling(window=self.bb_period_daily).mean()
        std = df['close'].rolling(window=self.bb_period_daily).std(ddof=0)
        out['BB_middle'] = mid
        out['BB_upper'] = mid + self.bb_stddev_daily * std
        out['BB_lower'] = mid - self.bb_stddev_daily * std
        out['BB_width'] = (out['BB_upper'] - out['BB_lower']) / mid

        # Historical Volatility
        out['hist_vol_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)  # Annualized

        # MAs
        for period in self.ma_periods_daily:
            out[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
            out[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Calculate distance from MAs
        for period in self.ma_periods_daily:
            out[f'SMA_{period}_distance'] = (df['close'] / out[f'SMA_{period}'] - 1) * 100

        # RSI
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(window=self.rsi_period_daily).mean()
        avg_loss = down.rolling(window=self.rsi_period_daily).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out[f'RSI_{self.rsi_period_daily}'] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = df['close'].ewm(span=self.macd_fast_daily, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow_daily, adjust=False).mean()
        macd_ = ema_fast - ema_slow
        out['MACD'] = macd_
        out['MACD_signal'] = macd_.ewm(span=self.macd_signal_daily, adjust=False).mean()
        out['MACD_hist'] = macd_ - out['MACD_signal']

        # ATR
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift(1)).abs()
        lc = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        out[f'ATR_{self.atr_period_daily}'] = tr.rolling(window=self.atr_period_daily).mean()
        out['ATR_percent'] = out[f'ATR_{self.atr_period_daily}'] / df['close'] * 100

        # MFI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        typical_price_diff = typical_price.diff()

        # Vectorized approach
        pos_mask = typical_price_diff > 0
        neg_mask = typical_price_diff < 0
        positive_flow = pd.Series(np.zeros_like(raw_money_flow), index=df.index)
        negative_flow = pd.Series(np.zeros_like(raw_money_flow), index=df.index)

        positive_flow[pos_mask] = raw_money_flow[pos_mask]
        negative_flow[neg_mask] = raw_money_flow[neg_mask]

        pos_sum = positive_flow.rolling(window=self.mfi_period_daily).sum()
        neg_sum = negative_flow.rolling(window=self.mfi_period_daily).sum()

        out[f'MFI_{self.mfi_period_daily}'] = 100 * np.where(pos_sum + neg_sum != 0,
                                                             pos_sum / (pos_sum + neg_sum),
                                                             0.5)

        # Chaikin Money Flow
        denom = (df['high'] - df['low']).replace(0, np.nan)
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / denom
        money_flow_volume = money_flow_multiplier * df['volume']
        cmf = money_flow_volume.rolling(window=self.cmf_period_daily).sum() / df['volume'].rolling(
            window=self.cmf_period_daily).sum().replace(0, np.nan)
        out[f'CMF_{self.cmf_period_daily}'] = cmf

        # Weekly and monthly returns
        out['weekly_return'] = df['close'].pct_change(5) * 100  # Approx 1 week
        out['monthly_return'] = df['close'].pct_change(20) * 100  # Approx 1 month

        # Replace infinities with NaN
        out.replace([np.inf, -np.inf], np.nan, inplace=True)

        return out

    def _compute_open_interest_features(self, df_oi: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Compute features from open interest data.

        Args:
            df_oi: DataFrame with open interest data
            target_index: Target index to align features to

        Returns:
            DataFrame with open interest features
        """
        if df_oi is None or df_oi.empty:
            return pd.DataFrame(index=target_index)

        try:
            out = pd.DataFrame(index=df_oi.index)

            # Basic OI features
            if 'openInterest' in df_oi.columns:
                out['openInterest'] = df_oi['openInterest']
                out['openInterest_change'] = df_oi['openInterest'].pct_change()
                out['openInterest_sma_12'] = df_oi['openInterest'].rolling(12).mean()  # 6 hour SMA for 30m data
                out['openInterest_sma_48'] = df_oi['openInterest'].rolling(48).mean()  # 24 hour SMA
                out['openInterest_ratio'] = out['openInterest'] / out['openInterest_sma_48']
            elif 'sumOpenInterest' in df_oi.columns:
                # Use alternative column if available
                out['openInterest'] = df_oi['sumOpenInterest']
                out['openInterest_change'] = df_oi['sumOpenInterest'].pct_change()
                out['openInterest_sma_12'] = df_oi['sumOpenInterest'].rolling(12).mean()
                out['openInterest_sma_48'] = df_oi['sumOpenInterest'].rolling(48).mean()
                out['openInterest_ratio'] = out['openInterest'] / out['openInterest_sma_48']

            # Ensure indices are datetime and no timezone
            if not isinstance(out.index, pd.DatetimeIndex):
                out.index = pd.to_datetime(out.index)
            if out.index.tz is not None:
                out.index = out.index.tz_localize(None)

            if not isinstance(target_index, pd.DatetimeIndex):
                target_index = pd.to_datetime(target_index)
            if target_index.tz is not None:
                target_index = target_index.tz_localize(None)

            # Try a safer alignment approach
            try:
                # Try a simpler reindex approach first
                aligned = out.reindex(target_index, method='ffill')
                return aligned
            except Exception as e:
                # Fall back to custom alignment method
                try:
                    aligned = self._align_timeframes(out, target_index)
                    return aligned
                except Exception as inner_e:
                    # If both methods fail, return empty DataFrame
                    self.logger.error(f"Error aligning open interest data: {str(inner_e)}")
                    return pd.DataFrame(index=target_index)

        except Exception as e:
            self.logger.error(f"Error computing open interest features: {str(e)}")
            return pd.DataFrame(index=target_index)

    def _compute_funding_features(self, df_funding: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Compute features from funding rate data.

        Args:
            df_funding: DataFrame with funding rate data
            target_index: Target index to align features to

        Returns:
            DataFrame with funding rate features
        """
        if df_funding is None or df_funding.empty:
            return pd.DataFrame(index=target_index)

        out = pd.DataFrame(index=df_funding.index)

        # Basic funding rate features
        if 'fundingRate' in df_funding.columns:
            out['fundingRate'] = df_funding['fundingRate']
            out['fundingRate_sma_3'] = df_funding['fundingRate'].rolling(3).mean()  # Average of last 3 funding payments
            out['fundingRate_cumulative'] = df_funding['fundingRate'].rolling(
                8).sum()  # Last 8 (~8 hours * 8 = ~2.7 days)

        # Align to target index
        aligned = self._align_timeframes(out, target_index)
        return aligned

    def _compute_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average Directional Index.

        Args:
            df: DataFrame with OHLCV data
            period: Period for ADX calculation

        Returns:
            Series with ADX values
        """
        # Vectorized calculation of DM
        high_diff = df['high'].diff()
        low_diff = df['low'].diff().multiply(-1)

        # Calculate True Range
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': (df['high'] - df['close'].shift(1)).abs(),
            'lc': (df['low'] - df['close'].shift(1)).abs()
        }).max(axis=1)

        # Calculate +DM and -DM
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # Convert to indicator (smoothed)
        tr_period = tr.rolling(window=period).sum()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).sum() / tr_period
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).sum() / tr_period

        # Calculate ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(window=period).mean()

        return adx

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with scaled features
        """
        # Copy dataframe
        df_scaled = df.copy()

        # Find numeric columns (exclude specified columns)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'market_regime',
                        'swing_high', 'swing_low', 'volatility_regime']

        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_cols)

        # Create scaler
        scaler = StandardScaler()

        # Fit and transform
        if len(numeric_cols) > 0:
            df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df_scaled

    def _compute_market_regime(self, df: pd.DataFrame) -> np.ndarray:
        """Detect market regime (trending vs ranging).

        Args:
            df: DataFrame with features

        Returns:
            NumPy array with market regime values
        """
        # Use ADX for trend strength
        if 'h4_ADX' in df.columns:
            adx = df['h4_ADX'].fillna(0).values
        else:
            # Fallback if ADX not available
            adx = np.zeros(len(df))

        # Directional movement
        if 'h4_SMA_20' in df.columns and 'h4_SMA_50' in df.columns:
            plus_di = df['h4_SMA_20'].diff().fillna(0).values
            minus_di = df['h4_SMA_50'].diff().fillna(0).values
        else:
            # Fallback
            plus_di = np.zeros(len(df))
            minus_di = np.zeros(len(df))

        # Define regime: 0=ranging, 1=uptrend, -1=downtrend
        regime = np.zeros(len(df))

        # Strong trend with ADX > 25
        trending_mask = adx > 25
        uptrend_mask = (trending_mask) & (plus_di > 0)
        downtrend_mask = (trending_mask) & (minus_di < 0)

        regime[uptrend_mask] = 1
        regime[downtrend_mask] = -1

        return regime

    def _compute_volume_zones(self, df: pd.DataFrame) -> np.ndarray:
        """Identify high-volume price zones.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            NumPy array with volume zone scores
        """
        prices = df['close'].values
        volumes = df['volume'].values

        # Create price bins (adaptive based on current price range)
        min_price = np.nanmin(prices)
        max_price = np.nanmax(prices)
        price_range = max_price - min_price
        bin_size = price_range / 10  # 10 bins

        if bin_size == 0:  # Handle edge case
            return np.zeros(len(prices))

        bins = np.arange(min_price, max_price + bin_size, bin_size)
        digitized = np.digitize(prices, bins)

        # Sum volume for each bin
        bin_volumes = np.zeros(len(bins))
        for i in range(len(prices)):
            bin_idx = digitized[i] - 1  # adjust for 0-indexing
            if 0 <= bin_idx < len(bin_volumes):
                bin_volumes[bin_idx] += volumes[i]

        # Assign volume zone score for each price point
        vol_zone = np.zeros(len(prices))
        max_bin_volume = np.max(bin_volumes) if np.max(bin_volumes) > 0 else 1

        for i in range(len(prices)):
            bin_idx = digitized[i] - 1
            if 0 <= bin_idx < len(bin_volumes):
                # Normalize by max volume
                vol_zone[i] = bin_volumes[bin_idx] / max_bin_volume

        return vol_zone

    def _detect_swing_highs(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Detect swing highs in price action.

        Args:
            df: DataFrame with OHLCV data
            threshold: Threshold for swing detection

        Returns:
            NumPy array with swing high indicators
        """
        highs = df['high'].values
        swing_highs = np.zeros(len(highs))

        # Require at least 11 candles to detect swings
        if len(highs) < 11:
            return swing_highs

        # Look for local maxima with surrounding lower prices
        for i in range(5, len(highs) - 5):
            # Check if a potential local maximum
            if np.all(highs[i] >= highs[i - 5:i]) and np.all(highs[i] >= highs[i + 1:i + 6]):
                # Measure strength by how much higher than neighbors
                left_diff = (highs[i] - np.min(highs[i - 5:i])) / highs[i]
                right_diff = (highs[i] - np.min(highs[i + 1:i + 6])) / highs[i]

                if left_diff > threshold and right_diff > threshold:
                    swing_highs[i] = 1

        return swing_highs

    def _detect_swing_lows(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Detect swing lows in price action.

        Args:
            df: DataFrame with OHLCV data
            threshold: Threshold for swing detection

        Returns:
            NumPy array with swing low indicators
        """
        lows = df['low'].values
        swing_lows = np.zeros(len(lows))

        # Require at least 11 candles to detect swings
        if len(lows) < 11:
            return swing_lows

        # Look for local minima with surrounding higher prices
        for i in range(5, len(lows) - 5):
            # Check if a potential local minimum
            if np.all(lows[i] <= lows[i - 5:i]) and np.all(lows[i] <= lows[i + 1:i + 6]):
                # Measure strength by how much lower than neighbors
                left_diff = (np.max(lows[i - 5:i]) - lows[i]) / lows[i]
                right_diff = (np.max(lows[i + 1:i + 6]) - lows[i]) / lows[i]

                if left_diff > threshold and right_diff > threshold:
                    swing_lows[i] = 1

        return swing_lows

    def _compute_volatility_regime(self, df: pd.DataFrame) -> np.ndarray:
        """Compute volatility regime.

        Args:
            df: DataFrame with features

        Returns:
            NumPy array with volatility regime values
        """
        # Use ATR relative to price to determine volatility regime
        if 'd1_ATR_14' in df.columns:
            atr = df['d1_ATR_14'].values
            close = df['close'].values
            atr_pct = np.divide(atr, close, out=np.zeros_like(atr), where=close != 0)

            # Calculate percentiles for volatility
            low_vol_threshold = np.nanpercentile(atr_pct, 25)
            high_vol_threshold = np.nanpercentile(atr_pct, 75)

            # Define regime: -1=low vol, 0=normal vol, 1=high vol
            regime = np.zeros(len(df))
            regime[atr_pct < low_vol_threshold] = -1
            regime[atr_pct > high_vol_threshold] = 1

            return regime
        else:
            # Fallback to historical volatility if ATR not available
            if 'hist_vol_20' in df.columns:
                hist_vol = df['hist_vol_20'].values

                # Calculate percentiles
                low_vol_threshold = np.nanpercentile(hist_vol, 25)
                high_vol_threshold = np.nanpercentile(hist_vol, 75)

                # Define regime
                regime = np.zeros(len(df))
                regime[hist_vol < low_vol_threshold] = -1
                regime[hist_vol > high_vol_threshold] = 1

                return regime

            # Final fallback
            return np.zeros(len(df))

    def _compute_mean_reversion(self, df: pd.DataFrame) -> np.ndarray:
        """Compute mean reversion potential.

        Args:
            df: DataFrame with features

        Returns:
            NumPy array with mean reversion potential values
        """
        close = df['close'].values

        # Use MA deviation if available
        if 'h4_SMA_20' in df.columns and 'h4_SMA_50' in df.columns:
            ma20 = df['h4_SMA_20'].values
            ma50 = df['h4_SMA_50'].values

            # Calculate z-score of price deviation from moving averages
            deviation = (close - (ma20 + ma50) / 2)
            rolling_std = pd.Series(deviation).rolling(window=20).std().values

            # Avoid division by zero
            z_score = np.zeros_like(deviation)
            mask = rolling_std > 0
            z_score[mask] = deviation[mask] / rolling_std[mask]

            return z_score

        # Fallback to BB if MAs not available
        elif 'BB_middle' in df.columns and 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            middle = df['BB_middle'].values
            upper = df['BB_upper'].values
            lower = df['BB_lower'].values

            # Calculate position within Bollinger Bands
            band_width = upper - lower
            position = np.zeros_like(close)
            mask = band_width > 0
            position[mask] = (close[mask] - middle[mask]) / (band_width[mask] / 2)

            return position

        # Final fallback
        return np.zeros(len(df))

    def _compute_trend_strength(self, df: pd.DataFrame) -> np.ndarray:
        """Compute trend strength indicator.

        Args:
            df: DataFrame with features

        Returns:
            NumPy array with trend strength values
        """
        # Use ADX if available
        if 'h4_ADX' in df.columns:
            adx = df['h4_ADX'].fillna(0).values

            # Use SMA ratio for direction
            if 'h4_SMA_20' in df.columns and 'h4_SMA_50' in df.columns:
                sma_ratio = (df['h4_SMA_20'] / df['h4_SMA_50']).fillna(1).values
            else:
                # Fallback for direction
                sma_ratio = np.ones(len(df))

            # Combine ADX and SMA ratio for trend strength
            trend_strength = np.zeros(len(df))

            for i in range(len(df)):
                if adx[i] > 30:  # Strong trend
                    # Positive when SMA20 > SMA50 (uptrend)
                    # Negative when SMA20 < SMA50 (downtrend)
                    direction = 1 if sma_ratio[i] > 1 else -1
                    strength = min(adx[i] / 100, 1)  # Normalize to -1 to 1
                    trend_strength[i] = direction * strength
                else:
                    # Weak or no trend
                    trend_strength[i] = 0

            return trend_strength

        # Fallback if ADX not available
        else:
            return np.zeros(len(df))