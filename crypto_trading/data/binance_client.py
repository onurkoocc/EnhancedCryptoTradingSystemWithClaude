"""
Binance API client for fetching cryptocurrency market data.

This module provides a client for fetching market data from the Binance API,
including OHLCV data, open interest, funding rates, and liquidation data.
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
import pandas as pd
from binance.um_futures import UMFutures
from binance.error import ClientError
from pathlib import Path

from ..utils.logging_utils import RetryWithBackoff, exception_handler


class BitcoinData:
    """Client for fetching Bitcoin data from Binance."""

    def __init__(self,
                 csv_30m: str = 'data/btc_data_30m.csv',
                 csv_4h: str = 'data/btc_data_4h.csv',
                 csv_daily: str = 'data/btc_data_daily.csv',
                 csv_oi: str = 'data/btc_open_interest.csv',
                 csv_funding: str = 'data/btc_funding_rates.csv',
                 symbol: str = 'BTCUSDT',
                 base_url: str = "https://fapi.binance.com",
                 use_testnet: bool = False,
                 timeout: int = 30,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize BitcoinData client.

        Args:
            csv_30m: Path to 30m data CSV file
            csv_4h: Path to 4h data CSV file
            csv_daily: Path to daily data CSV file
            csv_oi: Path to open interest CSV file
            csv_funding: Path to funding rates CSV file
            symbol: Trading symbol (default: 'BTCUSDT')
            base_url: Binance API base URL
            use_testnet: Whether to use Binance testnet
            timeout: API request timeout in seconds
            api_key: Binance API key (optional, falls back to env var)
            api_secret: Binance API secret (optional, falls back to env var)
            logger: Logger to use
        """
        # Store file paths
        self.csv_30m = csv_30m
        self.csv_4h = csv_4h
        self.csv_daily = csv_daily
        self.csv_oi = csv_oi
        self.csv_funding = csv_funding
        self.symbol = symbol

        # Ensure directories exist
        for csv_path in [csv_30m, csv_4h, csv_daily, csv_oi, csv_funding]:
            directory = os.path.dirname(csv_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

        # Get API credentials
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")

        # Use testnet URL if specified
        if use_testnet:
            base_url = "https://testnet.binancefuture.com"

        # Initialize UMFutures client
        self.client = UMFutures(
            key=self.api_key,
            secret=self.api_secret,
            base_url=base_url,
            timeout=timeout
        )

        # Setup logger
        self.logger = logger or logging.getLogger("BitcoinData")

    @exception_handler(reraise=True)
    def fetch_30m_data(self, live: bool = False, lookback_candles: int = 7000) -> pd.DataFrame:
        """Fetch 30-minute data from file or Binance API.

        Args:
            live: Whether to force fetch from API even if file exists
            lookback_candles: Number of candles to fetch

        Returns:
            DataFrame with 30-minute OHLCV data
        """
        # Check if file exists and we're not in live mode
        if os.path.exists(self.csv_30m) and not live:
            self.logger.info(f"Loading 30m data from {self.csv_30m}")
            return self._read_csv_with_numeric(self.csv_30m)

        self.logger.info("Fetching 30m data from Binance API")
        try:
            # Calculate start time in milliseconds
            start_time = int((datetime.now().timestamp() - (lookback_candles * 30 * 60)) * 1000)

            # Fetch all data with pagination
            all_klines = []
            while True:
                # Fetch up to 1000 candles (Binance limit)
                klines = self._fetch_klines_with_retry(
                    symbol=self.symbol,
                    interval="30m",
                    limit=1000,
                    startTime=start_time if len(all_klines) == 0 else None,
                    endTime=None
                )

                if not klines:
                    break

                all_klines.extend(klines)

                # If we got less than 1000 candles, we've reached the end
                if len(klines) < 1000:
                    break

                # Update start time for next batch
                start_time = int(klines[-1][0]) + 1

                # Check if we have enough data
                if len(all_klines) >= lookback_candles:
                    break

            # Convert to DataFrame
            df = self._process_klines_to_dataframe(all_klines)

            # Save to CSV
            df.to_csv(self.csv_30m, index=True)
            self.logger.info(f"Fetched {len(df)} 30m candles and saved to {self.csv_30m}")

            return df

        except ClientError as e:
            self.logger.error(f"Binance API error fetching 30m data: {e}")
            # If API fails but we have a CSV file, use it as fallback
            if os.path.exists(self.csv_30m):
                self.logger.info(f"Using existing {self.csv_30m} as fallback")
                return self._read_csv_with_numeric(self.csv_30m)
            raise

    @RetryWithBackoff(max_retries=3, exceptions=(ClientError, ConnectionError),
                      initial_backoff=2, backoff_multiplier=2)
    def _fetch_klines_with_retry(self, **kwargs) -> List[List]:
        """Fetch klines with retry mechanism.

        Args:
            **kwargs: Arguments to pass to the klines method

        Returns:
            List of klines data
        """
        return self.client.klines(**kwargs)

    def _process_klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """Process klines data into a DataFrame.

        Args:
            klines: List of klines data from Binance API

        Returns:
            Processed DataFrame
        """
        # Create DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Process data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

        # Calculate turnover (quote volume) for compatibility with existing code
        df['turnover'] = df['quote_asset_volume']

        # Select only needed columns
        df = df[['open', 'high', 'low', 'close', 'volume', 'turnover']]

        return df

    @exception_handler(reraise=True)
    def fetch_open_interest(self, live: bool = False, lookback_candles: int = 7000) -> pd.DataFrame:
        """Fetch open interest data from file or Binance API."""
        # Check if file exists and we're not in live mode
        if os.path.exists(self.csv_oi) and not live:
            self.logger.info(f"Loading open interest data from {self.csv_oi}")
            return pd.read_csv(self.csv_oi, index_col='timestamp', parse_dates=True)

        self.logger.info("Fetching open interest data from Binance API")
        try:
            data = []
            limit = 500  # Maximum allowed by Binance API
            end_time = int(datetime.now().timestamp() * 1000) if live else None

            # We need to make multiple API calls to get enough history
            for _ in range(int(np.ceil(lookback_candles / limit))):
                params = {
                    "symbol": self.symbol,
                    "period": "30m",  # Match our candle timeframe
                    "limit": limit
                }

                if end_time:
                    params["endTime"] = end_time

                oi_data = self._fetch_open_interest_with_retry(**params)

                if not oi_data:
                    break

                # Debug logging to see response structure
                if _ == 0:
                    self.logger.debug(f"First open interest record: {oi_data[0]}")

                data.extend(oi_data)

                if len(oi_data) < limit:
                    break

                # Use the timestamp of the last record minus 1ms as the end time for next batch
                end_time = int(oi_data[-1]['timestamp']) - 1

            # Convert to DataFrame
            if not data:
                self.logger.warning("No open interest data received from API")
                return pd.DataFrame(columns=['openInterest', 'sumOpenInterest'])

            df = pd.DataFrame(data)

            # Debug logging to see available columns
            self.logger.debug(f"Open interest data columns: {df.columns.tolist()}")

            # Check if we have the expected columns and handle missing columns
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            else:
                self.logger.error("Missing 'timestamp' column in API response")
                return pd.DataFrame(columns=['openInterest', 'sumOpenInterest'])

            # Convert numeric columns if they exist
            if 'openInterest' in df.columns:
                df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce').astype(np.float32)
            else:
                self.logger.warning("Missing 'openInterest' column in API response")
                # Check for alternative column names
                possible_columns = ['openInterest', 'open_interest', 'oi', 'sumOpenInterest']
                for col in possible_columns:
                    if col in df.columns:
                        self.logger.info(f"Using alternative column '{col}' for open interest")
                        df['openInterest'] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
                        break
                else:
                    # No suitable column found, create empty one
                    df['openInterest'] = np.nan

            if 'sumOpenInterest' in df.columns:
                df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce').astype(np.float32)
            else:
                df['sumOpenInterest'] = np.nan

            # Save to CSV
            df.to_csv(self.csv_oi, index=True)
            self.logger.info(f"Fetched {len(df)} open interest records and saved to {self.csv_oi}")

            return df

        except Exception as e:
            self.logger.error(f"Binance API error fetching open interest data: {str(e)}")
            # If API fails but we have a CSV file, use it as fallback
            if os.path.exists(self.csv_oi):
                self.logger.info(f"Using existing {self.csv_oi} as fallback")
                return pd.read_csv(self.csv_oi, index_col='timestamp', parse_dates=True)
            # Otherwise return an empty DataFrame with expected columns
            return pd.DataFrame(columns=['openInterest', 'sumOpenInterest'])

    @RetryWithBackoff(max_retries=3, exceptions=(ClientError, ConnectionError))
    def _fetch_open_interest_with_retry(self, **kwargs) -> List[Dict[str, Any]]:
        """Fetch open interest data with retry mechanism.

        Args:
            **kwargs: Arguments to pass to the open_interest_hist method

        Returns:
            List of open interest data
        """
        return self.client.open_interest_hist(**kwargs)

    @exception_handler(reraise=True)
    def fetch_funding_rates(self, live: bool = False, lookback_candles: int = 7000) -> pd.DataFrame:
        """Fetch funding rate history from file or Binance API.

        Args:
            live: Whether to force fetch from API even if file exists
            lookback_candles: Number of data points to fetch

        Returns:
            DataFrame with funding rate data
        """
        # Check if file exists and we're not in live mode
        if os.path.exists(self.csv_funding) and not live:
            self.logger.info(f"Loading funding rate data from {self.csv_funding}")
            return pd.read_csv(self.csv_funding, index_col='timestamp', parse_dates=True)

        self.logger.info("Fetching funding rate data from Binance API")
        try:
            data = []
            limit = 1000  # Maximum allowed by Binance API
            start_time = None

            # Funding rates occur every 8 hours, so we need to fetch more history
            # to match our candle data timeframe
            target_count = lookback_candles // 16  # Each funding rate covers 16 30m periods

            # We may need multiple API calls to get enough history
            while len(data) < target_count:
                params = {
                    "symbol": self.symbol,
                    "limit": limit
                }

                if start_time:
                    params["startTime"] = start_time

                funding_data = self._fetch_funding_rates_with_retry(**params)

                if not funding_data:
                    break

                data.extend(funding_data)

                if len(funding_data) < limit:
                    break

                # Use the timestamp of the last record plus 1ms as the start time for next batch
                start_time = int(funding_data[-1]['fundingTime']) + 1

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce').astype(np.float32)
            df.set_index('timestamp', inplace=True)

            # Save to CSV
            df.to_csv(self.csv_funding, index=True)
            self.logger.info(f"Fetched {len(df)} funding rate records and saved to {self.csv_funding}")

            return df

        except ClientError as e:
            self.logger.error(f"Binance API error fetching funding rate data: {e}")
            # If API fails but we have a CSV file, use it as fallback
            if os.path.exists(self.csv_funding):
                self.logger.info(f"Using existing {self.csv_funding} as fallback")
                return pd.read_csv(self.csv_funding, index_col='timestamp', parse_dates=True)
            raise

    @RetryWithBackoff(max_retries=3, exceptions=(ClientError, ConnectionError))
    def _fetch_funding_rates_with_retry(self, **kwargs) -> List[Dict[str, Any]]:
        """Fetch funding rate data with retry mechanism.

        Args:
            **kwargs: Arguments to pass to the funding rate method

        Returns:
            List of funding rate data
        """
        # Try the correct method name according to Binance API documentation
        if hasattr(self.client, 'get_funding_rate_history'):
            return self.client.get_funding_rate_history(**kwargs)
        elif hasattr(self.client, 'get_funding_rate'):
            return self.client.get_funding_rate(**kwargs)
        elif hasattr(self.client, 'funding_rate'):
            return self.client.funding_rate(**kwargs)
        else:
            self.logger.warning("Could not find funding rate method in Binance client, using fallback")
            # Fallback to another available method
            try:
                # Check what methods are available for future reference
                available_methods = [method for method in dir(self.client) if 'fund' in method.lower()]
                self.logger.info(f"Available funding-related methods: {available_methods}")

                # Try premium_index which often contains funding rate data
                if hasattr(self.client, 'mark_price'):
                    return self.client.mark_price(**kwargs)
                return []
            except Exception as e:
                self.logger.warning(f"Fallback funding rate method failed: {e}")
                return []

    @exception_handler(reraise=False, fallback_value=pd.DataFrame())
    def fetch_liquidation_data(self, live: bool = False, limit: int = 500) -> pd.DataFrame:
        """Fetch recent liquidation events from Binance API.

        Args:
            live: Whether to force fetch from API
            limit: Maximum number of liquidation events to fetch

        Returns:
            DataFrame with liquidation data, or empty DataFrame if error
        """
        self.logger.info("Fetching liquidation data from Binance API")

        # Check if API keys are available and valid
        if not self.api_key or len(self.api_key) < 10 or not self.api_secret or len(self.api_secret) < 10:
            self.logger.warning("Valid API key and secret are required for fetching liquidation data")
            return pd.DataFrame()

        try:
            # Get liquidation orders - this endpoint may require API key authentication
            liquidation_data = self._fetch_liquidations_with_retry(
                symbol=self.symbol,
                limit=limit
            )

            # Convert to DataFrame
            if not liquidation_data:
                return pd.DataFrame()

            df = pd.DataFrame(liquidation_data)

            if 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
                df.set_index('timestamp', inplace=True)

            # Convert numeric columns
            numeric_cols = ['price', 'origQty', 'executedQty', 'averagePrice']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

            return df

        except ClientError as e:
            self.logger.error(f"Binance API error fetching liquidation data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    @RetryWithBackoff(max_retries=3, exceptions=(ClientError, ConnectionError))
    def _fetch_liquidations_with_retry(self, **kwargs) -> List[Dict[str, Any]]:
        """Fetch liquidation data with retry mechanism.

        Args:
            **kwargs: Arguments to pass to the force_orders method

        Returns:
            List of liquidation data
        """
        # This endpoint may require different methods based on the Binance client version
        if hasattr(self.client, 'force_orders'):
            return self.client.force_orders(**kwargs)
        elif hasattr(self.client, 'get_force_orders'):
            return self.client.get_force_orders(**kwargs)
        elif hasattr(self.client, 'get_all_liquidation_orders'):
            return self.client.get_all_liquidation_orders(**kwargs)
        else:
            self.logger.warning("Could not find liquidation order method in Binance client")
            return []

    @exception_handler(reraise=True)
    def derive_4h_data(self, df_30m: pd.DataFrame) -> pd.DataFrame:
        """Derive 4-hour data from 30-minute data using resampling.

        Args:
            df_30m: DataFrame with 30-minute OHLCV data

        Returns:
            DataFrame with 4-hour OHLCV data
        """
        # Create a copy to avoid modifying the original DataFrame
        df_30m_copy = df_30m.copy()

        # Ensure the index is datetime and sorted
        if not isinstance(df_30m_copy.index, pd.DatetimeIndex):
            df_30m_copy.index = pd.to_datetime(df_30m_copy.index)
        df_30m_copy = df_30m_copy.sort_index()

        # Resample to 4-hour timeframe
        df_4h = df_30m_copy.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'turnover': 'sum'
        })

        # Drop any NaN rows that might have been created during resampling
        df_4h = df_4h.dropna()

        # Optionally save to CSV
        df_4h.to_csv(self.csv_4h, index=True)
        self.logger.info(f"Derived {len(df_4h)} 4h candles and saved to {self.csv_4h}")

        return df_4h

    @exception_handler(reraise=True)
    def derive_daily_data(self, df_30m: pd.DataFrame) -> pd.DataFrame:
        """Derive daily data from 30-minute data using resampling.

        Args:
            df_30m: DataFrame with 30-minute OHLCV data

        Returns:
            DataFrame with daily OHLCV data
        """
        # Create a copy to avoid modifying the original DataFrame
        df_30m_copy = df_30m.copy()

        # Ensure the index is datetime and sorted
        if not isinstance(df_30m_copy.index, pd.DatetimeIndex):
            df_30m_copy.index = pd.to_datetime(df_30m_copy.index)
        df_30m_copy = df_30m_copy.sort_index()

        # Resample to daily timeframe - using 'D' for calendar day
        df_daily = df_30m_copy.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'turnover': 'sum'
        })

        # Drop any NaN rows that might have been created during resampling
        df_daily = df_daily.dropna()

        # Optionally save to CSV
        df_daily.to_csv(self.csv_daily, index=True)
        self.logger.info(f"Derived {len(df_daily)} daily candles and saved to {self.csv_daily}")

        return df_daily

    def _read_csv_with_numeric(self, filepath: str) -> pd.DataFrame:
        """Read CSV file and ensure numeric columns are properly typed.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with properly typed numeric columns
        """
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

        return df

    def fetch_all_data(self, live: bool = False, lookback_candles: int = 7000) -> Dict[str, pd.DataFrame]:
        """Fetch all needed data at once."""
        self.logger.info("Fetching all market data")

        # Fetch 30m data (required)
        try:
            df_30m = self.fetch_30m_data(live=live, lookback_candles=lookback_candles)
        except Exception as e:
            self.logger.error(f"Critical error fetching 30m data: {e}")
            raise  # This is essential data, so we re-raise the exception

        # Derive 4h and daily data (required)
        try:
            df_4h = self.derive_4h_data(df_30m)
            df_daily = self.derive_daily_data(df_30m)
        except Exception as e:
            self.logger.error(f"Error deriving timeframe data: {e}")
            # Try to create empty DataFrames with expected columns
            df_4h = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'turnover'])
            df_daily = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'turnover'])

        # Fetch additional data (optional)
        df_oi = pd.DataFrame()
        df_funding = pd.DataFrame()
        df_liquidations = pd.DataFrame()

        try:
            df_oi = self.fetch_open_interest(live=live, lookback_candles=lookback_candles)
        except Exception as e:
            self.logger.warning(f"Error fetching open interest data: {e}")

        try:
            df_funding = self.fetch_funding_rates(live=live, lookback_candles=lookback_candles)
        except Exception as e:
            self.logger.warning(f"Error fetching funding rate data: {e}")

        try:
            df_liquidations = self.fetch_liquidation_data(live=live)
        except Exception as e:
            self.logger.warning(f"Error fetching liquidation data: {e}")

        return {
            '30m': df_30m,
            '4h': df_4h,
            'daily': df_daily,
            'open_interest': df_oi,
            'funding_rates': df_funding,
            'liquidations': df_liquidations
        }