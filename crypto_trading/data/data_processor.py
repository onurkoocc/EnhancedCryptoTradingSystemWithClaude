"""
Data preparation for machine learning models.

This module provides classes and functions for preparing cryptocurrency
market data for use in machine learning models, including sequence building
and label creation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
import tensorflow as tf
import gc

from ..utils.logging_utils import exception_handler
from ..utils.memory_monitor import memory_usage_decorator, log_memory_usage


class CryptoDataPreparer:
    """Prepares crypto data for machine learning models."""

    def __init__(self, sequence_length: int = 144, horizon: int = 48,
                 normalize_method: str = 'zscore', price_column: str = 'close',
                 train_ratio: float = 0.8, add_noise: bool = True,
                 noise_level: float = 0.01, use_tf_dataset: bool = True,
                 logger: Optional[logging.Logger] = None):
        """Initialize data preparer.

        Args:
            sequence_length: Length of input sequences (lookback period)
            horizon: Forecast horizon in candles
            normalize_method: Normalization method ('zscore', 'minmax', or None)
            price_column: Column to use for price data
            train_ratio: Ratio of data to use for training vs validation
            add_noise: Whether to add Gaussian noise to training data
            noise_level: Level of Gaussian noise to add
            use_tf_dataset: Whether to return TensorFlow datasets
            logger: Logger to use
        """
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.price_column = price_column
        self.train_ratio = train_ratio
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.use_tf_dataset = use_tf_dataset
        self.logger = logger or logging.getLogger('DataPreparer')
        self.scaler = None

    @exception_handler(reraise=True)
    @memory_usage_decorator(threshold_gb=12)
    def prepare_data(self, df: pd.DataFrame, batch_size: int = 256) -> Tuple:
        """Prepare data for training and validation.

        Args:
            df: DataFrame with features
            batch_size: Batch size for TensorFlow datasets

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, df_val, fwd_returns_val)
            or (train_dataset, val_dataset, df_val, fwd_returns_val) if use_tf_dataset=True
        """
        # Drop NaN values
        df = df.dropna().copy()

        # Check if we have enough data
        if len(df) < (self.sequence_length + self.horizon):
            self.logger.warning(
                f"Insufficient data: {len(df)} < {self.sequence_length + self.horizon}"
            )
            if self.use_tf_dataset:
                return None, None, df, np.array([])
            else:
                return np.array([]), np.array([]), np.array([]), np.array([]), df, np.array([])

        # Create labels and extract feature matrix
        df_labeled, labels, fwd_returns_full = self._create_labels(df)
        data_array = df_labeled.values.astype(np.float32)

        # Build sequences
        X_full, y_full, fwd_returns_full = self._build_sequences(data_array, labels, fwd_returns_full)

        # Clean up to save memory
        del data_array, labels
        gc.collect()

        # Check if we have valid sequences
        if len(X_full) == 0:
            self.logger.warning("No valid sequences created")
            if self.use_tf_dataset:
                return None, None, df, np.array([])
            else:
                return np.array([]), np.array([]), np.array([]), np.array([]), df, np.array([])

        # Split into train/validation sets
        train_size = int(self.train_ratio * len(X_full))

        X_train, X_val = X_full[:train_size], X_full[train_size:]
        y_train, y_val = y_full[:train_size], y_full[train_size:]
        fwd_returns_val = fwd_returns_full[train_size:]

        # Get validation dataframe for metrics
        entry_indices = list(range(self.sequence_length - 1, len(df_labeled)))
        val_entry_indices = entry_indices[train_size:]
        df_val = df_labeled.iloc[val_entry_indices].copy() if val_entry_indices else df_labeled.iloc[-1:].copy()

        # Apply normalization if specified
        if self.normalize_method:
            X_train, X_val = self._normalize_data(X_train, X_val)

        # Add noise to training data if specified
        if self.add_noise and len(X_train) > 0:
            noise = np.random.normal(0, self.noise_level, X_train.shape)
            X_train += noise

        # Oversample extreme classes (0 and 4) for better class balance
        X_train, y_train = self._oversample_extreme_classes(X_train, y_train)

        # Return as TensorFlow datasets if specified
        if self.use_tf_dataset:
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.shuffle(buffer_size=min(10000, len(X_train)))
            train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            return train_dataset, val_dataset, df_val, fwd_returns_val
        else:
            return X_train, y_train, X_val, y_val, df_val, fwd_returns_val

    @exception_handler(reraise=True)
    def prepare_test_data(self, df: pd.DataFrame, batch_size: int = 256) -> Tuple:
        """Prepare data for testing/inference.

        Args:
            df: DataFrame with features
            batch_size: Batch size for TensorFlow datasets

        Returns:
            Tuple of (X_test, y_test, df_test, fwd_returns_test)
            or (test_dataset, df_test, fwd_returns_test) if use_tf_dataset=True
        """
        # Drop NaN values
        df = df.dropna().copy()

        # Check if we have enough data
        if len(df) < (self.sequence_length + self.horizon):
            self.logger.warning(
                f"Insufficient test data: {len(df)} < {self.sequence_length + self.horizon}"
            )
            if self.use_tf_dataset:
                return None, df, np.array([])
            else:
                return np.array([]), np.array([]), df, np.array([])

        # Create labels and extract feature matrix
        df_labeled, labels, fwd_returns = self._create_labels(df)
        data_array = df_labeled.values.astype(np.float32)

        # Build sequences
        X_test, y_test, fwd_returns_test = self._build_sequences(data_array, labels, fwd_returns)

        # Clean up to save memory
        del data_array, labels
        gc.collect()

        # Apply normalization if we have a scaler and valid test data
        if self.scaler is not None and len(X_test) > 0:
            shape_0, shape_1, shape_2 = X_test.shape
            X_test_flat = X_test.reshape(-1, shape_2)
            X_test = self.scaler.transform(X_test_flat).reshape(shape_0, shape_1, shape_2)

        # Return as TensorFlow dataset if specified
        if self.use_tf_dataset and len(X_test) > 0:
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            return test_dataset, df_labeled, fwd_returns_test
        else:
            return X_test, y_test, df_labeled, fwd_returns_test

    def _create_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Create labels for training.

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (DataFrame without horizon, label array, forward returns array)
        """
        price = df[self.price_column]

        # Calculate forward returns
        fwd_return = (price.shift(-self.horizon) / price - 1).iloc[:-self.horizon]

        # Use ATR for adaptive thresholds if available
        if 'd1_ATR_14' in df.columns:
            atr_pct = (df['d1_ATR_14'] / df['close']).iloc[:-self.horizon]
        else:
            # Calculate ATR on the fly as fallback
            high_low = df['high'] - df['low']
            high_close_prev = (df['high'] - df['close'].shift(1)).abs()
            low_close_prev = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            atr_pct = (atr / df['close']).iloc[:-self.horizon]

        # Remove horizon periods from data
        df = df.iloc[:-self.horizon].copy()

        # Create labels with adaptive thresholds
        labels = np.zeros(len(fwd_return), dtype=int)

        # Vectorize the labeling
        labels[(fwd_return < -2 * atr_pct)] = 0  # Strong bearish
        labels[(fwd_return >= -2 * atr_pct) & (fwd_return < -0.5 * atr_pct)] = 1  # Moderate bearish
        labels[(fwd_return >= -0.5 * atr_pct) & (fwd_return < 0.5 * atr_pct)] = 2  # Neutral
        labels[(fwd_return >= 0.5 * atr_pct) & (fwd_return < 2 * atr_pct)] = 3  # Moderate bullish
        labels[(fwd_return >= 2 * atr_pct)] = 4  # Strong bullish

        return df, labels, fwd_return.values.astype(np.float32)

    def _build_sequences(self, data_array: np.ndarray, labels_array: np.ndarray,
                         fwd_returns_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build sequences for training.

        Args:
            data_array: NumPy array with feature data
            labels_array: NumPy array with labels
            fwd_returns_array: NumPy array with forward returns

        Returns:
            Tuple of (X, y, fwd_returns)
        """
        num_samples = len(data_array) - self.sequence_length + 1

        if num_samples <= 0:
            return np.array([]), np.array([]), np.array([])

        # Initialize arrays
        X = np.zeros((num_samples, self.sequence_length, data_array.shape[1]), dtype=np.float32)
        y = np.zeros((num_samples, 5), dtype=np.float32)  # 5 classes for classification
        fwd_returns = np.zeros(num_samples, dtype=np.float32)

        # Build sequences (can be optimized with stride_tricks for very large datasets)
        for i in range(num_samples):
            X[i] = data_array[i:i + self.sequence_length]

            # One-hot encode the label
            label = labels_array[i + self.sequence_length - 1]
            y[i, label] = 1  # Set the corresponding class to 1

            # Store forward return
            fwd_returns[i] = fwd_returns_array[i + self.sequence_length - 1]

        return X, y, fwd_returns

    def _normalize_data(self, X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize data using specified method.

        Args:
            X_train: Training data
            X_val: Validation data

        Returns:
            Tuple of (normalized X_train, normalized X_val)
        """
        if self.normalize_method.lower() == 'zscore':
            self.scaler = StandardScaler()

            # Flatten data for scaling
            X_train_flat = X_train.reshape(-1, X_train.shape[2])
            X_val_flat = X_val.reshape(-1, X_val.shape[2])

            # Fit on training data
            self.scaler.fit(X_train_flat)

            # Transform data and reshape back
            X_train = self.scaler.transform(X_train_flat).reshape(X_train.shape)
            X_val = self.scaler.transform(X_val_flat).reshape(X_val.shape)

        elif self.normalize_method.lower() == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()

            # Flatten data for scaling
            X_train_flat = X_train.reshape(-1, X_train.shape[2])
            X_val_flat = X_val.reshape(-1, X_val.shape[2])

            # Fit on training data
            self.scaler.fit(X_train_flat)

            # Transform data and reshape back
            X_train = self.scaler.transform(X_train_flat).reshape(X_train.shape)
            X_val = self.scaler.transform(X_val_flat).reshape(X_val.shape)

        return X_train, X_val

    def _oversample_extreme_classes(self, X: np.ndarray, y: np.ndarray,
                                    oversampling_ratio: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """Oversample extreme classes for better balance.

        Args:
            X: Feature data
            y: One-hot encoded labels
            oversampling_ratio: Ratio to oversample by

        Returns:
            Tuple of (oversampled X, oversampled y)
        """
        # Convert one-hot encoded y to class indices
        y_indices = np.argmax(y, axis=1)

        # Find indices of extreme classes (0 and 4)
        extreme_indices = np.where((y_indices == 0) | (y_indices == 4))[0]

        if len(extreme_indices) == 0:
            return X, y

        # Determine how many samples to add (oversampling_ratio - 1 times the original count)
        n_samples_to_add = int((oversampling_ratio - 1) * len(extreme_indices))

        if n_samples_to_add <= 0:
            return X, y

        # Randomly select indices to duplicate (with replacement)
        indices_to_duplicate = np.random.choice(extreme_indices, size=n_samples_to_add, replace=True)

        # Add selected samples
        X_additional = X[indices_to_duplicate]
        y_additional = y[indices_to_duplicate]

        # Add small random noise to duplicated samples to avoid exact duplicates
        if self.add_noise:
            noise = np.random.normal(0, self.noise_level, X_additional.shape)
            X_additional += noise

        # Concatenate original and additional samples
        X_oversampled = np.concatenate([X, X_additional])
        y_oversampled = np.concatenate([y, y_additional])

        return X_oversampled, y_oversampled

    def get_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """Calculate class weights to handle imbalanced classes.

        Args:
            y_train: One-hot encoded training labels

        Returns:
            Dictionary with class weights
        """
        # Convert one-hot encoded y to class indices
        y_indices = np.argmax(y_train, axis=1)

        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_indices),
            y=y_indices
        )

        # Convert to dictionary
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        return class_weight_dict

    def analyze_class_distribution(self, y: np.ndarray) -> Dict[str, Any]:
        """Analyze class distribution in labels.

        Args:
            y: One-hot encoded labels

        Returns:
            Dictionary with distribution statistics
        """
        # Convert one-hot encoded y to class indices
        y_indices = np.argmax(y, axis=1)

        # Count classes
        class_counts = np.bincount(y_indices, minlength=5)

        # Calculate proportions
        total = len(y_indices)
        class_proportions = class_counts / total if total > 0 else np.zeros(5)

        # Create result dictionary
        result = {
            'total_samples': total,
            'class_counts': {i: count for i, count in enumerate(class_counts)},
            'class_proportions': {i: prop for i, prop in enumerate(class_proportions)},
            'most_common_class': np.argmax(class_counts) if total > 0 else None,
            'imbalance_ratio': (
                max(class_counts) / min(class_counts) if min(class_counts) > 0 else float('inf')) if total > 0 else 0
        }

        return result