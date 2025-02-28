"""
Enhanced model architecture for cryptocurrency trading.

This module provides an enhanced deep learning model architecture
for cryptocurrency price prediction and trading.
"""

import gc
import json
import logging
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.src.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from keras.src.layers import (
    Dense, LSTM, GRU, Bidirectional, Conv1D, BatchNormalization,
    Dropout, GlobalAveragePooling1D, GlobalMaxPooling1D,
    MultiHeadAttention, Concatenate, Add, LayerNormalization
)
from keras.src.optimizers import Adam
from keras.src.optimizers.schedules import (
    CosineDecay, ExponentialDecay, PiecewiseConstantDecay
)
from keras.src.regularizers import L2
from keras.src.saving import load_model

from ..utils.path_manager import get_path_manager

# Try to import keras_tuner, but handle the case where it's not installed
try:
    import keras_tuner as kt
    from keras_tuner import Hyperband, BayesianOptimization, RandomSearch, Objective
    KERAS_TUNER_AVAILABLE = True
except ImportError:
    KERAS_TUNER_AVAILABLE = False

from .metrics import TradingMetrics
from .callbacks import RiskAdjustedTradeMetric, SaveBestModelCallback, MemoryCheckpoint
from ..utils.logging_utils import exception_handler
from ..utils.memory_monitor import memory_usage_decorator, log_memory_usage, clear_memory


class EnhancedCryptoModel:
    """Enhanced deep learning model for cryptocurrency trading."""

    def __init__(self, project_name: str = "enhanced_crypto_model",
                 max_trials: int = 100, tuner_type: str = "bayesian",
                 model_save_path: str = "best_enhanced_model.keras",
                 label_smoothing: float = 0.1, ensemble_size: int = 3,
                 output_classes: int = 5, use_mixed_precision: bool = True,
                 use_xla_acceleration: bool = True, seed: int = 42,
                 logger: Optional[logging.Logger] = None):
        """Initialize model.

        Args:
            project_name: Name of the project for tuner
            max_trials: Maximum number of tuning trials
            tuner_type: Type of tuner ('bayesian', 'hyperband', or 'random')
            model_save_path: Path to save the best model
            label_smoothing: Label smoothing factor for loss function
            ensemble_size: Number of models in ensemble
            output_classes: Number of output classes
            use_mixed_precision: Whether to use mixed precision
            use_xla_acceleration: Whether to use XLA acceleration
            seed: Random seed for reproducibility
            logger: Logger to use
        """
        self.project_name = project_name
        self.max_trials = max_trials
        self.tuner_type = tuner_type
        self.model_save_path = model_save_path
        self.label_smoothing = label_smoothing
        self.ensemble_size = ensemble_size
        self.output_classes = output_classes
        self.use_mixed_precision = use_mixed_precision
        self.use_xla_acceleration = use_xla_acceleration
        self.seed = seed
        self.logger = logger or logging.getLogger('EnhancedCryptoModel')

        # Set random seeds for reproducibility
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        # Storage for models
        self.best_model = None
        self.best_hp = None
        self.ensemble_models = []

        # Configure TensorFlow for performance
        self._configure_tensorflow()

        # Get path manager
        self.path_manager = get_path_manager

        # Update model save path to use path manager if it's a default path
        if model_save_path == "best_enhanced_model.keras":
            model_dir = self.path_manager.get_model_dir(project_name)
            self.model_save_path = str(model_dir / f"best_{project_name}.keras")
        else:
            self.model_save_path = model_save_path

    def _configure_tensorflow(self):
        """Configure TensorFlow for performance."""
        # Enable mixed precision if requested
        if self.use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            self.logger.info("Mixed precision enabled")

        # Enable XLA acceleration if requested
        if self.use_xla_acceleration:
            tf.config.optimizer.set_jit(True)
            self.logger.info("XLA acceleration enabled")

        # Configure GPU memory growth
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except Exception as e:
                    self.logger.warning(f"Could not set memory growth on {device}: {str(e)}")
            self.logger.info(f"GPU memory growth enabled for {len(physical_devices)} devices")
        else:
            self.logger.warning("No GPU devices found")

        # Set optimal thread counts
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    def _transformer_block(self, x: tf.Tensor, units: int, num_heads: int,
                           dropout_rate: float = 0.1, use_layer_norm: bool = True) -> tf.Tensor:
        """Implement a transformer block with residual connections.

        Args:
            x: Input tensor
            units: Number of units
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            use_layer_norm: Whether to use layer normalization

        Returns:
            Output tensor
        """
        # Get input dimensions for residual connection
        input_shape = tf.shape(x)
        input_dim = x.shape[-1]

        # Multi-head attention
        if use_layer_norm:
            attention_input = LayerNormalization()(x)
        else:
            attention_input = x

        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units // num_heads,
            dropout=dropout_rate
        )(attention_input, attention_input)

        # Residual connection
        x = Add()([x, attention_output])

        # Feed-forward network
        if use_layer_norm:
            ffn_input = LayerNormalization()(x)
        else:
            ffn_input = x

        ffn_output = Dense(units * 4, activation='gelu')(ffn_input)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        ffn_output = Dense(input_dim)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)

        # Second residual connection
        x = Add()([x, ffn_output])

        return x

    def _build_model(self, hp, input_shape: Tuple[int, int], total_steps: int) -> Model:
        """Build model architecture with hyperparameters.

        Args:
            hp: Hyperparameters object
            input_shape: Shape of input data
            total_steps: Total training steps for scheduling

        Returns:
            Compiled model
        """
        # Input layer
        inputs = Input(shape=input_shape, dtype=tf.float32)

        # Initial convolution layer
        filter0 = hp.Int("conv_filter_0", min_value=32, max_value=128, step=32)
        kernel_size = hp.Choice("kernel_size", values=[3, 5, 7])
        x = Conv1D(
            filters=filter0,
            kernel_size=kernel_size,
            padding='same',
            activation='relu'
        )(inputs)
        x = BatchNormalization()(x)
        x = Dropout(rate=hp.Float("dropout_rate_0", min_value=0.1, max_value=0.3, step=0.1))(x)

        # Option to use multi-scale convolutions
        use_multiscale = hp.Boolean("use_multiscale")
        if use_multiscale:
            # Multi-scale convolutional block
            conv3 = Conv1D(filters=filter0 // 2, kernel_size=3, padding='same', activation='relu')(x)
            conv5 = Conv1D(filters=filter0 // 2, kernel_size=5, padding='same', activation='relu')(x)
            conv7 = Conv1D(filters=filter0 // 2, kernel_size=7, padding='same', activation='relu')(x)
            x = Concatenate()([conv3, conv5, conv7])
            x = BatchNormalization()(x)

        # First recurrent block
        rnn_type = hp.Choice("rnn_type", values=["LSTM", "GRU"])
        dropout_rate1 = hp.Float("dropout_rate_1", min_value=0.1, max_value=0.5, step=0.1)
        l2_reg1 = hp.Float("l2_reg_1", min_value=1e-5, max_value=1e-2, sampling='log')
        unit1 = hp.Int("unit_1", min_value=32, max_value=128, step=32)

        if rnn_type == "LSTM":
            x = Bidirectional(LSTM(
                unit1,
                return_sequences=True,
                dropout=dropout_rate1,
                kernel_regularizer=L2(l2_reg1)
            ))(x)
        else:
            x = Bidirectional(GRU(
                unit1,
                return_sequences=True,
                dropout=dropout_rate1,
                kernel_regularizer=L2(l2_reg1)
            ))(x)

        x = BatchNormalization()(x)

        # Transformer blocks
        num_transformer_blocks = hp.Int("num_transformer_blocks", min_value=1, max_value=3)
        transformer_units = hp.Int("transformer_units", min_value=32, max_value=128, step=32)
        num_heads = hp.Int("num_heads", min_value=2, max_value=8, step=2)
        use_layer_norm = hp.Boolean("use_layer_norm")

        for i in range(num_transformer_blocks):
            x = self._transformer_block(
                x=x,
                units=transformer_units,
                num_heads=num_heads,
                dropout_rate=dropout_rate1,
                use_layer_norm=use_layer_norm
            )

        # Global pooling
        pooling_type = hp.Choice("pooling_type", values=["average", "max", "concat"])

        if pooling_type == "average":
            x = GlobalAveragePooling1D()(x)
        elif pooling_type == "max":
            x = GlobalMaxPooling1D()(x)
        else:  # "concat"
            avg_pool = GlobalAveragePooling1D()(x)
            max_pool = GlobalMaxPooling1D()(x)
            x = Concatenate()([avg_pool, max_pool])

        # Final dense layers
        final_units = hp.Int("final_units", min_value=64, max_value=256, step=64)
        final_dropout = hp.Float("final_dropout", min_value=0.1, max_value=0.5, step=0.1)

        x = Dense(final_units, activation="relu")(x)
        x = Dropout(rate=final_dropout)(x)

        # Output layer
        outputs = Dense(self.output_classes, activation="softmax", dtype=tf.float32)(x)

        # Create model
        model = Model(inputs, outputs)

        # Compile with metrics
        metrics = TradingMetrics(
            num_classes=self.output_classes,
            classes_to_monitor=list(range(self.output_classes))
        ).get_metrics()

        # Loss function - Focal loss with adjustable gamma
        gamma = hp.Float("gamma", min_value=1.0, max_value=5.0, step=0.5)
        loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
            gamma=gamma,
            label_smoothing=self.label_smoothing
        )

        # Learning rate schedule
        lr_schedule_type = hp.Choice("lr_schedule", values=["cosine", "exponential", "step"])
        initial_lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling='log')

        if lr_schedule_type == "cosine":
            lr_schedule = CosineDecay(
                initial_learning_rate=initial_lr,
                decay_steps=total_steps,
                alpha=0.1
            )
        elif lr_schedule_type == "exponential":
            lr_schedule = ExponentialDecay(
                initial_learning_rate=initial_lr,
                decay_steps=total_steps // 4,
                decay_rate=0.9
            )
        else:  # "step"
            lr_schedule = PiecewiseConstantDecay(
                boundaries=[total_steps // 3, total_steps * 2 // 3],
                values=[initial_lr, initial_lr * 0.1, initial_lr * 0.01]
            )

        # Optimizer
        optimizer = Adam(learning_rate=lr_schedule)

        # Compile model
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        return model

    @exception_handler(reraise=True)
    @memory_usage_decorator(threshold_gb=12)
    def tune_and_train(self, iteration: int, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray, df_val: pd.DataFrame,
                       fwd_returns_val: np.ndarray, epochs: int = 32, batch_size: int = 256,
                       class_weight: Optional[Dict[int, float]] = None, callbacks: Optional[List] = None) -> Tuple[
        Model, Dict]:
        """Tune hyperparameters and train model.

        Args:
            iteration: Current iteration
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            df_val: Validation DataFrame
            fwd_returns_val: Validation forward returns
            epochs: Maximum epochs for training
            batch_size: Batch size for training
            class_weight: Class weights dictionary
            callbacks: Additional callbacks

        Returns:
            Tuple of (best model, best hyperparameters)
        """
        if not KERAS_TUNER_AVAILABLE:
            self.logger.error("keras-tuner is not installed. Please install it with: pip install keras-tuner")
            raise ImportError("keras-tuner is not installed")

        if len(X_train) == 0:
            self.logger.error("No training data. Skipping tuner.")
            return None, None

        # Get input shape and calculate steps
        input_shape = (X_train.shape[1], X_train.shape[2])
        steps_per_epoch = len(X_train) // batch_size + (1 if len(X_train) % batch_size != 0 else 0)
        total_steps = steps_per_epoch * epochs

        # Define objective
        objective = Objective("val_avg_risk_adj_return", direction="max")

        # Create tuner
        if self.tuner_type.lower() == "hyperband":
            tuner = Hyperband(
                hypermodel=lambda hp: self._build_model(hp, input_shape, total_steps),
                objective=objective,
                max_epochs=epochs,
                factor=3,
                executions_per_trial=1,
                project_name=self.project_name,
                overwrite=True,
                directory=os.path.join("tuner_results", self.project_name),
                seed=self.seed
            )
        elif self.tuner_type.lower() == "random":
            tuner = RandomSearch(
                hypermodel=lambda hp: self._build_model(hp, input_shape, total_steps),
                objective=objective,
                max_trials=self.max_trials,
                executions_per_trial=1,
                project_name=self.project_name,
                overwrite=True,
                directory=os.path.join("tuner_results", self.project_name),
                seed=self.seed
            )
        else:  # default to bayesian
            tuner = BayesianOptimization(
                hypermodel=lambda hp: self._build_model(hp, input_shape, total_steps),
                objective=objective,
                max_trials=self.max_trials,
                executions_per_trial=1,
                project_name=self.project_name,
                overwrite=True,
                directory=os.path.join("tuner_results", self.project_name),
                seed=self.seed
            )

        # Create logs directory
        # Create logs directory using path manager
        logs_dir = self.path_manager.get_timestamped_dir(
            'logs_tensorboard',
            f"{self.project_name}_iter_{iteration}"
        )
        os.makedirs(logs_dir, exist_ok=True)

        # Default callbacks
        default_callbacks = [
            EarlyStopping(
                monitor='val_avg_risk_adj_return',
                patience=6,
                restore_best_weights=True,
                mode='max'
            ),
            SaveBestModelCallback(
                filepath=self.model_save_path,
                monitor='val_avg_risk_adj_return',
                mode='max',
                save_best_only=True,
                verbose=1,
                logger=self.logger
            ),
            RiskAdjustedTradeMetric(
                X_val=X_val,
                y_val=y_val,
                fwd_returns_val=fwd_returns_val,
                df_val=df_val,
                logger=self.logger
            ),
            MemoryCheckpoint(
                threshold_gb=14,
                cleanup_threshold_gb=16,
                logger=self.logger
            ),
            TensorBoard(
                log_dir=logs_dir,
                histogram_freq=1,
                profile_batch=0  # Disable profiling to save memory
            ),
            CSVLogger(
                os.path.join(logs_dir, 'training_log.csv'),
                append=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Combine callbacks
        all_callbacks = default_callbacks
        if callbacks:
            all_callbacks.extend(callbacks)

        # Create TensorFlow datasets for better performance
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=10000, seed=self.seed)
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Run hyperparameter search
        self.logger.info(f"Starting hyperparameter tuning with {self.tuner_type} search")
        start_time = time.time()

        tuner.search(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=all_callbacks,
            class_weight=class_weight,
            verbose=1
        )

        tuning_time = time.time() - start_time
        self.logger.info(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")

        # Get best hyperparameters
        best_trial = tuner.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        self.best_hp = best_hp.values

        # Build and compile best model
        self.best_model = tuner.hypermodel.build(best_hp)

        # Log best hyperparameters
        self.logger.info(f"Best hyperparameters: {self.best_hp}")

        # Save best hyperparameters to file
        hp_path = os.path.join(logs_dir, 'best_hyperparameters.json')
        with open(hp_path, 'w') as f:
            json.dump(self.best_hp, f, indent=2)

        # Clean up
        del tuner
        gc.collect()
        clear_memory()

        return self.best_model, self.best_hp

    @exception_handler(reraise=True)
    @memory_usage_decorator(threshold_gb=12)
    def build_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                       y_val: np.ndarray, df_val: pd.DataFrame, fwd_returns_val: np.ndarray,
                       epochs: int = 32, batch_size: int = 256,
                       class_weight: Optional[Dict[int, float]] = None) -> List[Tuple[Model, Dict]]:
        """Train an ensemble of models with different seeds and architectures.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            df_val: Validation DataFrame
            fwd_returns_val: Validation forward returns
            epochs: Maximum epochs for training
            batch_size: Batch size for training
            class_weight: Class weights dictionary

        Returns:
            List of (model, hyperparameters) tuples
        """
        # Save original seed
        original_seed = self.seed

        # Create ensemble directory
        # Create ensemble directory
        ensemble_dir = self.path_manager.get_model_dir(
            model_name=os.path.basename(os.path.splitext(self.model_save_path)[0]),
            ensemble=True
        )
        os.makedirs(ensemble_dir, exist_ok=True)

        # Empty existing ensemble
        self.ensemble_models = []

        for i in range(self.ensemble_size):
            # Use different seeds for ensemble diversity
            self.seed = original_seed + i * 100
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

            # Custom model save path for each ensemble member
            base_path = os.path.splitext(self.model_save_path)[0]
            ensemble_path = f"{base_path}_ensemble_{i}.keras"

            self.logger.info(f"Training ensemble model {i + 1}/{self.ensemble_size}")

            # Clean up before each model
            tf.keras.backend.clear_session()
            gc.collect()

            try:
                # Train model
                model, hp = self.tune_and_train(
                    iteration=i,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    df_val=df_val,
                    fwd_returns_val=fwd_returns_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=class_weight
                )

                if model is not None:
                    # Save model
                    model.save(ensemble_path)
                    self.ensemble_models.append((model, hp))
                    self.logger.info(
                        f"Ensemble model {i + 1}/{self.ensemble_size} trained and saved to {ensemble_path}")

            except Exception as e:
                self.logger.error(f"Error training ensemble model {i + 1}: {str(e)}")

        # Restore original seed
        self.seed = original_seed
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        # Log ensemble results
        self.logger.info(
            f"Ensemble training completed: {len(self.ensemble_models)}/{self.ensemble_size} models trained")

        return self.ensemble_models

    @exception_handler(reraise=True)
    def predict_with_ensemble(self, X_new: np.ndarray, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the ensemble of models.

        Args:
            X_new: Input features
            batch_size: Batch size for prediction

        Returns:
            Tuple of (ensemble predictions, uncertainty)
        """
        if not self.ensemble_models:
            self.logger.error("No ensemble models found. Train ensemble first.")
            raise RuntimeError("No ensemble models found. Train ensemble first.")

        # Log memory before prediction
        log_memory_usage()

        all_predictions = []
        for i, (model, _) in enumerate(self.ensemble_models):
            # Check memory before each model prediction
            memory_gb = log_memory_usage()
            if memory_gb > 16:
                self.logger.warning(f"Memory usage high ({memory_gb:.2f}GB) before prediction with model {i + 1}")
                clear_memory()

            # Make prediction
            try:
                pred = model.predict(X_new, batch_size=batch_size, verbose=0)
                all_predictions.append(pred)
            except Exception as e:
                self.logger.error(f"Error predicting with ensemble model {i + 1}: {str(e)}")
                # Skip this model but continue with others
                continue

            # Clear individual model from memory if not last one
            if i < len(self.ensemble_models) - 1:
                tf.keras.backend.clear_session()
                gc.collect()

        if not all_predictions:
            self.logger.error("All ensemble predictions failed.")
            raise RuntimeError("All ensemble predictions failed.")

        # Average predictions from all models
        ensemble_pred = np.mean(all_predictions, axis=0)

        # Calculate uncertainty (standard deviation across models)
        uncertainty = np.std(all_predictions, axis=0)

        # Cleanup after prediction
        clear_memory()

        return ensemble_pred, uncertainty

    @exception_handler(reraise=True)
    def load_ensemble(self, base_path: Optional[str] = None, num_models: Optional[int] = None) -> bool:
        """Load a previously trained ensemble.

        Args:
            base_path: Base path for ensemble models
            num_models: Number of models to load

        Returns:
            True if successful, False otherwise
        """
        if base_path is None:
            base_path = os.path.splitext(self.model_save_path)[0]

        if num_models is None:
            num_models = self.ensemble_size

        self.ensemble_models = []

        for i in range(num_models):
            # Clear memory before loading each model
            tf.keras.backend.clear_session()
            gc.collect()

            model_path = f"{base_path}_ensemble_{i}.keras"

            if os.path.exists(model_path):
                try:
                    model = load_model(model_path)
                    self.ensemble_models.append((model, None))  # We don't have hyperparameters here
                    self.logger.info(f"Loaded ensemble model {i + 1} from {model_path}")
                except Exception as e:
                    self.logger.error(f"Error loading ensemble model {i + 1}: {str(e)}")
            else:
                self.logger.warning(f"Ensemble model file not found: {model_path}")

            # Monitor memory after loading each model
            log_memory_usage()

        self.logger.info(f"Loaded {len(self.ensemble_models)} ensemble models")

        return len(self.ensemble_models) > 0

    @exception_handler(reraise=True)
    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 256) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        results = {}

        # Evaluate with ensemble if available
        if hasattr(self, 'ensemble_models') and self.ensemble_models:
            self.logger.info("Evaluating ensemble model")

            # Get ensemble predictions
            ensemble_preds, uncertainties = self.predict_with_ensemble(X_val, batch_size)

            # Convert to class indices
            y_pred = np.argmax(ensemble_preds, axis=1)
            y_true = np.argmax(y_val, axis=1)

            # Calculate accuracy
            accuracy = np.mean(y_pred == y_true)
            results['accuracy'] = accuracy

            # Calculate per-class metrics
            for class_id in range(self.output_classes):
                # True positives, false positives, etc.
                true_pos = np.sum((y_true == class_id) & (y_pred == class_id))
                false_pos = np.sum((y_true != class_id) & (y_pred == class_id))
                false_neg = np.sum((y_true == class_id) & (y_pred != class_id))

                # Precision and recall
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

                # F1 score
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                # Store results
                results[f'precision_class_{class_id}'] = precision
                results[f'recall_class_{class_id}'] = recall
                results[f'f1_class_{class_id}'] = f1

            # Mean uncertainty
            results['mean_uncertainty'] = np.mean(uncertainties)

            # Log results
            self.logger.info(f"Ensemble evaluation results: {results}")

            return results

        # Evaluate single model if no ensemble
        if self.best_model is None:
            self.logger.error("No model to evaluate.")
            return {}

        # Evaluate model
        metrics_vals = self.best_model.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)

        # Store metrics in dictionary
        metric_names = self.best_model.metrics_names
        for name, value in zip(metric_names, metrics_vals):
            results[name] = value

        # Log results
        self.logger.info(f"Model evaluation results: {results}")

        return results

    @exception_handler(reraise=True)
    def load_best_model(self) -> bool:
        """Load the best model from disk.

        Returns:
            True if successful, False otherwise
        """
        if os.path.exists(self.model_save_path):
            try:
                self.best_model = load_model(self.model_save_path)
                self.logger.info(f"Loaded model from {self.model_save_path}")
                return True
            except Exception as e:
                self.logger.error(f"Error loading model from {self.model_save_path}: {str(e)}")
                return False
        else:
            self.logger.warning(f"No model found at {self.model_save_path}")
            return False

    @exception_handler(reraise=True)
    def predict_signals(self, X_new: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Predict signals for new data.

        Args:
            X_new: Input features
            batch_size: Batch size for prediction

        Returns:
            Array of predictions
        """
        # If we have no model, generate random predictions for backtesting
        if (self.best_model is None and
                (not hasattr(self, 'ensemble_models') or not self.ensemble_models)):
            self.logger.warning(
                "No model found. Generating random predictions for backtesting purposes."
            )
            # Generate random predictions biased toward the middle class (neutral)
            # Class probabilities: 0: 15%, 1: 15%, 2: 40%, 3: 15%, 4: 15%
            num_samples = len(X_new)
            preds = np.zeros((num_samples, 5), dtype=np.float32)

            # Add some randomness but bias toward neutral
            for i in range(num_samples):
                # Base probabilities
                base_probs = np.array([0.15, 0.15, 0.40, 0.15, 0.15], dtype=np.float32)
                # Add random noise
                noise = np.random.normal(0, 0.05, 5)
                # Combine and normalize
                combined = base_probs + noise
                combined = np.clip(combined, 0.01, 0.99)  # Ensure no zeros
                preds[i] = combined / combined.sum()  # Normalize to sum to 1

            return preds

        # Use ensemble if available
        if hasattr(self, 'ensemble_models') and self.ensemble_models:
            try:
                preds, _ = self.predict_with_ensemble(X_new, batch_size)
                return preds
            except Exception as e:
                self.logger.error(f"Error in ensemble prediction: {str(e)}")
                # Fall back to single model if ensemble fails
                if self.best_model is not None:
                    return self.best_model.predict(X_new, batch_size=batch_size, verbose=0)
                else:
                    raise RuntimeError("No model available for prediction")

        # Use single model
        if self.best_model is None:
            self.logger.error("No model found. Train or load a model first.")
            raise RuntimeError("No model found. Train or load a model first.")

        return self.best_model.predict(X_new, batch_size=batch_size, verbose=0)

    def save_model_summary(self, filepath: Optional[str] = None) -> None:
        """Save model architecture summary to file.

        Args:
            filepath: Path to save summary (optional)
        """
        model = self.best_model

        if model is None:
            self.logger.error("No model to summarize.")
            return

        if filepath is None:
            model_name = os.path.splitext(os.path.basename(self.model_save_path))[0]
            summary_dir = self.path_manager.get_path('models_saved')
            filepath = str(summary_dir / f"{model_name}_summary.txt")

        with open(filepath, 'w') as f:
            # Store model summary
            model.summary(print_fn=lambda x: f.write(x + '\n'))

            # Store hyperparameters if available
            if self.best_hp:
                f.write("\nBest Hyperparameters:\n")
                for param, value in self.best_hp.items():
                    f.write(f"{param}: {value}\n")

        self.logger.info(f"Model summary saved to {filepath}")