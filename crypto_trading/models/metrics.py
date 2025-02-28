"""
Custom metrics for cryptocurrency trading models.

This module provides custom metrics for evaluating cryptocurrency
trading models, including per-class metrics and trading-specific metrics.
"""

from typing import Dict, List, Optional, Any

import tensorflow as tf
from keras.src.metrics import AUC, Metric


class PerClassAUC(Metric):
    """Area under the ROC/PR curve for a specific class.

    This metric calculates the AUC for a specific class in a multi-class problem.
    """

    def __init__(self, class_id: int, curve: str = 'PR', name: Optional[str] = None, **kwargs):
        """Initialize the metric.

        Args:
            class_id: ID of the class to calculate AUC for
            curve: Curve type ('ROC' or 'PR')
            name: Name of the metric
            **kwargs: Additional keyword arguments for the parent class
        """
        name = name or f'per_class_auc_{class_id}'
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.auc = AUC(curve=curve)

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None):
        """Update metric state.

        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            sample_weight: Optional sample weights
        """
        y_true_class = y_true[:, self.class_id]
        y_pred_class = y_pred[:, self.class_id]
        self.auc.update_state(y_true_class, y_pred_class, sample_weight)

    def result(self) -> tf.Tensor:
        """Return metric result.

        Returns:
            AUC value
        """
        return self.auc.result()

    def reset_states(self):
        """Reset metric state."""
        self.auc.reset_states()

    def get_config(self) -> Dict[str, Any]:
        """Get metric configuration.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'class_id': self.class_id,
            'curve': self.auc._curve  # Access private attribute
        })
        return config


class WeightedAccuracy(Metric):
    """Weighted accuracy metric that assigns different weights to different classes.

    This metric gives higher weight to correctly predicting extreme movements.
    """

    def __init__(self, class_weights: Optional[List[float]] = None, name: str = 'weighted_accuracy', **kwargs):
        """Initialize the metric.

        Args:
            class_weights: List of weights for each class
            name: Name of the metric
            **kwargs: Additional keyword arguments for the parent class
        """
        super().__init__(name=name, **kwargs)
        self.class_weights = class_weights or [1.5, 1.0, 0.5, 1.0, 1.5]
        self.total_weighted_correct = self.add_weight(
            name='total_weighted_correct', initializer='zeros'
        )
        self.total_weighted_samples = self.add_weight(
            name='total_weighted_samples', initializer='zeros'
        )

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None):
        """Update metric state.

        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            sample_weight: Optional sample weights
        """
        # Convert to class indices
        y_true_idx = tf.argmax(y_true, axis=1)
        y_pred_idx = tf.argmax(y_pred, axis=1)

        # Check correctness
        correct = tf.cast(tf.equal(y_true_idx, y_pred_idx), tf.float32)

        # Get class weights
        weights = tf.gather(tf.constant(self.class_weights, dtype=tf.float32), y_true_idx)

        # Apply sample weights if provided
        if sample_weight is not None:
            weights = weights * sample_weight

        # Update counters
        self.total_weighted_correct.assign_add(tf.reduce_sum(correct * weights))
        self.total_weighted_samples.assign_add(tf.reduce_sum(weights))

    def result(self) -> tf.Tensor:
        """Return metric result.

        Returns:
            Weighted accuracy value
        """
        return self.total_weighted_correct / tf.maximum(self.total_weighted_samples, 1e-7)

    def reset_states(self):
        """Reset metric state."""
        self.total_weighted_correct.assign(0.0)
        self.total_weighted_samples.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Get metric configuration.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'class_weights': self.class_weights,
        })
        return config


def f1_score(y_true: tf.Tensor, y_pred: tf.Tensor, class_id: int = None) -> tf.Tensor:
    """Calculate F1 score for a specific class or overall.

    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        class_id: Optional class ID to calculate F1 for

    Returns:
        F1 score
    """
    # If class_id is provided, extract binary labels for that class
    if class_id is not None:
        y_true = y_true[:, class_id]
        y_pred = y_pred[:, class_id]
    else:
        # For multi-class, convert to class indices
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)

        # Convert to one-hot for multi-class F1
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        y_pred = tf.one_hot(y_pred, depth=tf.shape(y_pred)[1])

    # Calculate precision and recall
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    actual_positives = tf.reduce_sum(y_true)

    precision = true_positives / tf.maximum(predicted_positives, 1e-7)
    recall = true_positives / tf.maximum(actual_positives, 1e-7)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / tf.maximum(precision + recall, 1e-7)

    return f1


class TradingMetrics:
    """Collection of trading-specific metrics for model evaluation."""

    def __init__(self, num_classes: int = 5, classes_to_monitor: Optional[List[int]] = None):
        """Initialize trading metrics.

        Args:
            num_classes: Number of classes
            classes_to_monitor: List of class IDs to monitor (default: all classes)
        """
        if classes_to_monitor is None:
            classes_to_monitor = list(range(num_classes))

        self.num_classes = num_classes
        self.classes_to_monitor = classes_to_monitor

        # Initialize metrics
        self.weighted_accuracy = WeightedAccuracy()
        self.precision = {}
        self.recall = {}
        self.f1 = {}
        self.pr_auc_metrics = {}
        self.roc_auc_metrics = {}

        # Create metrics for each class
        for class_id in self.classes_to_monitor:
            # Precision for each class
            self.precision[class_id] = self._precision_class(class_id)

            # Recall for each class
            self.recall[class_id] = self._recall_class(class_id)

            # F1 score for each class
            self.f1[class_id] = self._f1_score_class(class_id)

            # AUC metrics
            self.pr_auc_metrics[class_id] = PerClassAUC(
                class_id=class_id,
                curve='PR',
                name=f'pr_auc_class_{class_id}'
            )
            self.roc_auc_metrics[class_id] = PerClassAUC(
                class_id=class_id,
                curve='ROC',
                name=f'roc_auc_class_{class_id}'
            )

    def _precision_class(self, class_id: int):
        """Create precision metric for a specific class.

        Args:
            class_id: Class ID

        Returns:
            Precision function
        """

        def precision(y_true, y_pred):
            y_true_class = y_true[:, class_id]
            y_pred_class = tf.cast(tf.argmax(y_pred, axis=1) == class_id, tf.float32)

            true_positives = tf.reduce_sum(y_true_class * y_pred_class)
            predicted_positives = tf.reduce_sum(y_pred_class)

            return true_positives / tf.maximum(predicted_positives, 1e-7)

        precision.__name__ = f'precision_class_{class_id}'
        return precision

    def _recall_class(self, class_id: int):
        """Create recall metric for a specific class.

        Args:
            class_id: Class ID

        Returns:
            Recall function
        """

        def recall(y_true, y_pred):
            y_true_class = y_true[:, class_id]
            y_pred_class = tf.cast(tf.argmax(y_pred, axis=1) == class_id, tf.float32)

            true_positives = tf.reduce_sum(y_true_class * y_pred_class)
            actual_positives = tf.reduce_sum(y_true_class)

            return true_positives / tf.maximum(actual_positives, 1e-7)

        recall.__name__ = f'recall_class_{class_id}'
        return recall

    def _f1_score_class(self, class_id: int):
        """Create F1 score metric for a specific class.

        Args:
            class_id: Class ID

        Returns:
            F1 score function
        """

        def f1(y_true, y_pred):
            return f1_score(y_true, y_pred, class_id)

        f1.__name__ = f'f1_class_{class_id}'
        return f1

    def get_metrics(self) -> List:
        """Get all metrics for model compilation.

        Returns:
            List of metrics
        """
        metrics = ['accuracy', self.weighted_accuracy]

        for class_id in self.classes_to_monitor:
            metrics.extend([
                self.precision[class_id],
                self.recall[class_id],
                self.f1[class_id],
                self.pr_auc_metrics[class_id],
                self.roc_auc_metrics[class_id]
            ])

        return metrics