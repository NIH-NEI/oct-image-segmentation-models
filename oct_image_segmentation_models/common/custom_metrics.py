import numpy as np
from surface_distance import (
    compute_surface_distances,
    compute_average_surface_distance,
    compute_robust_hausdorff,
)
import tensorflow as tf
from tensorflow.keras import backend as K
from typing import Tuple


def _dice_coef(is_y_true_sparse, num_classes):
    def dice_coef(y_true, y_pred):
        if is_y_true_sparse:
            y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), num_classes)
        y_true_f = K.flatten(y_true)
        y_pred = K.cast(y_pred, "float32")
        y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), "float32")
        intersection = y_true_f * y_pred_f
        score = 2.0 * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
        return score

    return dice_coef


custom_metric_objects = {"dice_coef": _dice_coef}


def average_surface_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spacing: Tuple[float],
) -> dict:
    surface_distances = compute_surface_distances(y_true, y_pred, spacing)
    return compute_average_surface_distance(surface_distances)


def hausdorff_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spacing: Tuple[float],
    percent: float,
) -> float:
    surface_distances = compute_surface_distances(y_true, y_pred, spacing)
    return compute_robust_hausdorff(surface_distances, percent)
