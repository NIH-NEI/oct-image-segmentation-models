import datetime
from pathlib import Path
import tensorflow as tf

from unet.model import custom_losses
from unet.model import custom_metrics


def get_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H_%M_%S")

    return timestamp

def load_model(model_path: Path):
    custom_objects = dict(
        list(custom_losses.custom_loss_objects.items())
        + list(custom_metrics.custom_metric_objects.items())
    )

    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)