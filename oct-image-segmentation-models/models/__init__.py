from tensorflow.keras.models import Model
from typeguard import typechecked
from typing import Tuple

from . import deeplabv3plus
from . import unet

model_name_map = {
    "deeplabv3plus": deeplabv3plus.DeeplabV3Plus,
    "unet": unet.unet,
}


@typechecked
def build_model(
    model_name: str, **model_hyperparameters
) -> Tuple[Model, dict]:
    model = model_name_map.get(model_name)

    if model is None:
        raise ValueError(f"Model name: '{model_name}' could not be found.")

    return model(**model_hyperparameters)
