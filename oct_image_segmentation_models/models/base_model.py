import abc
from tensorflow.keras import Model
from typeguard import typechecked
from typing import Callable, Tuple


@typechecked
class BaseModel:
    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        image_height: int,
        image_width: int,
    ):
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.image_height = image_height
        self.image_width = image_width

    @abc.abstractclassmethod
    def build_model(self, **kwargs) -> Tuple[Model, dict]:
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractclassmethod
    def get_preprocess_input_fn(self) -> Callable:
        raise NotImplementedError("Must be implemented in subclasses.")
