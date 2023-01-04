from typeguard import typechecked


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
