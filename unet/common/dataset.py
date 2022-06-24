import numpy as np
from pathlib import Path


class Dataset:
    def __init__(
        self,
        images: np.array,
        images_masks: np.array,
        images_names: list[Path],
        images_output_dirs: list[Path]
    ):
        self.images = images
        self.images_masks = images_masks
        self.images_names = images_names
        self.images_output_dirs = images_output_dirs
