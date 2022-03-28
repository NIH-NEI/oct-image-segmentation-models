from __future__ import annotations

import h5py
import logging as log
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import to_categorical

from unet.min_path_processing import utils
from unet.model import augmentation as aug
from unet.model import dataset_construction as dc
from unet.model import dataset_loader as dl
from unet.model import eval_helper
from unet.model import image_database as imdb
from unet.model import save_parameters
from unet.model.evaluation_parameters import EvaluationParameters, Dataset


def evaluate_model(
    eval_params: EvaluationParameters,
):

    dataset = eval_params.dataset
    test_images = dataset.images
    test_labels = dataset.images_masks
    test_image_names = dataset.images_names

    AREA_NAMES = ["area_" + str(i) for i in range(eval_params.num_classes)]
    BOUNDARY_NAMES = ["boundary_" + str(i) for i in range(eval_params.num_classes - 1)]

    test_segments = None
    if eval_params.is_evaluate:
        test_segments = np.swapaxes(utils.generate_boundary(np.squeeze(test_labels, axis=3), axis=2), 0, 1)
        test_labels = to_categorical(test_labels, eval_params.num_classes)

    eval_imdb = imdb.ImageDatabase(
        images=test_images,
        labels=test_labels,
        segs=test_segments,
        image_names=test_image_names,
        boundary_names=BOUNDARY_NAMES,
        area_names=AREA_NAMES,
        fullsize_class_names=AREA_NAMES,
        num_classes=eval_params.num_classes,
        filename=None,
        mode_type='fullsize'
    )

    if eval_params.col_error_range is None:
        eval_params.col_error_range = range(eval_imdb.image_width)

    eval_helper.evaluate_network(
        eval_imdb,
        eval_params,
    )
