from __future__ import annotations

import h5py
import logging as log
from pathlib import Path
from tensorflow.keras.utils import to_categorical

from unet.model import augmentation as aug
from unet.model import dataset_construction as dc
from unet.model import dataset_loader as dl
from unet.model import eval_helper
from unet.model import image_database as imdb
from unet.model import save_parameters
from unet.model.evaluation_parameters import EvaluationParameters, PredictionDataset


def evaluate_model(
    eval_params: EvaluationParameters,
):
    if type(eval_params.prediction_dataset) == Path:
        test_dataset_file = h5py.File(eval_params.dataset_file_path, 'r')

        test_images, test_labels, test_segments, test_image_names = dl.load_testing_data(
            test_dataset_file
        )

        test_image_names = [ Path(x) for x in test_image_names ]
    elif type(eval_params.prediction_dataset) == PredictionDataset:
        pred_dataset = eval_params.prediction_dataset
        test_images = pred_dataset.prediction_images
        test_labels = None
        test_segments = None
        test_image_names = pred_dataset.prediction_images_names
    else:
        raise TypeError("'prediction_dataset' should be a 'Path' or a 'PredictionDataset'")

    if eval_params.is_evaluate:
        # If segments are provided, then build labels from segments and ignore provided labels
        if not test_segments is None:
            log.info("Found 'test_segs' in HDF5 dataset so constructing labels from them")
            test_labels = dc.create_all_area_masks(test_images, test_segments)
        test_labels = to_categorical(test_labels, eval_params.num_classes)
    else:
        test_labels = None

    AREA_NAMES = ["area_" + str(i) for i in range(eval_params.num_classes)]
    BOUNDARY_NAMES = ["boundary_" + str(i) for i in range(eval_params.num_classes - 1)]

    filename = eval_params.prediction_dataset if type(eval_params.prediction_dataset) == Path else None

    eval_imdb = imdb.ImageDatabase(
        images=test_images,
        labels=test_labels,
        segs=test_segments,
        image_names=test_image_names,
        boundary_names=BOUNDARY_NAMES,
        area_names=AREA_NAMES,
        fullsize_class_names=AREA_NAMES,
        num_classes=eval_params.num_classes,
        filename=filename,
        mode_type='fullsize'
    )

    if eval_params.col_error_range is None:
        eval_params.col_error_range = range(eval_imdb.image_width)

    eval_helper.evaluate_network(
        eval_imdb,
        eval_params,
    )
