from __future__ import annotations

import time

import h5py
import logging as log
from matplotlib import cm
import numpy as np
from pathlib import Path
from typeguard import typechecked

from tensorflow.keras.utils import to_categorical

from oct_image_segmentation_models.common import (
    dataset_construction as datacon,
    plotting,
    utils,
)
from oct_image_segmentation_models.min_path_processing import graph_search
from oct_image_segmentation_models.prediction.prediction_parameters import (
    PredictionParams,
)


@typechecked
class PredictionOutput:
    def __init__(
        self,
        image: np.array,
        image_name: Path,
        image_output_dir: Path,
        predicted_labels: np.array,
        categorical_pred: np.array,
        boundary_maps: np.array,
        flattened_image,
        offsets,
        flatten_boundary,
        delineations,
    ) -> None:
        self.image = image
        self.image_name = image_name
        self.image_output_dir = image_output_dir
        self.predicted_labels = predicted_labels
        self.categorical_pred = categorical_pred
        self.boundary_maps = boundary_maps
        self.flattened_image = flattened_image
        self.offsets = offsets
        self.flatten_boundary = flatten_boundary
        self.delineations = delineations


def predict(predict_params: PredictionParams) -> list[PredictionOutput]:
    dataset = predict_params.dataset
    predict_images = dataset.images
    predict_image_names = dataset.image_names
    predict_image_output_dirs = dataset.image_output_dirs

    save_predict_config_file(predict_params)

    prediction_outputs = []

    # pass images to network one at a time
    for i, (predict_image, image_name, image_output_dir) in enumerate(
        zip(predict_images, predict_image_names, predict_image_output_dirs)
    ):
        log.info(f"Inferring image {i}: {image_name}")
        start_predict_time = time.time()
        predicted_probs = predict_params.loaded_model.predict(
            np.expand_dims(
                predict_image / 255, axis=0
            ),  # keras works with batches
            verbose=2,
            batch_size=1,
        )
        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time

        log.info("Converting predictions to boundary maps...")

        start_convert_time = time.time()

        """
        predicted_labels: A matrix of shape (1, image_width, image_height) that
        contains the predicted classes numbered from 0 to num_classes - 1.

        categorical_pred: A matrix of shape (num_classes, image_width,
        image_height) that contains:
            - If 'bin' == True: 1 on pixels that belong the class and 0
            otherwise.
            - If 'bin' == False: Prediction probabilities for each pixel per
            class.
        """
        [predicted_labels, categorical_pred] = utils.perform_argmax(
            predicted_probs, bin=True
        )

        boundary_maps = utils.convert_predictions_to_maps_semantic(
            np.array(categorical_pred),
            bg_ilm=True,
            bg_csi=False,
        )

        end_convert_time = time.time()
        convert_time = end_convert_time - start_convert_time

        predicted_labels = np.squeeze(predicted_labels)
        categorical_pred = np.squeeze(categorical_pred)
        boundary_maps = np.squeeze(boundary_maps)

        # save data to files
        save_image_prediction_results(
            predict_params,
            predict_image,
            image_name,
            predicted_labels,
            categorical_pred,
            boundary_maps,
            predict_time,
            convert_time,
            image_output_dir,
        )

        # Segment probability maps using graph search
        log.info("Running graph search, segmenting boundary maps...")
        num_classes = len(categorical_pred)
        graph_structure = graph_search.create_graph_structure(
            predict_image.shape
        )

        start_graph_time = time.time()
        delineations, _, _ = graph_search.segment_maps(
            boundary_maps, None, graph_structure
        )

        reconstructed_maps = datacon.create_area_mask(
            predict_image.shape, delineations
        )
        reconstructed_maps = to_categorical(
            reconstructed_maps, num_classes=num_classes
        )
        reconstructed_maps = np.expand_dims(reconstructed_maps, axis=0)

        [prediction_label_gs, reconstructed_maps] = utils.perform_argmax(
            reconstructed_maps
        )

        prediction_label_gs = np.squeeze(prediction_label_gs)

        [flattened_image, offsets, flatten_boundary] = (None, None, None)
        if predict_params.flatten_image:
            [
                flattened_image,
                offsets,
                flatten_boundary,
            ] = datacon.flatten_image_boundary(
                predict_image,
                delineations[predict_params.flatten_ind],
                poly=predict_params.flatten_poly,
            )

        end_graph_time = time.time()
        graph_time = end_graph_time - start_graph_time

        save_graph_based_prediction_results(
            predict_params,
            predict_image,
            image_name,
            prediction_label_gs,
            delineations,
            graph_time,
            flattened_image,
            flatten_boundary,
            offsets,
            image_output_dir,
        )

        prediction_output = PredictionOutput(
            image=predict_image,
            image_name=image_name,
            image_output_dir=image_output_dir,
            predicted_labels=predicted_labels,
            categorical_pred=categorical_pred,
            boundary_maps=boundary_maps,
            flattened_image=flattened_image,
            offsets=offsets,
            flatten_boundary=flatten_boundary,
            delineations=delineations,
        )

        prediction_outputs.append(prediction_output)
        log.info(f"DONE processing image number {i}: {image_name}")

    return prediction_outputs


@typechecked
def save_predict_config_file(predict_params: PredictionParams):
    config_file = h5py.File(
        predict_params.config_output_dir / Path("prediction_params.hdf5"), "w"
    )

    config_file.attrs["model_filename"] = np.array(
        predict_params.model_path, dtype="S1000"
    )
    config_file.attrs["error_col_inc_range"] = np.array(
        (predict_params.col_error_range[0], predict_params.col_error_range[-1])
    )
    config_file.close()


@typechecked
def save_image_prediction_results(
    pred_params: PredictionParams,
    predict_image: np.array,
    image_name: Path,
    predicted_labels: np.array,
    categorical_pred: np.array,
    boundary_maps: np.array,
    predict_time: float,
    convert_time: float,
    output_dir: Path,
):
    hdf5_file = h5py.File(output_dir / Path("unet_prediction_info.hdf5"), "w")

    if pred_params.save_params.categorical_pred is True:
        hdf5_file.create_dataset(
            "categorical_pred", data=categorical_pred, dtype="uint8"
        )

        if pred_params.save_params.png_images is True:
            for map_ind in range(len(categorical_pred)):
                plotting.save_image_plot(
                    categorical_pred[map_ind],
                    output_dir / Path("categorical_pred_" + map_ind + ".png"),
                    cmap=cm.Blues,
                )

    if pred_params.save_params.predicted_labels is True:
        hdf5_file.create_dataset(
            "predicted_labels", data=predicted_labels, dtype="uint8"
        )

        if pred_params.save_params.png_images is True:
            plotting.save_image_plot(
                predicted_labels,
                output_dir / Path("predicted_labels.png"),
                cmap=plotting.colors.ListedColormap(
                    plotting.region_colours, N=len(categorical_pred)
                ),
            )

    if pred_params.save_params.boundary_maps is True:
        hdf5_file.create_dataset(
            "boundary_maps", data=boundary_maps, dtype="uint8"
        )

    hdf5_file.create_dataset("raw_image", data=predict_image, dtype="uint8")

    plotting.save_image_plot(
        predict_image,
        output_dir / Path("raw_image.png"),
        cmap=None if predict_image.shape[2] == 3 else cm.gray,
        vmin=0,
        vmax=255,
    )

    hdf5_file.attrs["model_filename"] = np.array(
        pred_params.model_path, dtype="S1000"
    )
    hdf5_file.attrs["image_name"] = np.array(image_name, dtype="S1000")
    hdf5_file.attrs["timestamp"] = np.array(
        utils.get_timestamp(), dtype="S1000"
    )
    hdf5_file.attrs["predict_time"] = np.array(predict_time)
    hdf5_file.attrs["convert_time"] = convert_time
    hdf5_file.close()


@typechecked
def save_graph_based_prediction_results(
    predict_params: PredictionParams,
    predict_image: np.array,
    image_name: Path,
    prediction_label_gs: np.array,
    delineations: np.array,
    graph_time: float,
    flattened_image: np.array,
    flatten_boundary: np.array,
    offsets: np.array,
    output_dir: Path,
):
    num_classes = delineations.shape[0] + 1
    # Save graph search based prediction results
    hdf5_file = h5py.File(
        output_dir / Path("graph_search_prediction_info.hdf5"), "w"
    )

    np.savetxt(
        output_dir / Path("boundaries.csv"),
        delineations,
        delimiter=",",
        fmt="%d",
    )

    np.savetxt(
        output_dir / Path("segmentation_map.csv"),
        np.transpose(prediction_label_gs),
        fmt="%d",
        delimiter=",",
    )

    hdf5_file.create_dataset("delineations", data=delineations, dtype="uint16")

    hdf5_file.create_dataset(
        "prediction_label_gs", data=prediction_label_gs, dtype="uint8"
    )

    plotting.save_image_plot(
        prediction_label_gs,
        output_dir / Path("prediction_label_gs.png"),
        cmap=plotting.colors.ListedColormap(
            plotting.region_colours, N=num_classes
        ),
    )

    plotting.save_segmentation_plot(
        predict_image,
        cm.gray,
        output_dir / Path("delin_plot.png"),
        delineations,
        predictions=None,
        column_range=predict_params.col_error_range,
    )

    hdf5_file.attrs["model_filename"] = np.array(
        predict_params.model_path, dtype="S1000"
    )
    hdf5_file.attrs["image_name"] = np.array(image_name, dtype="S1000")
    hdf5_file.attrs["timestamp"] = np.array(
        utils.get_timestamp(), dtype="S1000"
    )
    hdf5_file.attrs["graph_time"] = np.array(graph_time)

    if predict_params.flatten_image:
        hdf5_file.create_dataset(
            "flatten_boundary", data=flatten_boundary, dtype="uint16"
        )
        hdf5_file.create_dataset(
            "flatten_image", data=flattened_image, dtype="uint8"
        )
        hdf5_file.create_dataset(
            "flatten_offsets", data=offsets, dtype="uint16"
        )
        if predict_params.save_params.png_images:
            plotting.save_image_plot(
                flattened_image,
                output_dir / Path("flattened_image.png"),
                cmap=cm.gray,
                vmin=0,
                vmax=255,
            )
            plotting.save_segmentation_plot(
                predict_image,
                cm.gray,
                output_dir / Path("flatten_boundary_plot.png"),
                np.expand_dims(flatten_boundary, axis=0),
                predictions=None,
                column_range=predict_params.col_error_range,
            )

    hdf5_file.close()
