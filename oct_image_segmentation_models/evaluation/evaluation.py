from __future__ import annotations

import os

import h5py
from matplotlib import cm
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import to_categorical
import time
from typeguard import typechecked
from typing import Union

from oct_image_segmentation_models.min_path_processing import utils
from oct_image_segmentation_models.evaluation import (
    DICE_METRIC,
    AVERAGE_SURFACE_DISTANCE_METRIC,
    HAUSDORFF_DISTANCE_METRIC,
)
from oct_image_segmentation_models.evaluation.evaluation_parameters import (
    EvaluationParameters,
)
from oct_image_segmentation_models.common import (
    custom_metrics,
    dataset_construction as datacon,
    dataset_loader as dl,
    plotting,
    utils as common_utils,
)
from oct_image_segmentation_models.min_path_processing import graph_search

EVALUATION_RESULTS_FILENAME = "evaluation_results.hdf5"
GS_EVALUATION_RESULTS_FILENAME = "gs_evaluation_results.hdf5"


@typechecked
class EvaluationOutput:
    def __init__(
        self,
        image: np.ndarray,
        image_name: Path,
        image_segments: np.ndarray,
        image_output_dir: Path,
        predicted_labels: np.ndarray,
        categorical_pred: np.ndarray,
        boundary_maps: np.ndarray,
        gs_pred_segs: Union[np.ndarray, None],
        errors: Union[np.ndarray, None],
        mean_abs_err: Union[np.ndarray, None],
        mean_err: Union[np.ndarray, None],
        abs_err_sd: Union[np.ndarray, None],
        err_sd: Union[np.ndarray, None],
    ) -> None:
        self.image = image
        self.image_name = image_name
        self.image_segments = image_segments
        self.image_output_dir = image_output_dir
        self.predicted_labels = predicted_labels
        self.categorical_pred = categorical_pred
        self.boundary_maps = boundary_maps
        self.gs_pred_segs = gs_pred_segs
        self.errors = errors
        self.mean_abs_err = mean_abs_err
        self.mean_err = mean_err
        self.abs_err_sd = abs_err_sd
        self.err_sd = err_sd


@typechecked
def evaluate_model(
    eval_params: EvaluationParameters,
) -> list[EvaluationOutput]:

    test_dataset_file = h5py.File(eval_params.test_dataset_path, "r")

    eval_images, eval_labels, eval_image_names = dl.load_testing_data(
        test_dataset_file
    )

    eval_image_output_dirs = []
    for i in range(eval_images.shape[0]):
        eval_image_output_dirs.append(
            eval_params.save_foldername / Path(f"image_{i}")
        )

    eval_segments = np.swapaxes(
        utils.generate_boundary(np.squeeze(eval_labels, axis=3), axis=1), 0, 1
    )

    test_labels = to_categorical(eval_labels, eval_params.num_classes)

    save_eval_config_file(eval_params)

    eval_outputs = []

    #  pass images to network one at a time
    for ind in range(eval_images.shape[0]):
        eval_image = eval_images[ind]
        eval_label = test_labels[ind]
        eval_image_name = eval_image_names[ind]
        eval_seg = eval_segments[ind]
        eval_image_output_dir = eval_image_output_dirs[ind]

        if not os.path.exists(eval_image_output_dir):
            os.makedirs(eval_image_output_dir)

        print(
            "Evaluating image number: "
            + str(ind + 1)
            + " ("
            + str(eval_image_name)
            + ")..."
        )

        print("Running network predictions...")

        start_predict_time = time.time()
        predicted_probs = eval_params.loaded_model.predict(
            np.expand_dims(
                eval_image / 255, axis=0
            ),  # keras works with batches
            verbose=2,
            batch_size=1,
        )
        end_predict_time = time.time()

        predict_time = end_predict_time - start_predict_time

        print("Completed running network predictions...")

        # convert predictions to usable boundary probability maps
        """
        predicted_labels: A matrix of shape (1, image_height, image_width) that
        contains the predicted classes numbered from 0 to num_classes - 1.

        categorical_pred: A matrix of shape (1, num_classes, image_height,
        image_width) that contains:
            - If 'bin' == True: 1 on pixels that belong the class and 0
            otherwise.
            - If 'bin' == False: Prediction probabilities for each pixel per
            class.
        """
        [predicted_labels, categorical_pred] = common_utils.perform_argmax(
            predicted_probs, bin=True
        )

        """
        boundary_maps: A matrix of shape (# of classes (layers) - 1,
        imgage_height, image_width). Each "image" contains the delineation of
        the ith layer as a line marked with 255. The rest of the elements are
        0.
        """
        boundary_maps = common_utils.convert_predictions_to_maps_semantic(
            categorical_pred,
            bg_ilm=True,
            bg_csi=False,
        )

        eval_label_t = np.expand_dims(
            np.transpose(eval_label, axes=(2, 0, 1)),
            axis=0,
        )  # Class should be in the 2nd dimension.

        if DICE_METRIC in eval_params.metrics:
            dices = _soft_dice(eval_label_t, categorical_pred)

        if AVERAGE_SURFACE_DISTANCE_METRIC in eval_params.metrics:
            average_surface_distances = []
            average_surface_distances_gt_to_pred = []
            average_surface_distances_pred_to_gt = []

            # Skip background class
            for class_idx in range(1, eval_params.num_classes):
                class_eval_label = eval_label[:, :, class_idx].astype(bool)
                class_catgorical_pred = categorical_pred[
                    0, class_idx, :, :
                ].astype(bool)
                (
                    average_distance_gt_to_pred,
                    average_distance_pred_to_gt,
                ) = custom_metrics.average_surface_distance(
                    class_eval_label,
                    class_catgorical_pred,
                    spacing=(0.01111111, 0.01111111),
                )
                average_surface_distances_gt_to_pred.append(
                    average_distance_gt_to_pred
                )
                average_surface_distances_pred_to_gt.append(
                    average_distance_pred_to_gt
                )
                average_surface_distances.append(
                    (average_distance_gt_to_pred + average_distance_pred_to_gt)
                    / 2.0
                )
        else:
            average_surface_distances = None
            average_surface_distances_gt_to_pred = None
            average_surface_distances_pred_to_gt = None

        if HAUSDORFF_DISTANCE_METRIC in eval_params.metrics:
            hausdorff_distances = []
            # Skip background class
            for class_idx in range(1, eval_params.num_classes):
                class_eval_label = eval_label[:, :, class_idx].astype(bool)
                class_catgorical_pred = categorical_pred[
                    0, class_idx, :, :
                ].astype(bool)
                hausdorff_distances.append(
                    custom_metrics.hausdorff_distance(
                        class_eval_label,
                        class_catgorical_pred,
                        spacing=(0.01111111, 0.01111111),
                        percent=95,
                    )
                )
        else:
            hausdorff_distances = None

        predicted_labels = np.squeeze(predicted_labels)
        categorical_pred = np.squeeze(categorical_pred)
        boundary_maps = np.squeeze(boundary_maps)

        # save data to files
        _save_image_evaluation_results(
            eval_params,
            eval_image,
            eval_image_name,
            eval_seg,
            predicted_labels,
            categorical_pred,
            eval_label,
            eval_seg,
            dices,
            np.array(average_surface_distances),
            np.array(average_surface_distances_gt_to_pred),
            np.array(average_surface_distances_pred_to_gt),
            np.array(hausdorff_distances),
            predict_time,
            eval_image_output_dir,
        )

        if eval_params.graph_search:
            print("Running graph search, segmenting boundary maps...")
            eval_image_t = np.transpose(eval_image, axes=[1, 0, 2])
            boundary_maps_t = np.transpose(boundary_maps, axes=[0, 2, 1])
            graph_structure = graph_search.create_graph_structure(
                eval_image_t.shape
            )

            start_graph_time = time.time()

            """
            gs_pred_segs: A matrix of shape (# of classes (layers) - 1,
            imgage_width) where gs_pred_segs[i, col] specifies the row
            (y-dimension) of the ith layer

            errors: A matrix of shape (# of classes (layers) - 1, imgage_width)
            where error[i, col] specifies the pixel difference between
            delineation (i.e. prediction) and the true label (i.e. eval_seg)

            The third element in the tuple (originally named 'trim_maps'): A
            matrix of shape (# of classes (layers) - 1, imgage_width,
            image_height). It is the normalized version (0 to 1) of the
            'boundary_maps' input.
            """
            gs_pred_segs, errors, _ = graph_search.segment_maps(
                boundary_maps_t,
                eval_seg,
                graph_structure,
            )

            reconstructed_maps = datacon.create_area_mask(
                eval_image_t.shape, gs_pred_segs
            )
            reconstructed_maps = to_categorical(
                reconstructed_maps, num_classes=eval_params.num_classes
            )
            reconstructed_maps = np.expand_dims(reconstructed_maps, axis=0)

            [gs_eval_label, reconstructed_maps] = common_utils.perform_argmax(
                reconstructed_maps
            )

            gs_dices = _soft_dice(
                np.expand_dims(
                    # We transpose width and height (0, 1) and we bring the
                    # class dimension to the front since it needs to be in
                    # the 2nd dimension for _soft_dice.
                    np.transpose(eval_label, axes=[2, 1, 0]),
                    axis=0,
                ),
                reconstructed_maps,
            )

            gs_eval_label = np.transpose(np.squeeze(gs_eval_label))

            end_graph_time = time.time()
            graph_time = end_graph_time - start_graph_time

            """
            Vectors of length '# of classes (layers) - 1' with pixel difference
            (error) statistics.
            """
            (
                mean_abs_err,
                mean_err,
                abs_err_sd,
                err_sd,
            ) = graph_search.calculate_overall_errors(errors)

            _save_graph_based_evaluation_results(
                eval_params,
                eval_image,
                eval_image_name,
                eval_seg,
                gs_eval_label,
                gs_pred_segs,
                gs_dices,
                errors,
                mean_abs_err,
                mean_err,
                abs_err_sd,
                err_sd,
                graph_time,
                eval_image_output_dir,
            )
        else:
            print("Skipping graph search...")
            gs_pred_segs = None
            errors = None
            mean_abs_err = None
            mean_err = None
            abs_err_sd = None
            err_sd = None

        eval_output = EvaluationOutput(
            image=eval_image,
            image_name=eval_image_name,
            image_segments=eval_seg,
            image_output_dir=eval_image_output_dir,
            predicted_labels=predicted_labels,
            categorical_pred=categorical_pred,
            boundary_maps=boundary_maps,
            gs_pred_segs=gs_pred_segs,
            errors=errors,
            mean_abs_err=mean_abs_err,
            mean_err=mean_err,
            abs_err_sd=abs_err_sd,
            err_sd=err_sd,
        )

        eval_outputs.append(eval_output)

        print(
            "DONE image number: "
            + str(ind + 1)
            + " ("
            + str(eval_image_name)
            + ")..."
        )
        print("______________________________")

    _calc_overall_dataset_errors(
        eval_params,
        eval_image_names,
    )

    return eval_outputs


@typechecked
def _save_image_evaluation_results(
    eval_params: EvaluationParameters,
    eval_image: np.ndarray,
    image_name: Path,
    truth_label_segs: np.ndarray,
    predicted_labels: np.ndarray,
    categorical_pred: np.ndarray,
    eval_labels: np.ndarray,
    eval_segs: np.ndarray,
    dices: Union[np.ndarray, None],
    average_surface_distances: Union[np.ndarray, None],
    average_surface_distances_gt_to_pred: Union[np.ndarray, None],
    average_surface_distances_pred_to_gt: Union[np.ndarray, None],
    hausdorff_distances: Union[np.ndarray, None],
    predict_time: float,
    output_dir: Path,
):

    with open(output_dir / "input_image_name.txt", "w") as f:
        f.write(str(image_name))

    hdf5_file = h5py.File(output_dir / Path(EVALUATION_RESULTS_FILENAME), "w")
    if eval_params.save_params.categorical_pred is True:
        hdf5_file.create_dataset(
            "categorical_pred", data=categorical_pred, dtype="uint8"
        )

        if eval_params.save_params.png_images is True:
            for map_ind in range(len(categorical_pred)):
                plotting.save_image_plot(
                    categorical_pred[map_ind],
                    output_dir / Path("categorical_pred_" + map_ind + ".png"),
                    cmap=cm.Blues,
                )

    if eval_params.save_params.predicted_labels is True:
        hdf5_file.create_dataset(
            "non_gs_predicted_segmentation_map",
            data=predicted_labels,
            dtype="uint8",
        )

        if eval_params.save_params.png_images is True:
            plotting.save_image_plot(
                predicted_labels,
                output_dir / Path("non_gs_predicted_segmentation_map.png"),
                cmap=plotting.colors.ListedColormap(
                    plotting.region_colours, N=len(categorical_pred)
                ),
            )

    hdf5_file.create_dataset("raw_image", data=eval_image, dtype="uint8")

    plotting.save_image_plot(
        eval_image,
        output_dir / Path("raw_image.png"),
        cmap=None if eval_image.shape[2] == 3 else cm.gray,
        vmin=0,
        vmax=255,
    )

    hdf5_file.create_dataset("eval_labels", data=eval_labels, dtype="uint8")

    plotting.save_image_plot(
        np.argmax(eval_labels, axis=2),
        output_dir / Path("truth_segmentation_map.png"),
        cmap=plotting.colors.ListedColormap(
            plotting.region_colours, N=len(categorical_pred)
        ),
    )

    plotting.save_segmentation_plot(
        eval_image,
        cm.gray,
        output_dir / Path("truth_plot.png"),
        truth_label_segs,
        predictions=None,
        column_range=range(eval_image.shape[1]),
    )

    hdf5_file.create_dataset("raw_segs", data=eval_segs, dtype="uint16")

    if dices is not None:
        hdf5_file.create_dataset(
            "dices", data=np.squeeze(dices), dtype="float64"
        )

    if average_surface_distances is not None:
        hdf5_file.create_dataset(
            "average_surface_distances",
            data=average_surface_distances,
            dtype="float64",
        )

    if average_surface_distances_gt_to_pred is not None:
        hdf5_file.create_dataset(
            "average_surface_distances_gt_to_pred",
            data=average_surface_distances_gt_to_pred,
            dtype="float64",
        )

    if average_surface_distances_pred_to_gt is not None:
        hdf5_file.create_dataset(
            "average_surface_distances_pred_to_gt",
            data=average_surface_distances_pred_to_gt,
            dtype="float64",
        )

    if hausdorff_distances is not None:
        hdf5_file.create_dataset(
            "hausdorff_distances", data=hausdorff_distances, dtype="float64"
        )

    hdf5_file.attrs["model_filename"] = np.array(
        eval_params.model_path, dtype="S1000"
    )
    hdf5_file.attrs["image_name"] = np.array(image_name, dtype="S1000")
    hdf5_file.attrs["timestamp"] = np.array(
        common_utils.get_timestamp(), dtype="S1000"
    )
    hdf5_file.attrs["predict_time"] = np.array(predict_time)
    hdf5_file.close()


@typechecked
def _save_graph_based_evaluation_results(
    eval_params: EvaluationParameters,
    eval_image: np.ndarray,
    image_name: Path,
    truth_label_segs: np.ndarray,
    gs_eval_label: np.ndarray,
    gs_pred_segs: np.ndarray,
    gs_dices: np.ndarray,
    errors: np.ndarray,
    mean_abs_err: np.ndarray,
    mean_err: np.ndarray,
    abs_err_sd: np.ndarray,
    err_sd: np.ndarray,
    graph_time: float,
    output_dir: Path,
):
    num_classes = gs_pred_segs.shape[0] + 1
    # Save graph search based prediction results
    hdf5_file = h5py.File(
        output_dir / Path(GS_EVALUATION_RESULTS_FILENAME), "w"
    )

    np.savetxt(
        output_dir / Path("gs_boundaries.csv"),
        gs_pred_segs,
        delimiter=",",
        fmt="%d",
    )

    np.savetxt(
        output_dir / Path("gs_segmentation_map.csv"),
        gs_eval_label,
        fmt="%d",
        delimiter=",",
    )

    hdf5_file.create_dataset("gs_pred_segs", data=gs_pred_segs, dtype="uint16")
    hdf5_file.create_dataset("errors", data=errors, dtype="float64")
    hdf5_file.create_dataset(
        "mean_abs_err", data=mean_abs_err, dtype="float64"
    )
    hdf5_file.create_dataset("mean_err", data=mean_err, dtype="float64")
    hdf5_file.create_dataset("abs_err_sd", data=abs_err_sd, dtype="float64")
    hdf5_file.create_dataset("err_sd", data=err_sd, dtype="float64")
    hdf5_file.create_dataset(
        "gs_dices", data=np.squeeze(gs_dices), dtype="float64"
    )
    hdf5_file.create_dataset(
        "gs_predicted_labels", data=gs_eval_label, dtype="uint8"
    )

    plotting.save_image_plot(
        gs_eval_label,
        output_dir / Path("gs_predicted_segmentation_map.png"),
        cmap=plotting.colors.ListedColormap(
            plotting.region_colours, N=num_classes
        ),
    )

    plotting.save_segmentation_plot(
        eval_image,
        cm.gray,
        output_dir / Path("gs_pred_and_truth_overlay_plot.png"),
        truth_label_segs,
        gs_pred_segs,
        column_range=range(eval_image.shape[1]),
    )

    plotting.save_segmentation_plot(
        eval_image,
        cm.gray,
        output_dir / Path("gs_predicted_boundaries_ovelay_plot.png"),
        gs_pred_segs,
        predictions=None,
        column_range=range(eval_image.shape[1]),
    )

    hdf5_file.attrs["model_filename"] = np.array(
        eval_params.model_path, dtype="S1000"
    )
    hdf5_file.attrs["image_name"] = np.array(image_name, dtype="S1000")
    hdf5_file.attrs["timestamp"] = np.array(
        common_utils.get_timestamp(), dtype="S1000"
    )
    hdf5_file.attrs["graph_time"] = np.array(graph_time)


def save_eval_config_file(eval_params: EvaluationParameters):
    config_file = h5py.File(
        eval_params.save_foldername / Path("eval_params.hdf5"), "w"
    )
    config_file.attrs["model_filename"] = np.array(
        eval_params.model_path, dtype="S1000"
    )
    config_file.attrs["mlflow_tracking_uri"] = np.array(
        eval_params.mlflow_tracking_uri, dtype="S1000"
    )
    config_file.attrs["test_dataset_path"] = np.array(
        eval_params.test_dataset_path, dtype="S1000"
    )
    config_file.attrs["test_dataset_md5"] = np.array(
        common_utils.md5(eval_params.test_dataset_path), dtype="S1000"
    )

    config_file.attrs["gsgrad"] = np.array(eval_params.gsgrad)
    config_file.close()


def _soft_dice(y_true, y_pred, eps=1e-5):
    """
    c is number of classes
    :param y_pred: b x c x X x Y( x Z...) network output, must sum to 1 over
    c channel (such as after softmax)
    :param y_true: b x c x X x Y( x Z...) one hot encoding of ground truth
    :param eps:
    :return:
    """
    axes = tuple(range(2, len(y_pred.shape)))
    intersect = np.sum(y_pred * y_true, axis=axes)
    denom = np.sum(y_pred + y_true, axis=axes)

    class_dices = ((2.0 * intersect) + eps) / (denom + eps)

    intersect_total = np.sum(np.sum(y_pred * y_true, axis=axes))
    denom_total = np.sum(np.sum(y_pred + y_true, axis=axes))
    overall_dice = ((2.0 * intersect_total) + eps) / (denom_total + eps)
    overall_dice = np.expand_dims(overall_dice, axis=(0, 1))
    return np.concatenate((class_dices, overall_dice), axis=1)


@typechecked
def _calc_overall_dataset_errors(
    eval_params: EvaluationParameters, eval_image_names: list[Path]
):
    """
    'errors' shape: (# of images, number of boundaries, image width)
    'dices' shape: (# of images, number of boundaries + 2 == number of classes
    + 1)
    """
    output_dir = eval_params.save_foldername
    graph_search = eval_params.graph_search
    metrics = eval_params.metrics

    errors = None
    dices = None
    dices_recon = None
    average_surface_distances = None
    average_surface_distances_gt_to_pred = None
    average_surface_distances_pred_to_gt = None
    hausdorff_distances = None

    # Loop through each evaluated image (non graph search)
    dir_list = [
        Path(output_dir) / Path(f"image_{i}")
        for i in range(len(eval_image_names))
    ]
    for obj_name in dir_list:
        eval_filename = (
            output_dir / Path(obj_name) / Path(EVALUATION_RESULTS_FILENAME)
        )
        eval_file = h5py.File(eval_filename, "r")

        @typechecked
        def concat_metric_from_hdf5(
            hdf5_file: h5py.File,
            metric_name: str,
            metric: Union[np.ndarray, None],
        ) -> np.ndarray:
            file_metric = hdf5_file[f"{metric_name}"][:]
            if metric is None:
                metric = np.expand_dims(file_metric, 0)
            else:
                metric = np.concatenate(
                    (metric, np.expand_dims(file_metric, 0)), 0
                )
            return metric

        if DICE_METRIC in metrics:
            dices = concat_metric_from_hdf5(eval_file, "dices", dices)

        if AVERAGE_SURFACE_DISTANCE_METRIC in metrics:
            average_surface_distances = concat_metric_from_hdf5(
                eval_file,
                "average_surface_distances",
                average_surface_distances,
            )
            average_surface_distances_gt_to_pred = concat_metric_from_hdf5(
                eval_file,
                "average_surface_distances_gt_to_pred",
                average_surface_distances_gt_to_pred,
            )
            average_surface_distances_pred_to_gt = concat_metric_from_hdf5(
                eval_file,
                "average_surface_distances_pred_to_gt",
                average_surface_distances_pred_to_gt,
            )

        if HAUSDORFF_DISTANCE_METRIC in metrics:
            hausdorff_distances = concat_metric_from_hdf5(
                eval_file, "hausdorff_distances", hausdorff_distances
            )

    if graph_search:
        for obj_name in dir_list:
            gs_eval_filename = (
                output_dir
                / Path(obj_name)
                / Path(GS_EVALUATION_RESULTS_FILENAME)
            )
            gs_eval_file = h5py.File(gs_eval_filename, "r")
            file_errors = gs_eval_file["errors"]

            if errors is None:
                errors = np.expand_dims(file_errors, 0)
            else:
                errors = np.concatenate(
                    (errors, np.expand_dims(file_errors, 0)), 0
                )

            file_dices_recon = gs_eval_file["gs_dices"][:]
            if dices_recon is None:
                dices_recon = np.expand_dims(file_dices_recon, 0)
            else:
                dices_recon = np.concatenate(
                    (dices_recon, np.expand_dims(file_dices_recon, 0)), 0
                )

    save_filename = output_dir / Path("errors_stats.hdf5")
    save_file = h5py.File(save_filename, "w")

    save_textfilename = output_dir / Path("error_stats.csv")
    save_textfile = open(save_textfilename, "w")

    save_file["image_names"] = np.array(eval_image_names, dtype="S1000")

    @typechecked
    def save_metric(metric_name: str, metric: np.ndarray):
        save_file[metric_name] = metric

        mean_metric = np.nanmean(metric, axis=0)
        sd_metric = np.nanstd(metric, axis=0)

        save_file[f"mean_{metric_name}"] = mean_metric
        save_file[f"sd_{metric_name}"] = sd_metric

        save_textfile.write(f"Mean {metric_name},")
        save_textfile.write(",".join([f"{e:.7f}" for e in mean_metric]) + "\n")

        save_textfile.write(f"SD {metric_name},")
        save_textfile.write(",".join([f"{e:.7f}" for e in sd_metric]) + "\n")

    # Dice
    if DICE_METRIC in metrics:
        save_metric("dices", dices)

    if AVERAGE_SURFACE_DISTANCE_METRIC in metrics:
        save_metric("average_surface_distances", average_surface_distances)
        save_metric(
            "average_surface_distances_gt_to_pred",
            average_surface_distances_gt_to_pred,
        )
        save_metric(
            "average_surface_distances_pred_to_gt",
            average_surface_distances_pred_to_gt,
        )

    if HAUSDORFF_DISTANCE_METRIC in metrics:
        save_metric("hausdorff_distances", hausdorff_distances)

    # Boundary Errors
    if graph_search:
        mean_abs_errors_cols = np.nanmean(np.abs(errors), axis=0)
        mean_abs_errors_samples = np.nanmean(np.abs(errors), axis=2)
        sd_abs_errors_samples = np.nanstd(np.abs(errors), axis=2)
        mean_abs_errors = np.nanmean(mean_abs_errors_samples, axis=0)
        sd_abs_errors = np.nanstd(mean_abs_errors_samples, axis=0)

        median_abs_errors = np.nanmedian(mean_abs_errors_samples, axis=0)

        mean_errors_cols = np.nanmean(errors, axis=0)
        mean_errors_samples = np.nanmean(errors, axis=2)
        mean_errors = np.nanmean(mean_errors_samples, axis=0)
        sd_errors = np.nanstd(mean_errors_samples, axis=0)

        median_errors = np.nanmedian(mean_errors_samples, axis=0)

        save_file["mean_abs_errors_cols"] = mean_abs_errors_cols
        save_file["mean_abs_errors_samples"] = mean_abs_errors_samples
        save_file["mean_abs_errors"] = mean_abs_errors
        save_file["sd_abs_errors"] = sd_abs_errors
        save_file["median_abs_errors"] = median_abs_errors
        save_file["sd_abs_errors_samples"] = sd_abs_errors_samples

        save_file["mean_errors_cols"] = mean_errors_cols
        save_file["mean_errors_samples"] = mean_errors_samples
        save_file["mean_errors"] = mean_errors
        save_file["sd_errors"] = sd_errors
        save_file["median_errors"] = median_errors

        save_file["errors"] = errors

        save_textfile.write("Mean abs errors,")
        save_textfile.write(
            ",".join([f"{e:.7f}" for e in mean_abs_errors]) + "\n"
        )

        save_textfile.write("Mean errors,")
        save_textfile.write(",".join([f"{e:.7f}" for e in mean_errors]) + "\n")

        save_textfile.write("Median absolute errors,")
        save_textfile.write(
            ",".join([f"{e:.7f}" for e in median_abs_errors]) + "\n"
        )

        save_textfile.write("SD abs errors,")
        save_textfile.write(
            ",".join([f"{e:.7f}" for e in sd_abs_errors]) + "\n"
        )

        save_textfile.write("SD errors,")
        save_textfile.write(",".join([f"{e:.7f}" for e in sd_errors]) + "\n")

        # Reconstructed Dice Coeff.
        save_file["dices_recon"] = dices_recon

        mean_dices_recon = np.mean(dices_recon, axis=0)
        sd_dices_recon = np.std(dices_recon, axis=0)

        save_file["mean_dices_recon"] = mean_dices_recon
        save_file["sd_dices_recon"] = sd_dices_recon

        save_textfile.write("Mean dices recon,")
        save_textfile.write(
            ",".join([f"{e:.7f}" for e in mean_dices_recon]) + "\n"
        )

        save_textfile.write("SD dices recon,")
        save_textfile.write(
            ",".join([f"{e:.7f}" for e in sd_dices_recon]) + "\n"
        )

    save_file.close()
    save_textfile.close()
