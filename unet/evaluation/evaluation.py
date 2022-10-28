from __future__ import annotations

import os

import h5py
import logging as log
from matplotlib import cm
import numpy as np
from pathlib import Path
from typeguard import typechecked
from tensorflow.keras.utils import to_categorical
import time

from unet.min_path_processing import utils
from unet.evaluation.evaluation_parameters import EvaluationParameters
from unet.common import (
    dataset_construction as datacon,
    plotting,
    utils as common_utils,
)
from unet.min_path_processing import graph_search

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
        delineations: np.ndarray,
        errors: np.ndarray,
        mean_abs_err: np.ndarray,
        mean_err: np.ndarray,
        abs_err_sd: np.ndarray,
        err_sd: np.ndarray,
    ) -> None:
        self.image = image
        self.image_name = image_name
        self.image_segments = image_segments
        self.image_output_dir = image_output_dir
        self.predicted_labels = predicted_labels
        self.categorical_pred = categorical_pred
        self.boundary_maps = boundary_maps
        self.delineations = delineations
        self.errors = errors
        self.mean_abs_err = mean_abs_err
        self.mean_err = mean_err
        self.abs_err_sd = abs_err_sd
        self.err_sd = err_sd


@typechecked
def evaluate_model(
    eval_params: EvaluationParameters,
) -> list[EvaluationOutput]:
    dataset = eval_params.dataset
    eval_images = dataset.images
    eval_labels = dataset.image_masks
    eval_image_names = dataset.image_names
    eval_image_output_dirs = dataset.image_output_dirs

    eval_segments = np.swapaxes(
        utils.generate_boundary(np.squeeze(eval_labels, axis=3), axis=2), 0, 1
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

        print(
            "Evaluating image number: "
            + str(ind + 1)
            + " ("
            + str(eval_image_name)
            + ")..."
        )

        log.info("Running network predictions...")

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

        print("Converting predictions to boundary maps...")

        # convert predictions to usable boundary probability maps
        """
        predicted_labels: A matrix of shape (1, image_width, image_height) that
        contains the predicted classes numbered from 0 to num_classes - 1.

        categorical_pred: A matrix of shape (1, num_classes, image_width,
        image_height) that contains:
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
        imgage_width, image_height). Each "image" contains the delineation of
        the ith layer as a line marked with 255. The rest of the elements are
        0.
        """
        boundary_maps = common_utils.convert_predictions_to_maps_semantic(
            categorical_pred,
            bg_ilm=True,
            bg_csi=False,
        )

        dices = _calc_dice(
            categorical_pred, np.expand_dims(eval_label, axis=0)
        )

        predicted_labels = np.squeeze(predicted_labels)
        categorical_pred = np.squeeze(categorical_pred)
        boundary_maps = np.squeeze(boundary_maps)

        # save data to files
        _save_image_evaluation_results(
            eval_params,
            eval_image,
            eval_image_name,
            predicted_labels,
            categorical_pred,
            eval_label,
            eval_seg,
            dices,
            predict_time,
            eval_image_output_dir,
        )

        print("Running graph search, segmenting boundary maps...")
        graph_structure = graph_search.create_graph_structure(eval_image.shape)

        start_graph_time = time.time()

        """
        delineations: A matrix of shape (# of classes (layers) - 1,
        imgage_width) where delineations[i, col] specifies the row
        (y-dimension) of the ith layer

        errors: A matrix of shape (# of classes (layers) - 1, imgage_width)
        where error[i, col] specifies the pixel difference between delineation
        (i.e. prediction) and the true label (i.e. eval_seg)

        The third element in the tuple (originally named 'trim_maps'): A matrix
        of shape (# of classes (layers) - 1, imgage_width, image_height). It is
        the normalized version (0 to 1) of the 'boundary_maps' input.
        """
        delineations, errors, _ = graph_search.segment_maps(
            boundary_maps,
            eval_seg,
            graph_structure,
        )

        reconstructed_maps = datacon.create_area_mask(
            eval_image.shape, delineations
        )
        reconstructed_maps = to_categorical(
            reconstructed_maps, num_classes=eval_params.num_classes
        )
        reconstructed_maps = np.expand_dims(reconstructed_maps, axis=0)

        [eval_label_gs, reconstructed_maps] = common_utils.perform_argmax(
            reconstructed_maps
        )

        gs_dices = _calc_dice(
            reconstructed_maps,
            np.expand_dims(eval_label, axis=0),
        )

        eval_label_gs = np.squeeze(eval_label_gs)

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
            eval_label_gs,
            delineations,
            eval_seg,
            gs_dices,
            errors,
            mean_abs_err,
            mean_err,
            abs_err_sd,
            err_sd,
            graph_time,
            eval_image_output_dir,
        )

        eval_output = EvaluationOutput(
            image=eval_image,
            image_name=eval_image_name,
            image_segments=eval_seg,
            image_output_dir=eval_image_output_dir,
            predicted_labels=predicted_labels,
            categorical_pred=categorical_pred,
            boundary_maps=boundary_maps,
            delineations=delineations,
            errors=errors,
            mean_abs_err=mean_abs_err,
            mean_err=mean_err,
            abs_err_sd=abs_err_sd,
            err_sd=err_sd,
        )

        eval_outputs.append(eval_output)

        _print_error_summary(
            mean_abs_err, mean_err, abs_err_sd, err_sd, eval_image_name
        )
        print(
            "DONE image number: "
            + str(ind + 1)
            + " ("
            + str(eval_image_name)
            + ")..."
        )
        print("______________________________")

    _calc_overall_dataset_errors(eval_image_names, eval_params.save_foldername)

    return eval_outputs


@typechecked
def _save_image_evaluation_results(
    eval_params: EvaluationParameters,
    eval_image: np.ndarray,
    image_name: Path,
    predicted_labels: np.ndarray,
    categorical_pred: np.ndarray,
    eval_labels: np.ndarray,
    eval_segs: np.ndarray,
    dices: np.ndarray,
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
            "non_gs_predicted_labels", data=predicted_labels, dtype="uint8"
        )

        if eval_params.save_params.png_images is True:
            plotting.save_image_plot(
                predicted_labels,
                output_dir / Path("non_gs_predicted_labels.png"),
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
        output_dir / Path("truth_labels.png"),
        cmap=plotting.colors.ListedColormap(
            plotting.region_colours, N=len(categorical_pred)
        ),
    )

    hdf5_file.create_dataset("raw_segs", data=eval_segs, dtype="uint16")
    hdf5_file.create_dataset("dices", data=np.squeeze(dices), dtype="float64")

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
    eval_label_gs: np.ndarray,
    delineations: np.ndarray,
    label_seg: np.ndarray,
    gs_dices: np.ndarray,
    errors: np.ndarray,
    mean_abs_err: np.ndarray,
    mean_err: np.ndarray,
    abs_err_sd: np.ndarray,
    err_sd: np.ndarray,
    graph_time: float,
    output_dir: Path,
):
    num_classes = delineations.shape[0] + 1
    # Save graph search based prediction results
    hdf5_file = h5py.File(
        output_dir / Path(GS_EVALUATION_RESULTS_FILENAME), "w"
    )

    np.savetxt(
        output_dir / Path("boundaries.csv"),
        delineations,
        delimiter=",",
        fmt="%d",
    )

    np.savetxt(
        output_dir / Path("segmentation_map.csv"),
        np.transpose(eval_label_gs),
        fmt="%d",
        delimiter=",",
    )

    hdf5_file.create_dataset("delineations", data=delineations, dtype="uint16")
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
        "gs_predicted_labels", data=eval_label_gs, dtype="uint8"
    )

    plotting.save_image_plot(
        eval_label_gs,
        output_dir / Path("gs_predicted_labels.png"),
        cmap=plotting.colors.ListedColormap(
            plotting.region_colours, N=num_classes
        ),
    )

    plotting.save_segmentation_plot(
        eval_image,
        cm.gray,
        output_dir / Path("gs_truth_pred_overlay_plot.png"),
        label_seg,
        delineations,
        column_range=range(eval_image.shape[0]),
    )

    plotting.save_segmentation_plot(
        eval_image,
        cm.gray,
        output_dir / Path("truth_plot.png"),
        label_seg,
        predictions=None,
        column_range=range(eval_image.shape[0]),
    )

    plotting.save_segmentation_plot(
        eval_image,
        cm.gray,
        output_dir / Path("truth_boundaries_ovelay_plot.png"),
        delineations,
        predictions=None,
        column_range=range(eval_image.shape[0]),
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
        eval_params.save_foldername / Path("config.hdf5"), "w"
    )

    config_file.attrs["model_filename"] = np.array(
        eval_params.model_path.name, dtype="S100"
    )

    config_file.attrs["gsgrad"] = np.array(eval_params.gsgrad)
    config_file.close()


@typechecked
def _print_error_summary(
    mean_abs_err: np.ndarray,
    mean_err: np.ndarray,
    abs_err_sd: np.ndarray,
    err_sd: np.ndarray,
    cur_image_name: Path,
):
    num_boundaries = mean_abs_err.shape[0]

    # overall errors: list of four numpy arrays: [mean abs error,
    # mean error, abs error sd, error sd]
    print("\n")
    print("Error summary for image: " + str(cur_image_name))
    print("_" * 92)
    print(
        "BOUNDARY".center(30)
        + "|"
        + "Mean absolute error [px]".center(30)
        + "|"
        + "Mean error [px]".center(30)
    )
    print("_" * 92)
    for boundary_ind in range(num_boundaries):
        mae = mean_abs_err[boundary_ind]
        me = mean_err[boundary_ind]
        ae_sd = abs_err_sd[boundary_ind]
        e_sd = err_sd[boundary_ind]
        first_col_str = "{:.2f} ({:.2f})".format(mae, ae_sd)
        second_col_str = "{:.2f} ({:.2f})".format(me, e_sd)
        print(
            f"boundary_{boundary_ind}".center(30)
            + "|"
            + first_col_str.center(30)
            + "|"
            + second_col_str.center(30)
        )
    print("\n")


def _soft_dice_numpy(y_pred, y_true, eps=1e-7):
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

    for i in range(intersect.shape[1]):
        # if there is no region for a class to predict, there shouldn't be a
        #  penalty for correctly predicting empty
        if intersect[0, i] == 0 and denom[0, i] == 0:
            # set to 1
            # intersect[0, i] = 0.5
            # denom[0, i] = 1 - eps

            # OR

            # set to NaN
            intersect[0, i] = np.nan

    class_dices = 2.0 * intersect / (denom + eps)

    intersect_total = np.sum(np.sum(y_pred * y_true, axis=axes))
    denom_total = np.sum(np.sum(y_pred + y_true, axis=axes))

    overall_dice = np.array(
        [np.array([2.0 * intersect_total / (denom_total + eps)])]
    )

    return np.concatenate((class_dices, overall_dice), axis=1)


@typechecked
def _calc_dice(predictions: np.ndarray, labels: np.ndarray):
    dices = _soft_dice_numpy(
        predictions,
        np.transpose(labels, axes=(0, 3, 1, 2)),
    )

    return dices


@typechecked
def _calc_overall_dataset_errors(
    eval_image_names: list[Path], output_dir: Path
):
    """
    'errors' shape: (# of images, number of boundaries, image width)
    'dices' shape: (# of images, number of boundaries + 2 == number of classes
    + 1)
    """
    errors = None
    dices = None
    dices_recon = None

    dir_list = os.listdir(output_dir)
    # Loop through each evaluated image
    for obj_name in dir_list:
        if os.path.isdir(output_dir / Path(obj_name)):
            eval_filename = (
                output_dir / Path(obj_name) / Path(EVALUATION_RESULTS_FILENAME)
            )
            eval_file = h5py.File(eval_filename, "r")
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

            file_dices = eval_file["dices"][:]
            if dices is None:
                dices = np.expand_dims(file_dices, 0)
            else:
                dices = np.concatenate(
                    (dices, np.expand_dims(file_dices, 0)), 0
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

    # Boundary Errors
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
    save_textfile.write(",".join([f"{e:.7f}" for e in mean_abs_errors]) + "\n")

    save_textfile.write("Mean errors,")
    save_textfile.write(",".join([f"{e:.7f}" for e in mean_errors]) + "\n")

    save_textfile.write("Median absolute errors,")
    save_textfile.write(
        ",".join([f"{e:.7f}" for e in median_abs_errors]) + "\n"
    )

    save_textfile.write("SD abs errors,")
    save_textfile.write(",".join([f"{e:.7f}" for e in sd_abs_errors]) + "\n")

    save_textfile.write("SD errors,")
    save_textfile.write(",".join([f"{e:.7f}" for e in sd_errors]) + "\n")

    # Dice
    save_file["dices"] = dices

    mean_dices = np.nanmean(dices, axis=0)
    sd_dices = np.nanstd(dices, axis=0)

    save_file["mean_dices"] = mean_dices
    save_file["sd_dices"] = sd_dices

    save_textfile.write("Mean dices,")
    save_textfile.write(",".join([f"{e:.7f}" for e in mean_dices]) + "\n")

    save_textfile.write("SD dices,")
    save_textfile.write(",".join([f"{e:.7f}" for e in sd_dices]) + "\n")

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
    save_textfile.write(",".join([f"{e:.7f}" for e in sd_dices_recon]) + "\n")

    save_file.close()
    save_textfile.close()
