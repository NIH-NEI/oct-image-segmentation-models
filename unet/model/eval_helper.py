import os
import time

import h5py
from matplotlib import cm
import numpy as np
from pathlib import Path
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical

from unet.common import dataset_construction as datacon, plotting, utils
from unet.min_path_processing import graph_search
from unet.model import data_generator
from unet.model import evaluation_output as eoutput
from unet.model import evaluation_parameters as eparams
from unet.model import image_database as image_db
from unet.model import results_collation


def evaluate_network(
    imdb,
    eval_params,
):
    if eval_params.save_params.disable is False:
        save_eval_config_file(eval_params, imdb)

    eval_outputs = evaluate_semantic_network(eval_params, imdb)

    if not eval_params.save_params.disable and eval_params.collate_results and eval_params.is_evaluate:
        if eval_params.eval_mode == "network" or eval_params.boundaries is False or eval_params.boundary_errors is False:
            inc_boundary_errors = False
        else:
            inc_boundary_errors = True

        results_collation.calc_overall_dataset_errors(
            eval_params.save_foldername,
            inc_boundary_errors=inc_boundary_errors,
            inc_dice=eval_params.dice_errors,
        )

    return eval_outputs


def evaluate_semantic_network(eval_params, imdb):
    # pass images to network one at a time
    for ind in imdb.image_range:
        eval_output = eoutput.EvaluationOutput()

        cur_raw_image = imdb.get_image(ind)
        cur_label = imdb.get_label(ind)
        cur_image_name = str(imdb.get_image_name(ind))
        cur_seg = None
        if eval_params.is_evaluate:
            cur_seg = imdb.get_seg(ind)

        eval_output.raw_image = cur_raw_image
        eval_output.raw_label = cur_label
        eval_output.image_name = cur_image_name
        eval_output.raw_seg = cur_seg

        if eval_params.verbosity >= 2:
            print(
                "Evaluating image number: "
                + str(ind + 1)
                + " ("
                + cur_image_name
                + ")..."
            )

        # PERFORM STEP 1: evaluate/predict patches with network
        if eval_params.verbosity >= 2:
            print("Augmenting data using augmentation: " + eval_params.aug_desc + "...")

        aug_fn = eval_params.aug_fn_arg[0]
        aug_arg = eval_params.aug_fn_arg[1]

        # augment raw full sized image and label
        augment_image, augment_label, augment_seg, _, augment_time = aug_fn(
            cur_raw_image, cur_label, cur_seg, aug_arg, sample_ind=ind, set=imdb.set
        )

        eval_output.aug_image = augment_image
        eval_output.aug_label = augment_label
        eval_output.aug_seg = augment_seg

        if eval_params.verbosity >= 2:
            print("Running network predictions...")

        images = np.expand_dims(augment_image, axis=0)
        labels = np.expand_dims(augment_label, axis=0)

        single_image_imdb = image_db.ImageDatabase(images=images, labels=labels)

        # use a generator to supply data to model (predict_generator)

        start_gen_time = time.time()
        gen = data_generator.DataGenerator(
            single_image_imdb,
            1,
            aug_fn_args=[],
            aug_mode="none",
            aug_probs=[],
            aug_fly=False,
            shuffle=False,
            transpose=eval_params.transpose,
            normalise=eval_params.normalise_input,
        )
        end_gen_time = time.time()
        gen_time = end_gen_time - start_gen_time

        start_predict_time = time.time()

        predicted_labels = eval_params.loaded_model.predict(
            gen, verbose=eval_params.predict_verbosity
        )

        end_predict_time = time.time()

        predict_time = end_predict_time - start_predict_time

        activations = None
        layer_outputs = None

        if eval_params.verbosity >= 2:
            print("Converting predictions to boundary maps...")

        if eval_params.transpose is True:
            predicted_labels = np.transpose(predicted_labels, axes=(0, 2, 1, 3))

        # convert predictions to usable boundary probability maps

        start_convert_time = time.time()

        [comb_area_map, area_maps] = utils.perform_argmax(
            predicted_labels, bin=eval_params.binarize
        )

        eval_output.comb_area_map = comb_area_map
        eval_output.area_maps = area_maps

        if (
            eval_params.boundaries is False
            or eval_params.save_params.boundary_maps is False
        ):
            boundary_maps = None
        else:
            boundary_maps = utils.convert_predictions_to_maps_semantic(
                np.array(area_maps),
                bg_ilm=eval_params.bg_ilm,
                bg_csi=eval_params.bg_csi,
            )

        eval_output.boundary_maps = boundary_maps

        end_convert_time = time.time()
        convert_time = end_convert_time - start_convert_time

        if eval_params.is_evaluate and eval_params.dice_errors is True:
            dices = calc_dice(eval_params, area_maps, labels)
        else:
            dices = None

        area_maps = np.squeeze(area_maps)
        comb_area_map = np.squeeze(comb_area_map)
        boundary_maps = np.squeeze(boundary_maps)

        # save data to files
        if eval_params.save_params.disable is False:
            intermediate_save_semantic(
                eval_params,
                imdb,
                cur_image_name,
                boundary_maps,
                predict_time,
                augment_time,
                gen_time,
                augment_image,
                augment_label,
                augment_seg,
                cur_raw_image,
                cur_label,
                cur_seg,
                area_maps,
                comb_area_map,
                dices,
                convert_time,
                activations,
                layer_outputs,
                eval_params.dataset.images_output_dirs[ind]
            )

        if eval_params.boundaries is True and (
            eval_params.eval_mode == "both" or eval_params.eval_mode == "gs"
        ):
            cur_image_name = imdb.get_image_name(ind)
            cur_seg = imdb.get_seg(ind)
            cur_raw_image = imdb.get_image(ind)
            cur_label = imdb.get_label(ind)

            aug_fn = eval_params.aug_fn_arg[0]
            aug_arg = eval_params.aug_fn_arg[1]

            # augment raw full sized image and label
            augment_image, augment_label, augment_seg, _, _ = aug_fn(
                cur_raw_image, cur_label, cur_seg, aug_arg, sample_ind=ind, set=imdb.set
            )

            if (
                eval_params.is_evaluate
                and eval_params.save_params.disable is False
                and eval_params.save_params.boundary_maps is True
            ):
                boundary_maps = load_dataset_extra(
                    eval_params.dataset.images_output_dirs[ind], cur_image_name, "boundary_maps"
                )
                if eval_params.is_evaluate and eval_params.dice_errors is True:
                    dices = load_dataset(eval_params.dataset.images_output_dirs[ind], cur_image_name, "dices")
                else:
                    dices = None

            # PERFORM STEP 2: segment probability maps using graph search
            eval_output = eval_second_step(
                eval_params,
                boundary_maps,
                augment_seg,
                cur_image_name,
                augment_image,
                augment_label,
                imdb,
                dices,
                eval_output,
                eval_params.dataset.images_output_dirs[ind]
            )

        elif eval_params.boundaries is False:
            if (
                eval_params.save_params.disable is False
                and eval_params.save_params.attributes is True
            ):
                save_final_attributes(eval_params, cur_image_name, graph_time=None, output_dir=eval_params.dataset.images_output_dirs[ind])

        if (
            eval_params.save_params.disable is False
            and eval_params.save_params.temp_extra is True
        ):
            delete_loadsaveextra_file(
                eval_params.dataset.images_output_dirs[ind],
                cur_image_name
            )

        if eval_params.verbosity >= 2:
            print(
                "DONE image number: "
                + str(ind + 1)
                + " ("
                + str(cur_image_name)
                + ")..."
            )
            print("______________________________")


def eval_second_step(
    eval_params: eparams.EvaluationParameters,
    prob_maps,
    cur_seg,
    cur_image_name: Path,
    cur_augment_image,
    cur_augment_label,
    imdb,
    dices,
    eval_output,
    output_dir,
):
    if eval_params.verbosity >= 2:
        print("Running graph search, segmenting boundary maps...")

    truths = cur_seg
    graph_structure = graph_search.create_graph_structure(
        (imdb.image_width, imdb.image_height), max_grad=eval_params.gsgrad
    )

    start_graph_time = time.time()
    delineations, errors, trim_maps = graph_search.segment_maps(
        prob_maps,
        truths,
        graph_structure,
        eval_params
    )

    reconstructed_maps = datacon.create_area_mask(cur_augment_image.shape, delineations)
    reconstructed_maps = to_categorical(
        reconstructed_maps, num_classes=imdb.num_classes
    )
    reconstructed_maps = np.expand_dims(reconstructed_maps, axis=0)

    [comb_area_map_recalc, reconstructed_maps] = utils.perform_argmax(reconstructed_maps)

    if eval_params.is_evaluate and eval_params.dice_errors == True:
        recalc_dices = calc_dice(
            eval_params, reconstructed_maps, np.expand_dims(cur_augment_label, axis=0)
        )

    comb_area_map_recalc = np.squeeze(comb_area_map_recalc)

    if eval_params.flatten_image is True:
        [flattened_image, offsets, flatten_boundary] = datacon.flatten_image_boundary(
            cur_augment_image,
            delineations[eval_params.flatten_ind],
            poly=eval_params.flatten_poly,
        )
        if eval_params.save_params.output_var is True:
            eval_output.flattened_image = flattened_image
            eval_output.offsets = offsets
            eval_output.flatten_boundary = flatten_boundary

    if eval_params.save_params.output_var is True:
        eval_output.delineations = delineations
        eval_output.errors = errors
        eval_output.trim_maps = trim_maps

    end_graph_time = time.time()
    graph_time = end_graph_time - start_graph_time

    # TODO code to calculate dice overlap of areas
    overall_errors = graph_search.calculate_overall_errors(
        errors, eval_params.col_error_range
    )

    if eval_params.verbosity == 3 and truths is not None:
        print_error_summary(overall_errors, imdb, cur_image_name)

    [mean_abs_err, mean_err, abs_err_sd, err_sd] = overall_errors

    # FINAL SAVE
    if eval_params.save_params.disable is False:
        if eval_params.save_params.delineations is True:
            np.savetxt(output_dir / cur_image_name.stem / Path("boundaries.csv"), delineations, delimiter=",", fmt="%d")

            np.savetxt(
                output_dir / cur_image_name.stem / Path("segmentation_map.csv"),
                np.transpose(comb_area_map_recalc), fmt="%d", delimiter=",")

            save_dataset(
                output_dir, cur_image_name, "delineations", "uint16", delineations
            )
        if eval_params.save_params.errors is True and truths is not None:
            save_dataset(output_dir, cur_image_name, "errors", "float64", errors)
            save_dataset(
                output_dir, cur_image_name, "mean_abs_err", "float64", mean_abs_err
            )
            save_dataset(output_dir, cur_image_name, "mean_err", "float64", mean_err)
            save_dataset(
                output_dir, cur_image_name, "abs_err_sd", "float64", abs_err_sd
            )
            save_dataset(output_dir, cur_image_name, "err_sd", "float64", err_sd)
            if dices is not None:
                save_dataset(output_dir, cur_image_name, "dices", "float64", dices)
                save_dataset(
                    output_dir,
                    cur_image_name,
                    "dices_recon",
                    "float64",
                    np.squeeze(recalc_dices),
                )

        if eval_params.dice_errors == True:
            save_dataset_extra(
                output_dir,
                cur_image_name,
                "comb_area_map_recalc",
                "uint8",
                comb_area_map_recalc,
            )

            plotting.save_image_plot(
                comb_area_map_recalc,
                get_loadsave_path(output_dir, cur_image_name) / Path("comb_area_map_recalc.png"),
                cmap=plotting.colors.ListedColormap(
                    plotting.region_colours, N=len(np.squeeze(reconstructed_maps))
                ),
            )

        if eval_params.save_params.seg_plot is True:
            if eval_params.save_params.pngimages is True:
                if truths is not None:
                    plotting.save_segmentation_plot(
                        cur_augment_image,
                        cm.gray,
                        get_loadsave_path(output_dir, cur_image_name) / Path("seg_plot.png"),
                        cur_seg,
                        delineations,
                        column_range=eval_params.col_error_range,
                    )
                    plotting.save_segmentation_plot(
                        cur_augment_image,
                        cm.gray,
                        get_loadsave_path(output_dir, cur_image_name) / Path("truth_plot.png"),
                        cur_seg,
                        predictions=None,
                        column_range=eval_params.col_error_range,
                    )

            plotting.save_segmentation_plot(
                cur_augment_image,
                cm.gray,
                get_loadsave_path(output_dir, cur_image_name)
                / Path("delin_plot.png"),
                delineations,
                predictions=None,
                column_range=eval_params.col_error_range,
            )

        if eval_params.save_params.indiv_seg_plot is True:
            if eval_params.save_params.pngimages is True:
                for i in range(delineations.shape[0]):
                    if truths is not None:
                        plotting.save_segmentation_plot(
                            cur_augment_image,
                            cm.gray,
                            get_loadsave_path(
                                output_dir, cur_image_name
                            )
                            + "/"
                            + "seg_plot_"
                            + str(i)
                            + ".png",
                            np.expand_dims(cur_seg[i], axis=0),
                            np.expand_dims(delineations[i], axis=0),
                            column_range=eval_params.col_error_range,
                            color="#ffe100",
                            linewidth=2.0,
                        )
                        plotting.save_segmentation_plot(
                            cur_augment_image,
                            cm.gray,
                            get_loadsave_path(
                                output_dir, cur_image_name
                            )
                            + "/"
                            + "truth_plot_"
                            + str(i)
                            + ".png",
                            np.expand_dims(cur_seg[i], axis=0),
                            predictions=None,
                            column_range=eval_params.col_error_range,
                            color="#ffe100",
                            linewidth=2.0,
                        )

                    plotting.save_segmentation_plot(
                        cur_augment_image,
                        cm.gray,
                        get_loadsave_path(output_dir, cur_image_name)
                        + "/"
                        + "delin_plot_"
                        + str(i)
                        + ".png",
                        np.expand_dims(delineations[i], axis=0),
                        predictions=None,
                        column_range=eval_params.col_error_range,
                        color="#ffe100",
                        linewidth=2.0,
                    )

        if eval_params.save_params.pair_seg_plot is True:
            if eval_params.save_params.pngimages is True:
                for i in range(delineations.shape[0] - 1):
                    if truths is not None:
                        plotting.save_segmentation_plot(
                            cur_augment_image,
                            cm.gray,
                            get_loadsave_path(
                                output_dir, cur_image_name
                            )
                            + "/"
                            + "seg_plot_pair"
                            + str(i)
                            + ".png",
                            cur_seg[i : i + 2],
                            np.expand_dims(delineations[i], axis=0),
                            column_range=eval_params.col_error_range,
                            color="#ffe100",
                            linewidth=2.0,
                        )
                        plotting.save_segmentation_plot(
                            cur_augment_image,
                            cm.gray,
                            get_loadsave_path(
                                output_dir, cur_image_name
                            )
                            + "/"
                            + "truth_plot_pair"
                            + str(i)
                            + ".png",
                            cur_seg[i : i + 2],
                            predictions=None,
                            column_range=eval_params.col_error_range,
                            color="#ffe100",
                            linewidth=2.0,
                        )

                    plotting.save_segmentation_plot(
                        cur_augment_image,
                        cm.gray,
                        get_loadsave_path(output_dir, cur_image_name)
                        / Path("delin_plot_pair" + str(i) + ".png"),
                        delineations[i : i + 2],
                        predictions=None,
                        column_range=eval_params.col_error_range,
                        color="#ffe100",
                        linewidth=2.0,
                    )

        if eval_params.save_params.ret_seg_plot is True:
            if eval_params.save_params.pngimages is True:
                if truths is not None:
                    plotting.save_segmentation_plot(
                        cur_augment_image,
                        cm.gray,
                        get_loadsave_path(output_dir, cur_image_name)
                        + "/"
                        + "seg_plot_ret.png",
                        np.concatenate(
                            [
                                np.expand_dims(cur_seg[0], axis=0),
                                np.expand_dims(cur_seg[6], axis=0),
                            ],
                            axis=0,
                        ),
                        np.expand_dims(delineations[i], axis=0),
                        column_range=eval_params.col_error_range,
                        color="#ffe100",
                        linewidth=2.0,
                    )
                    plotting.save_segmentation_plot(
                        cur_augment_image,
                        cm.gray,
                        get_loadsave_path(output_dir, cur_image_name)
                        + "/"
                        + "truth_plot_ret.png",
                        np.concatenate(
                            [
                                np.expand_dims(cur_seg[0], axis=0),
                                np.expand_dims(cur_seg[6], axis=0),
                            ],
                            axis=0,
                        ),
                        predictions=None,
                        column_range=eval_params.col_error_range,
                        color="#ffe100",
                        linewidth=2.0,
                    )

                plotting.save_segmentation_plot(
                    cur_augment_image,
                    cm.gray,
                    get_loadsave_path(output_dir, cur_image_name)
                    + "/"
                    + "delin_plot_ret.png",
                    np.concatenate(
                        [
                            np.expand_dims(delineations[0], axis=0),
                            np.expand_dims(delineations[6], axis=0),
                        ],
                        axis=0,
                    ),
                    predictions=None,
                    column_range=eval_params.col_error_range,
                    color="#ffe100",
                    linewidth=2.0,
                )

        if eval_params.save_params.attributes is True:
            save_final_attributes(eval_params, cur_image_name, graph_time, output_dir)
        if eval_params.save_params.error_plot is True:
            # TODO implement error profile plots
            pass

        if (
            eval_params.save_params.flatten_image is True
            and eval_params.flatten_image is True
        ):
            save_dataset_extra(
                output_dir,
                cur_image_name,
                "flatten_boundary",
                "uint16",
                flatten_boundary,
            )
            save_dataset_extra(
                output_dir, cur_image_name, "flatten_image", "uint8", flattened_image
            )
            save_dataset_extra(
                output_dir, cur_image_name, "flatten_offsets", "uint16", offsets
            )
            if eval_params.save_params.pngimages is True:
                plotting.save_image_plot(
                    flattened_image,
                    get_loadsave_path(output_dir, cur_image_name)
                    + "/flattened_image.png",
                    cmap=cm.gray,
                    vmin=0,
                    vmax=255,
                )
                plotting.save_segmentation_plot(
                    cur_augment_image,
                    cm.gray,
                    get_loadsave_path(output_dir, cur_image_name)
                    + "/"
                    + "flatten_boundary_plot.png",
                    np.expand_dims(flatten_boundary, axis=0),
                    predictions=None,
                    column_range=eval_params.col_error_range,
                )

        if (
            eval_params.boundaries is True
            and eval_params.save_params.boundary_maps is True
            and eval_params.trim_maps is True
        ):
            save_dataset_extra(
                output_dir, cur_image_name, "trim_maps", "uint8", trim_maps
            )
            if eval_params.save_params.pngimages is True:
                for map_ind in range(len(trim_maps)):
                    boundary_name = imdb.get_boundary_name(map_ind)
                    plotting.save_image_plot(
                        trim_maps[map_ind],
                        get_loadsave_path(output_dir, cur_image_name)
                        + "/trim_map_"
                        + boundary_name
                        + ".png",
                        cmap=cm.Blues,
                    )

    return eval_output


def save_eval_config_file(eval_params, imdb):
    config_file = h5py.File(str(eval_params.save_foldername) + "/config.hdf5", "w")

    config_file.attrs["model_filename"] = np.array(
        eval_params.model_file_path.name, dtype="S100"
    )
    config_file.attrs["data_filename"] = np.array(imdb.filename, dtype="S100")

    aug_fn = eval_params.aug_fn_arg[0]
    aug_arg = eval_params.aug_fn_arg[1]

    config_file.attrs["aug"] = np.array(aug_fn.__name__, dtype="S100")

    for aug_param_key in aug_arg.keys():
        val = str(aug_arg[aug_param_key])

        config_file.attrs["aug_param: " + aug_param_key] = np.array(val, dtype="S100")

    if eval_params.patch_size is not None:
        config_file.attrs["patch_size"] = np.array(eval_params.patch_size)

    config_file.attrs["normalise_input"] = np.array(eval_params.normalise_input)

    config_file.attrs["boundaries"] = np.array(eval_params.boundaries)

    if eval_params.boundaries is True:
        config_file.attrs["gsgrad"] = np.array(eval_params.gsgrad)
        config_file.attrs["error_col_inc_range"] = np.array(
            (eval_params.col_error_range[0], eval_params.col_error_range[-1])
        )

    config_file.close()


def print_error_summary(overall_errors, imdb, cur_image_name):
    num_boundaries = overall_errors[0].shape[0]

    # overall errors: list of four numpy arrays: [mean abs error, mean error, abs error sd, error sd]
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
        boundary_name = imdb.get_boundary_name(boundary_ind)
        mae = overall_errors[0][boundary_ind]
        me = overall_errors[1][boundary_ind]
        ae_sd = overall_errors[2][boundary_ind]
        e_sd = overall_errors[3][boundary_ind]
        first_col_str = "{:.2f} ({:.2f})".format(mae, ae_sd)
        second_col_str = "{:.2f} ({:.2f})".format(me, e_sd)
        print(
            boundary_name.center(30)
            + "|"
            + first_col_str.center(30)
            + "|"
            + second_col_str.center(30)
        )
    print("\n")


def intermediate_save_semantic(
    eval_params,
    imdb,
    cur_image_name,
    prob_maps,
    predict_time,
    augment_time,
    gen_time,
    augment_image,
    augment_labels,
    augment_segs,
    cur_raw_image,
    cur_labels,
    cur_seg,
    area_maps,
    comb_area_map,
    dices,
    convert_time,
    activations,
    layer_outputs,
    output_dir,
):
    if eval_params.save_params.area_maps is True:
        save_dataset_extra(output_dir, cur_image_name, "area_maps", "uint8", area_maps)

        if eval_params.save_params.pngimages is True:
            for map_ind in range(len(area_maps)):
                plotting.save_image_plot(
                    area_maps[map_ind],
                    get_loadsave_path(output_dir, cur_image_name)
                    / Path(
                        "area_map_" + imdb.get_fullsize_class_name(map_ind) + ".png"
                    ),
                    cmap=cm.Blues,
                )
    if eval_params.save_params.comb_area_maps is True:
        save_dataset_extra(
            output_dir, cur_image_name, "comb_area_map", "uint8", comb_area_map
        )

        if eval_params.save_params.pngimages is True:
            plotting.save_image_plot(
                comb_area_map,
                get_loadsave_path(output_dir, cur_image_name)
                / Path("comb_area_map.png"),
                cmap=plotting.colors.ListedColormap(
                    plotting.region_colours, N=len(area_maps)
                ),
            )
        if eval_params.save_params.crop_map is True:
            plotting.save_image_plot_crop(
                comb_area_map,
                get_loadsave_path(output_dir, cur_image_name)
                / Path("comb_area_map_crop.png"),
                cmap=plotting.colors.ListedColormap(
                    plotting.region_colours, N=len(area_maps)
                ),
                crop_bounds=eval_params.save_params.crop_bounds,
            )
    if eval_params.boundaries is True and eval_params.save_params.boundary_maps is True:
        save_dataset_extra(
            output_dir, cur_image_name, "boundary_maps", "uint8", prob_maps
        )
        if (
            eval_params.save_params.pngimages is True
            and eval_params.save_params.indivboundarypngs
        ):
            for map_ind in range(len(prob_maps)):
                boundary_name = imdb.get_boundary_name(map_ind)
                plotting.save_image_plot(
                    prob_maps[map_ind],
                    get_loadsave_path(output_dir, cur_image_name)
                    / Path("boundary_map_" + boundary_name + ".png"),
                    cmap=cm.Blues,
                )
    if eval_params.save_params.raw_image is True:
        save_dataset_extra(
            output_dir, cur_image_name, "raw_image", "uint8", cur_raw_image
        )
        if cur_raw_image.shape[2] == 3:
            plotting.save_image_plot(
                cur_raw_image,
                get_loadsave_path(output_dir, cur_image_name)
                + "/raw_image.png",
                cmap=None,
                vmin=0,
                vmax=255,
            )
        else:
            plotting.save_image_plot(
                cur_raw_image,
                get_loadsave_path(output_dir, cur_image_name)
                / Path("raw_image.png"),
                cmap=cm.gray,
                vmin=0,
                vmax=255,
            )
        if eval_params.save_params.pngimages is True:
            if eval_params.save_params.crop_map is True:
                plotting.save_image_plot_crop(
                    cur_raw_image,
                    get_loadsave_path(output_dir, cur_image_name)
                    + "/raw_image_crop.png",
                    cmap=cm.gray,
                    crop_bounds=eval_params.save_params.crop_bounds,
                    vmin=0,
                    vmax=255,
                )
    if eval_params.save_params.raw_labels is True:
        if eval_params.is_evaluate:
            save_dataset_extra(
                output_dir, cur_image_name, "raw_labels", "uint8", cur_labels
            )
            if eval_params.save_params.pngimages is True:
                plotting.save_image_plot(
                    np.argmax(cur_labels, axis=2),
                    get_loadsave_path(output_dir, cur_image_name) / Path("comb_raw_label.png"),
                    cmap=plotting.colors.ListedColormap(
                        plotting.region_colours, N=len(area_maps)
                    ),
                )
    if eval_params.save_params.raw_segs is True:
        save_dataset_extra(output_dir, cur_image_name, "raw_segs", "uint16", cur_seg)
    if eval_params.save_params.aug_image is True:
        save_dataset_extra(
            output_dir, cur_image_name, "augment_image", "uint8", augment_image
        )
        if eval_params.save_params.pngimages is True:
            plotting.save_image_plot(
                augment_image,
                get_loadsave_path(output_dir, cur_image_name)
                + "/augment_image.png",
                cmap=cm.gray,
                vmin=0,
                vmax=255,
            )
            if eval_params.save_params.crop_map is True:
                plotting.save_image_plot_crop(
                    augment_image,
                    get_loadsave_path(output_dir, cur_image_name)
                    + "/augment_image_crop.png",
                    cmap=cm.gray,
                    crop_bounds=eval_params.save_params.crop_bounds,
                    vmin=0,
                    vmax=255,
                )
    if eval_params.save_params.aug_labels is True:
        save_dataset_extra(
            output_dir, cur_image_name, "augment_labels", "uint8", augment_labels
        )
    if eval_params.save_params.aug_segs is True:
        save_dataset_extra(
            output_dir, cur_image_name, "augment_segs", "uint16", augment_segs
        )
    if eval_params.save_params.boundary_names is True:
        save_dataset_extra(
            output_dir,
            cur_image_name,
            "boundary_names",
            "S100",
            imdb.get_boundary_names(),
        )
    if eval_params.save_params.area_names is True:
        save_dataset_extra(
            output_dir, cur_image_name, "area_names", "S100", imdb.get_area_names()
        )
    if eval_params.save_params.patch_class_names is True:
        save_dataset_extra(
            output_dir,
            cur_image_name,
            "patch_class_names",
            "S100",
            imdb.get_patch_class_names(),
        )
    if eval_params.save_params.fullsize_class_names is True:
        save_dataset_extra(
            output_dir,
            cur_image_name,
            "fullsize_class_names",
            "S100",
            imdb.get_fullsize_class_names(),
        )
    if eval_params.save_params.errors is True:
        if dices is not None:
            save_dataset(
                output_dir, cur_image_name, "dices", "float64", np.squeeze(dices)
            )
    if eval_params.save_params.attributes is True:
        save_intermediate_attributes_semantic(
            eval_params,
            imdb,
            cur_image_name,
            predict_time,
            augment_time,
            gen_time,
            convert_time,
            output_dir,
        )
    if eval_params.save_params.activations is True:
        for i in range(len(activations)):
            layer_name = layer_outputs[i].name.split("/")[0]

            if eval_params.save_params.pngimages is True:
                dir_name = (
                    get_loadsave_path(output_dir, cur_image_name)
                    + "/activations/"
                    + layer_name
                )
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)

                for j in range(activations[i].shape[3]):
                    plotting.save_image_plot(
                        activations[i][0, :, :, j],
                        filename=dir_name + "/" + str(j),
                        cmap=cm.viridis,
                    )

            bin_act = np.array(activations[i][0, :, :, :])
            bin_act[bin_act != 0] = 1

            save_dataset_extra(
                output_dir,
                cur_image_name,
                "activations/" + layer_name,
                "float32",
                activations[i][0, :, :, :],
            )
            save_dataset_extra(
                output_dir,
                cur_image_name,
                "activations_bin/" + layer_name,
                "float32",
                bin_act,
            )


def delete_loadsaveextra_file(output_dir, name):
    loadsaveextra_path = get_loadsave_path(output_dir, name)

    if os.path.isfile(get_loadsaveextra_filename(loadsaveextra_path)):
        os.remove(get_loadsaveextra_filename(loadsaveextra_path))


def save_dataset(output_dir, name, dset_name, datatype, dataset):
    loadsave_file = open_append_loadsave_file(output_dir, name)

    dataset = np.array(dataset, dtype=datatype)
    dset = loadsave_file.require_dataset(dset_name, dataset.shape, dtype=datatype)
    dset[:] = dataset

    loadsave_file.close()


def save_dataset_extra(output_dir, name, dset_name, datatype, dataset):
    loadsave_file = open_append_loadsaveextra_file(output_dir, name)

    dataset = np.array(dataset, dtype=datatype)
    dset = loadsave_file.require_dataset(dset_name, dataset.shape, dtype=datatype)
    dset[:] = dataset

    loadsave_file.close()


def load_dataset(output_dir, name, dset_name):
    loadsave_file = open_read_loadsave_file(output_dir, name)

    dset = loadsave_file[dset_name][:]

    loadsave_file.close()

    return dset


def load_dataset_extra(output_dir, name, dset_name):
    loadsaveextra_file = open_read_loadsaveextra_file(output_dir, name)

    dset = loadsaveextra_file[dset_name][:]

    loadsaveextra_file.close()

    return dset


def save_intermediate_attributes_semantic(
    eval_params, imdb, name, predict_time, augment_time, generator_time, convert_time, output_dir
):
    if eval_params.verbosity >= 2:
        print("Saving intermediate attributes...")

    loadsave_file = open_append_loadsave_file(output_dir, name)

    loadsave_file.attrs["aug_desc"] = np.array(eval_params.aug_desc, dtype="S100")
    loadsave_file.attrs["model_filename"] = np.array(
        eval_params.model_file_path.name, dtype="S100"
    )
    loadsave_file.attrs["network_foldername"] = np.array(
        eval_params.model_file_path.parent, dtype="S100"
    )
    loadsave_file.attrs["data_filename"] = np.array(imdb.filename, dtype="S100")
    loadsave_file.attrs["predict_time"] = np.array(predict_time)
    loadsave_file.attrs["augment_time"] = np.array(augment_time)
    loadsave_file.attrs["generator_time"] = np.array(generator_time)
    loadsave_file.attrs["intermediate_timestamp"] = np.array(
        utils.get_timestamp(), dtype="S100"
    )
    if eval_params.save_params.boundary_maps is True:
        loadsave_file.attrs["complete_predict"] = True
    loadsave_file.attrs["convert_time"] = convert_time

    loadsave_file.close()


def save_final_attributes(eval_params, name, graph_time, output_dir):
    if eval_params.verbosity >= 2:
        print("Saving final attributes...")

    loadsave_file = open_append_loadsave_file(output_dir, name)

    loadsave_file.attrs["final_timestamp"] = np.array(
        utils.get_timestamp(), dtype="S100"
    )

    if eval_params.boundaries is True:
        loadsave_file.attrs["graph_time"] = np.array(graph_time)
        loadsave_file.attrs["error_col_bounds"] = np.array(
            (eval_params.col_error_range[0], eval_params.col_error_range[-1])
        )
        loadsave_file.attrs["gsgrad"] = np.array(eval_params.gsgrad)
        loadsave_file.attrs["complete_graph"] = True

    loadsave_file.close()


def open_append_loadsave_file(save_foldername, name):
    loadsave_path = get_loadsave_path(save_foldername, name)

    if not os.path.exists(loadsave_path):
        os.makedirs(loadsave_path)

    loadsave_file = h5py.File(get_loadsave_filename(loadsave_path), "a")

    return loadsave_file


def open_append_loadsaveextra_file(save_foldername, name):
    loadsaveextra_path = get_loadsave_path(save_foldername, name)

    if not os.path.exists(loadsaveextra_path):
        os.makedirs(loadsaveextra_path)

    loadsaveextra_file = h5py.File(get_loadsaveextra_filename(loadsaveextra_path), "a")

    return loadsaveextra_file


def open_read_loadsave_file(save_foldername, name):
    loadsave_path = get_loadsave_path(save_foldername, name)
    loadsave_file = h5py.File(get_loadsave_filename(loadsave_path), "r")

    return loadsave_file


def open_read_loadsaveextra_file(save_foldername, name):
    loadsaveextra_path = get_loadsave_path(save_foldername, name)
    loadsaveextra_file = h5py.File(get_loadsaveextra_filename(loadsaveextra_path), "r")

    return loadsaveextra_file


def get_loadsave_path(save_foldername, name):
    return save_foldername / Path(name).stem


def get_loadsave_filename(loadsave_path):
    return loadsave_path / Path("evaluations.hdf5")


def get_loadsaveextra_filename(loadsave_path):
    return loadsave_path / Path("evaluations_extra.hdf5")


def convert_maps_uint8(prob_maps):
    prob_maps *= 255
    prob_maps = prob_maps.astype("uint8")

    return prob_maps


def soft_dice_numpy(y_pred, y_true, eps=1e-7):
    """
    c is number of classes
    :param y_pred: b x c x X x Y( x Z...) network output, must sum to 1 over c channel (such as after softmax)
    :param y_true: b x c x X x Y( x Z...) one hot encoding of ground truth
    :param eps:
    :return:
    """

    axes = tuple(range(2, len(y_pred.shape)))
    intersect = np.sum(y_pred * y_true, axis=axes)
    denom = np.sum(y_pred + y_true, axis=axes)

    for i in range(intersect.shape[1]):
        # if there is no region for a class to predict, there shouldn't be a penalty for correctly predicting empty
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

    overall_dice = np.array([np.array([2.0 * intersect_total / (denom_total + eps)])])

    return np.concatenate((class_dices, overall_dice), axis=1)


def soft_dice_numpy_multiple(y_pred, y_true, eps=1e-7):
    """
    c is number of classes
    :param y_pred: b x c x X x Y( x Z...) network output, must sum to 1 over c channel (such as after softmax)
    :param y_true: b x c x X x Y( x Z...) one hot encoding of ground truth
    :param eps:
    :return:
    """

    axes = tuple(range(2, len(y_pred.shape)))
    intersect = np.sum(y_pred * y_true, axis=axes)
    denom = np.sum(y_pred + y_true, axis=axes)

    for j in range(intersect.shape[0]):
        for i in range(intersect.shape[1]):
            # if there is no region for a class to predict, there shouldn't be a penalty for correctly predicting empty
            if intersect[j, i] == 0 and denom[j, i] == 0:
                # set to 1
                # intersect[0, i] = 0.5
                # denom[0, i] = 1 - eps

                # OR

                # set to NaN
                intersect[j, i] = np.nan

    class_dices = 2.0 * intersect / (denom + eps)

    return class_dices


def calc_dice(eval_params, predictions, labels):
    if eval_params is not None:
        low = eval_params.col_error_range[0]
        high = eval_params.col_error_range[-1] + 1
    else:
        low = 0
        high = predictions.shape[2]

    if K.image_data_format() == "channels_last":
        dices = soft_dice_numpy(
            predictions[:, :, low:high, :],
            np.transpose(labels, axes=(0, 3, 1, 2))[:, :, low:high, :],
        )
    else:
        dices = soft_dice_numpy(
            predictions[:, :, low:high, :], labels[:, :, low:high, :]
        )

    return dices


def calc_areas(y_pred, y_true):
    areas = []
    area_diffs = []
    area_abs_diffs = []
    for i in range(y_pred.shape[0]):
        pred_area_size = np.count_nonzero(y_pred[i, :, :, 1])
        true_area_size = np.count_nonzero(y_true[i, :, :, 1])

        areas.append([pred_area_size, true_area_size])
        area_diffs.append(pred_area_size - true_area_size)
        area_abs_diffs.append(np.abs(pred_area_size - true_area_size))

    return [areas, area_diffs, area_abs_diffs]


def save_boundaries_to_csv(boundaries, ouput_path):
    np.savetxt(ouput_path, boundaries, delimiter=",", fmt="%d")

