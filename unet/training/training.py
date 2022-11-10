import os
import sys

import h5py
import logging as log
import mlflow
from mlflow.exceptions import MlflowException
import numpy as np
from pathlib import Path
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from typeguard import typechecked
from typing import Union

from unet.common import data_generator as data_gen, dataset_loader, utils
from unet.common.mlflow_parameters import MLflowParameters
from unet.model import (
    custom_losses,
    custom_metrics,
    image_database as imdb,
    unet,
)
from unet.training import training_callbacks, training_parameters as tparams


@typechecked
def save_training_params_file(
    save_foldername: Path,
    model_summary: str,
    training_dataset_md5: str,
    class_weight: Union[np.ndarray, None],
    timestamp,
    train_params,
    train_imdb,
    val_imdb,
    opt,
):
    config_filename = save_foldername / Path("training_params.hdf5")

    config_file = h5py.File(config_filename, "w")

    config_file.attrs["timestamp"] = np.array(timestamp, dtype="S100")
    config_file.attrs["model_summary"] = np.array(model_summary, dtype="S1000")
    config_file.attrs["train_dataset_md5"] = np.array(
        training_dataset_md5, dtype="S1000"
    )
    config_file.attrs["train_imdb"] = np.array(
        train_imdb.filename, dtype="S100"
    )
    config_file.attrs["val_imdb"] = np.array(val_imdb.filename, dtype="S100")
    config_file.attrs["epochs"] = train_params.epochs
    config_file.attrs["loss_name"] = np.array(train_params.loss, dtype="S1000")
    config_file.attrs["metric_name"] = np.array(
        train_params.metric, dtype="S1000"
    )

    if class_weight is None:
        config_file.attrs["class_weight"] = np.array("None", dtype="S1000")
    else:
        config_file.attrs["class_weight"] = np.array("array", dtype="S1000")
        config_file["class_weight"] = class_weight

    config_file.attrs["dim_names"] = np.array(
        train_imdb.dim_names, dtype="S100"
    )
    config_file.attrs["type"] = np.array(train_imdb.type, dtype="S100")
    if train_imdb.type == "patch":
        config_file.attrs["patch_size"] = np.array(
            (train_imdb.image_width, train_imdb.image_height)
        )

    if train_imdb.dim_inds is None:
        config_file.attrs["train_dim_inds"] = np.array("all", dtype="S100")
    else:
        dim_count = 0
        for dim_ind in train_imdb.dim_inds:
            if dim_ind is not None:
                config_file.attrs[
                    "train_dim_ind: " + train_imdb.dim_names[dim_count]
                ] = dim_ind
            dim_count += 1

    if val_imdb.dim_inds is None:
        config_file.attrs["val_dim_inds"] = np.array("all", dtype="S100")
    else:
        dim_count = 0
        for dim_ind in val_imdb.dim_inds:
            if dim_ind is not None:
                config_file.attrs[
                    "val_dim_ind: " + val_imdb.dim_names[dim_count]
                ] = dim_ind
            dim_count += 1

    config_file.attrs["metric"] = np.array(train_params.metric, dtype="S100")
    config_file.attrs["loss"] = np.array(train_params.loss, dtype="S100")
    config_file.attrs["batch_size"] = train_params.batch_size
    config_file.attrs["shuffle"] = train_params.shuffle
    if train_imdb.padding is not None:
        config_file.attrs["padding"] = np.array(train_imdb.padding)

    config_file.attrs["aug_mode"] = np.array(
        train_params.aug_mode, dtype="S100"
    )
    if train_params.aug_mode != "none":
        for aug_ind in range(len(train_params.aug_fn_args)):
            aug_fn = train_params.aug_fn_args[aug_ind][0]
            aug_arg = train_params.aug_fn_args[aug_ind][1]

            aug_desc = aug_fn(None, None, None, aug_arg, True)

            if type(aug_arg) is not dict:
                config_file.attrs["aug_" + str(aug_ind + 1)] = np.array(
                    aug_desc, dtype="S1000"
                )
            else:
                config_file.attrs["aug_" + str(aug_ind + 1)] = np.array(
                    aug_fn.__name__, dtype="S100"
                )

                for aug_param_key in aug_arg.keys():
                    val = aug_arg[aug_param_key]
                    if type(val) is int or type(val) is float:
                        config_file.attrs[
                            "aug_"
                            + str(aug_ind + 1)
                            + "_param: "
                            + aug_param_key
                        ] = np.array(val)
                    elif type(val) is str:
                        config_file.attrs[
                            "aug_"
                            + str(aug_ind + 1)
                            + "_param: "
                            + aug_param_key
                        ] = np.array(val, dtype="S100")
                    elif type(val) is list and (
                        type(val[0]) is int
                        or type(val[0]) is str
                        or type(val[0]) is float
                    ):
                        config_file.attrs[
                            "aug_"
                            + str(aug_ind + 1)
                            + "_param: "
                            + aug_param_key
                        ] = np.array(str(val), dtype="S100")

            if train_params.aug_mode == "one":
                config_file.attrs["aug_probs"] = np.array(
                    train_params.aug_probs
                )

        config_file.attrs["aug_fly"] = train_params.aug_fly
        config_file.attrs["aug_val"] = train_params.aug_val

    config_file.attrs["optimizer"] = np.array(
        train_params.opt_con.__name__, dtype="S100"
    )

    opt_config = opt.get_config()

    for key in opt_config:
        if type(opt_config[key]) is dict:
            config_file.attrs["opt_param: " + key] = np.string_(
                str(opt_config[key])
            )
        else:
            config_file.attrs["opt_param: " + key] = opt_config[key]

    config_file.attrs["normalise"] = np.array(train_params.normalise)

    config_file.attrs["ram_load"] = train_imdb.ram_load


def train_model(
    training_params: tparams.TrainingParams,
    mlflow_params: MLflowParameters = None,
):
    if mlflow_params:
        mlflow.keras.autolog(save_format="h5")
        if mlflow_params.username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_params.username

        if mlflow_params.password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_params.password

        mlflow.set_tracking_uri(mlflow_params.tracking_uri)
        try:
            mlflow.set_experiment(mlflow_params.experiment)
            mlflow_run = mlflow.start_run()
            log.info(f"MLFlow Run ID: {mlflow_run.info.run_id}")
        except MlflowException as exc:
            if exc.get_http_status_code() == 401:
                log.error(
                    "Looks like the MLFLow client is not authorized to log "
                    "into the MLFlow server. Make sure the environment "
                    "variables 'MLFLOW_TRACKING_USERNAME' and "
                    "'MLFLOW_TRACKING_PASSWORD' are correct"
                )
            log.exception(
                msg="An error occurred while setting MLflow experiment"
            )
            sys.exit(1)

    if training_params.channels_last:
        tf.keras.backend.set_image_data_format("channels_last")

    training_dataset_path = training_params.training_dataset_path
    training_hdf5_file = h5py.File(training_dataset_path, "r")

    # images numpy array should be of the shape: (number of images, image
    # width, image height, 1) segments numpy array should be of the shape:
    # (number of images, number of boundaries, image width)
    train_images, train_labels = dataset_loader.load_training_data(
        training_hdf5_file
    )
    val_images, val_labels = dataset_loader.load_validation_data(
        training_hdf5_file
    )

    num_classes = len(np.unique(train_labels))
    log.info(f"Detected {num_classes} classes")
    input_channels = train_images.shape[-1]
    log.info(f"Detected {input_channels} input channels")
    training_dataset_name = training_params.training_dataset_path.stem

    strategy = tf.distribute.MirroredStrategy()
    log.info(f"Number of devices: {strategy.num_replicas_in_sync}")

    optimizer_con = training_params.opt_con
    optimizer_params = training_params.opt_params

    optimizer = optimizer_con(**optimizer_params)

    loss = custom_losses.custom_loss_objects.get(training_params.loss)
    if loss is None:
        log.error(f"Loss '{training_params.loss}' not found. Exiting...")
        exit(1)
    else:
        if training_params.class_weight == "balanced":
            dataset_labels = np.concatenate((train_labels, val_labels))
            c_weight = class_weight.compute_class_weight(
                "balanced",
                classes=np.unique(dataset_labels),
                y=dataset_labels.flatten(),
            )
        elif type(training_params.class_weight) == list:
            c_weight = np.array(training_params.class_weight)
        else:
            c_weight = None
        loss_fn = loss["function"](class_weight=c_weight)
        sparse_labels = loss["takes_sparse"]

    metric = custom_metrics.custom_metric_objects.get(training_params.metric)
    if metric is None:
        log.error(f"Metric '{training_params.metric}' not found. Exiting...")
        exit(1)
    else:
        metric_fn = metric(sparse_labels, num_classes)

    if not sparse_labels:
        train_labels = to_categorical(train_labels, num_classes)
        val_labels = to_categorical(val_labels, num_classes)

    training_dataset_md5 = utils.md5(training_dataset_path)
    mlflow.log_params(
        {
            "training_dataset_path": training_dataset_path,
            "training_dataset_md5": training_dataset_md5,
            "augmentation_mode": training_params.aug_mode,
            "loss_name": training_params.loss,
            "metric_name": training_params.metric,
            "loss_fn_class_weight": training_params.class_weight,
            "class_weight_array": c_weight,
        }
    )

    train_imdb = imdb.ImageDatabase(
        images=train_images,
        labels=train_labels,
        name=training_dataset_name,
        filename=training_dataset_path,
        mode_type="fullsize",
        num_classes=num_classes,
    )

    val_imdb = imdb.ImageDatabase(
        images=val_images,
        labels=val_labels,
        name=training_dataset_name,
        filename=training_dataset_path,
        mode_type="fullsize",
        num_classes=num_classes,
    )

    epochs = training_params.epochs
    initial_model_path = training_params.initial_model
    early_stopping = training_params.early_stopping

    if initial_model_path:
        log.info(f"Starting training from model: {initial_model_path}")
        model = utils.load_model(initial_model_path)
    else:
        log.info("Starting training from scratch U-net model")
        with strategy.scope():
            model = unet.unet(
                8,
                4,
                2,
                (3, 3),
                (2, 2),
                input_channels=input_channels,
                output_channels=num_classes,
            )

            model.compile(
                optimizer=optimizer, loss=loss_fn, metrics=[metric_fn]
            )

    batch_size = training_params.batch_size
    aug_fn_args = training_params.aug_fn_args
    aug_mode = training_params.aug_mode
    aug_probs = training_params.aug_probs
    aug_fly = training_params.aug_fly
    aug_val = training_params.aug_val
    use_gen = training_params.use_gen
    patience = training_params.patience
    restore_best_weights = training_params.restore_best_weights
    ram_load = train_imdb.ram_load

    if use_gen is False and ram_load == 0:
        print("Incompatible parameter selection")
        exit(1)
    elif ram_load == 0 and aug_fly is False and aug_mode != "none":
        print("Incompatible parameter selection")
        exit(1)

    if aug_val is False:
        aug_val_mode = "none"
        aug_val_fn_args = []
        aug_val_probs = []
        aug_val_fly = False
    else:
        aug_val_mode = aug_mode
        aug_val_fn_args = aug_fn_args
        aug_val_probs = aug_probs
        aug_val_fly = aug_fly

    shuffle = training_params.shuffle
    normalise = training_params.normalise
    monitor = training_params.model_save_monitor

    save_best = training_params.model_save_best

    dataset_name = train_imdb.name

    timestamp = utils.get_timestamp()

    results_location = training_params.results_location
    save_foldername = (
        results_location
        / Path(mlflow_run.info.run_id)
        / Path(timestamp + "_U-net_" + dataset_name)
    )

    if not os.path.exists(save_foldername):
        os.makedirs(save_foldername)
    else:
        count = 2
        testsave_foldername = results_location / Path(
            timestamp + "_" + str(count) + "_U-net_" + dataset_name
        )
        while os.path.exists(testsave_foldername):
            count += 1
            testsave_foldername = results_location / Path(
                timestamp + "_" + str(count) + "_U-net_" + dataset_name
            )

        save_foldername = testsave_foldername
        os.makedirs(save_foldername)

    epoch_model_name = "model_epoch{epoch:02d}.hdf5"

    savemodel = ModelCheckpoint(
        filepath=save_foldername / Path(epoch_model_name),
        save_best_only=save_best,
        monitor=monitor[0],
        mode=monitor[1],
    )

    history = training_callbacks.SaveEpochInfo(
        save_folder=save_foldername,
        train_params=training_params,
    )

    callbacks_list = [savemodel, history]

    if early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=patience,
            restore_best_weights=restore_best_weights,
        )
        callbacks_list.append(early_stopping_callback)

    model_summary = []
    model.summary(print_fn=lambda line: model_summary.append(line))
    save_training_params_file(
        save_foldername,
        "\n".join(model_summary),
        training_dataset_md5,
        c_weight,
        timestamp,
        training_params,
        train_imdb,
        val_imdb,
        optimizer,
    )

    train_gen = data_gen.DataGenerator(
        train_imdb,
        batch_size,
        aug_fn_args,
        aug_mode,
        aug_probs,
        aug_fly,
        shuffle,
        normalise=normalise,
        ram_load=ram_load,
    )

    val_gen = data_gen.DataGenerator(
        val_imdb,
        batch_size,
        aug_val_fn_args,
        aug_val_mode,
        aug_val_probs,
        aug_val_fly,
        shuffle,
        normalise=normalise,
        ram_load=ram_load,
    )

    train_gen_total_samples = train_gen.get_total_samples()
    if batch_size > train_gen_total_samples:
        log.error(
            f"The batch size ({batch_size}) cannot be larger than the number "
            f"of training samples ({train_gen_total_samples})"
        )
        exit(1)
    log.info(
        f"Train generator total number of samples: {train_gen_total_samples}"
    )

    val_gen_total_samples = val_gen.get_total_samples()
    if batch_size > val_gen_total_samples:
        log.error(
            f"The batch size ({batch_size}) cannot be larger than the number "
            f"of validation samples ({val_gen_total_samples})"
        )
        exit(1)
    log.info(
        "Validation generator total number of samples: "
        f"{val_gen_total_samples}"
    )

    model.summary()
    model.fit(
        x=train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )
    mlflow.end_run()
