import os
import sys

import getpass
import h5py
import logging as log
import mlflow
from mlflow.exceptions import MlflowException
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from unet.common import utils
from unet.common.mlflow_parameters import MLflowParameters
from unet.model import image_database as imdb
from unet.model import data_generator as data_gen
from unet.model import dataset_loader
from unet.model import training_callbacks
from unet.model import training_parameters as tparams
from unet.model import unet

mlflow.autolog()


def save_config_file(
    save_foldername: Path,
    model_name,
    timestamp,
    train_params,
    train_imdb,
    val_imdb,
    opt,
):
    config_filename = save_foldername / Path("config.hdf5")

    config_file = h5py.File(config_filename, "w")

    config_file.attrs["timestamp"] = np.array(timestamp, dtype="S100")
    config_file.attrs["model_name"] = np.array(model_name, dtype="S1000")
    config_file.attrs["train_imdb"] = np.array(
        train_imdb.filename, dtype="S100"
    )
    config_file.attrs["val_imdb"] = np.array(val_imdb.filename, dtype="S100")
    config_file.attrs["epochs"] = train_params.epochs
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

    config_file.attrs["metric"] = np.array(
        train_params.metric_name, dtype="S100"
    )
    config_file.attrs["loss"] = np.array(train_params.loss_name, dtype="S100")
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


def train_network(
    train_imdb, val_imdb, train_params, input_channels, num_classes
):
    strategy = tf.distribute.MirroredStrategy()
    log.info(f"Number of devices: {strategy.num_replicas_in_sync}")

    optimizer_con = train_params.opt_con
    optimizer_params = train_params.opt_params

    optimizer = optimizer_con(**optimizer_params)

    loss = train_params.loss
    metric = train_params.metric
    epochs = train_params.epochs
    initial_model_path = train_params.initial_model
    early_stopping = train_params.early_stopping

    if initial_model_path:
        log.info(f"Starting training from model: {initial_model_path}")
        # TODO: Fix, 'descs' should be read from some metadata field or
        # separate config
        model, model_name, model_name_short = [
            utils.load_model(initial_model_path),
            "U-net",
            "U-net",
        ]
    else:
        log.info("Starting training from scratch oct-unet model")
        with strategy.scope():
            model, model_name, model_name_short = unet.get_standard_model(
                input_channels=input_channels,
                num_classes=num_classes,
            )
            model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    batch_size = train_params.batch_size
    aug_fn_args = train_params.aug_fn_args
    aug_mode = train_params.aug_mode
    aug_probs = train_params.aug_probs
    aug_fly = train_params.aug_fly
    aug_val = train_params.aug_val
    use_gen = train_params.use_gen
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

    shuffle = train_params.shuffle
    normalise = train_params.normalise
    monitor = train_params.model_save_monitor

    save_best = train_params.model_save_best

    dataset_name = train_imdb.name

    timestamp = utils.get_timestamp()

    results_location = train_params.results_location
    save_foldername = results_location / Path(
        timestamp + "_" + model_name_short + "_" + dataset_name
    )

    if not os.path.exists(save_foldername):
        os.makedirs(save_foldername)
    else:
        count = 2
        testsave_foldername = results_location / Path(
            timestamp
            + "_"
            + str(count)
            + "_"
            + model_name_short
            + "_"
            + dataset_name
        )
        while os.path.exists(testsave_foldername):
            count += 1
            testsave_foldername = results_location / Path(
                timestamp
                + "_"
                + str(count)
                + "_"
                + model_name_short
                + "_"
                + dataset_name
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
        model_name=model_name,
        train_params=train_params,
        train_imdb=train_imdb,
    )

    callbacks_list = [savemodel, history]

    if early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef", mode="max", patience=5
        )
        callbacks_list.append(early_stopping_callback)

    save_config_file(
        save_foldername,
        model_name,
        timestamp,
        train_params,
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
        class_weight=train_params.class_weight,
    )


def train_model(
    training_params: tparams.TrainingParams,
    mlflow_params: MLflowParameters = None,
):
    if mlflow_params:
        if mlflow_params.username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_params.username
        else:
            # Only set env variable if not already set
            if not os.getenv("MLFLOW_TRACKING_USERNAME"):
                os.environ["MLFLOW_TRACKING_USERNAME"] = getpass.getuser()
        if mlflow_params.password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_params.password
        else:
            if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
                log.error(
                    "MLFlow password not provided and "
                    "MLFLOW_TRACKING_PASSWORD environment not set. Exiting..."
                )
                exit(1)

        mlflow.set_tracking_uri(mlflow_params.tracking_uri)
        try:
            mlflow.set_experiment(mlflow_params.experiment)
        except MlflowException as e:
            log.exception(
                msg=f"An error occurred while setting MLflow "
                f"experiment. Exception message: {e.message}"
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
    training_dataset_name = training_params.training_dataset_name
    train_labels = to_categorical(train_labels, num_classes)
    val_labels = to_categorical(val_labels, num_classes)

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
        name=training_params.training_dataset_name,
        filename=training_dataset_path,
        mode_type="fullsize",
        num_classes=num_classes,
    )

    train_network(
        train_imdb,
        val_imdb,
        training_params,
        input_channels,
        num_classes,
    )
