import datetime
import h5py
import numpy as np
from pathlib import Path
from typeguard import typechecked

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical

from unet.model import custom_losses
from unet.model import custom_metrics


def get_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H_%M_%S")

    return timestamp


def load_model(model_path: Path):
    custom_objects = dict(
        list(custom_losses.custom_loss_objects.items())
        + list(custom_metrics.custom_metric_objects.items())
    )

    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def convert_maps_uint8(prob_maps):
    prob_maps *= 255
    prob_maps = prob_maps.astype("uint8")

    return prob_maps


def perform_argmax(predictions, bin=True):
    """
        Arguments:
            bin: If True 'categorical_pred' will contain 1 or 0s corresponding to the pixel
            belonging to a particular class or not. If 'False', 'categorical_pred' will contain the
            prediction probabilites for each pixel per class.

        Returns:
            argmax_pred: A matrix of shape (1, image_width, image_height) that contains the
            predicted classes numbered from 0 to num_classes - 1.
            categorical_pred: A matrix of shape (num_classes, image_width, image_height) that
            contains:
            - If 'bin' == True: 1 on pixels that belong the class and 0 otherwise.
            - If 'bin' == False: Prediction probabilities for each pixel per class.
    """
    if K.image_data_format() == "channels_last":
        pass
    else:
        predictions = np.transpose(predictions, (0, 2, 3, 1))

    num_maps = predictions.shape[3]

    if bin:
        argmax_pred = np.argmax(predictions, axis=3) #TODO: Refactor line

        categorical_pred = to_categorical(argmax_pred, num_maps)
        categorical_pred = np.transpose(categorical_pred, axes=(0, 3, 1, 2))
    else:
        argmax_pred = np.argmax(predictions, axis=3)
        categorical_pred = np.transpose(predictions, axes=(0, 3, 1, 2))

    return [argmax_pred, categorical_pred]


def convert_predictions_to_maps_semantic(categorical_pred, bg_ilm=True, bg_csi=False):
    """
    #TODO: Document functionality
    """
    num_samples = categorical_pred.shape[0]
    img_width = categorical_pred.shape[2]
    img_height = categorical_pred.shape[3]
    num_maps = categorical_pred.shape[1]

    boundary_maps = np.zeros(
        (num_samples, num_maps - 1, img_width, img_height), dtype="uint8"
    )

    for sample_ind in range(num_samples):
        for map_ind in range(1, num_maps):  # don't care about boundary for top region

            if (map_ind == 1 and bg_ilm is True) or (
                map_ind == num_maps - 1 and bg_csi is True
            ):
                cur_map = categorical_pred[sample_ind, map_ind - 1, :, :]

                grad_map = np.gradient(cur_map, axis=1)

                grad_map = -grad_map

                grad_map[grad_map < 0] = 0

                grad_map *= 2  # scale map to between 0 and 1

                rolled_grad = np.roll(grad_map, -1, axis=1)

                grad_map -= rolled_grad
                grad_map[grad_map < 0] = 0
                boundary_maps[sample_ind, map_ind - 1, :, :] = convert_maps_uint8(
                    grad_map
                )
            else:
                cur_map = categorical_pred[sample_ind, map_ind, :, :]

                grad_map = np.gradient(cur_map, axis=1)

                grad_map[grad_map < 0] = 0

                grad_map *= 2  # scale map to between 0 and 1

                rolled_grad = np.roll(grad_map, -1, axis=1)

                grad_map -= rolled_grad
                grad_map[grad_map < 0] = 0
                boundary_maps[sample_ind, map_ind - 1, :, :] = convert_maps_uint8(
                    grad_map
                )

    return boundary_maps