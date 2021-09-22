import augmentation as aug
import custom_losses
import custom_metrics
import dataset_construction
import dataset_loader
import eval_config as cfg
import eval_helper
import image_database as imdb
import os
import save_parameters

from keras.models import load_model
from keras.utils import to_categorical
import h5py

if hasattr(cfg, "DATASET_NAME"):
    dataset_name = cfg.DATASET_NAME
else:
    dataset_name = os.path.basename(cfg.TEST_DATASET_FILE)

test_hdf5_file = h5py.File(cfg.TEST_DATASET_FILE, 'r')

test_images, test_segs, test_image_names = dataset_loader.load_testing_data(test_hdf5_file)

test_labels = dataset_construction.create_all_area_masks(test_images, test_segs)
test_labels = to_categorical(test_labels, cfg.NUM_CLASSES)

# boundary names should be a list of strings with length = NUM_CLASSES - 1
# class names should be a list of strings with length = NUM_CLASSES
AREA_NAMES = ["area_" + str(i) for i in range(cfg.NUM_CLASSES)]
BOUNDARY_NAMES = ["boundary_" + str(i) for i in range(cfg.NUM_CLASSES - 1)]
BATCH_SIZE = 1  # DO NOT MODIFY
GSGRAD = 1
CUSTOM_OBJECTS = dict(list(custom_losses.custom_loss_objects.items()) +
                      list(custom_metrics.custom_metric_objects.items()))

eval_imdb = imdb.ImageDatabase(images=test_images, labels=test_labels, segs=test_segs, image_names=test_image_names,
                               boundary_names=BOUNDARY_NAMES, area_names=AREA_NAMES,
                               fullsize_class_names=AREA_NAMES, num_classes=cfg.NUM_CLASSES, name=dataset_name, filename=cfg.TEST_DATASET_FILE, mode_type='fullsize')

model_file = cfg.MODEL_FILE

loaded_model = load_model(model_file, custom_objects=CUSTOM_OBJECTS)

aug_fn_arg = (aug.no_aug, {})

eval_helper.evaluate_network(eval_imdb, os.path.basename(model_file), os.path.dirname(model_file),
                             BATCH_SIZE, save_parameters.SaveParameters(pngimages=True, raw_image=True, raw_labels=True, temp_extra=True, boundary_maps=True, area_maps=True, comb_area_maps=True, seg_plot=True),
                             gsgrad=GSGRAD, aug_fn_arg=aug_fn_arg, eval_mode='both', boundaries=True, boundary_errors=True, dice_errors=True, col_error_range=None, normalise_input=True, transpose=False)

