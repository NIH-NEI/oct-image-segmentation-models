from keras.utils import to_categorical
import h5py

from unet.model import augmentation as aug
from unet.model import dataset_construction as dc
from unet.model import dataset_loader as dl
from unet.model import eval_config as cfg
from unet.model import eval_helper
from unet.model import image_database as imdb
from unet.model import save_parameters


def evaluate_model_from_hdf5(model_file_path, test_dataset_file):
    test_hdf5_file = h5py.File(test_dataset_file, 'r')

    test_images, test_segs, test_image_names = dl.load_testing_data(
        test_hdf5_file
    )

    evaluate_model(model_file_path, test_images, test_image_names, test_segs)


def evaluate_model(
    model_file_path,
    test_images,
    test_image_names,
    is_evaluate,
    output_path,
    test_segments=None
):

    test_labels = None
    if test_segments:
        test_labels = dc.create_all_area_masks(test_images, test_segments)
        test_labels = to_categorical(test_labels, cfg.NUM_CLASSES)

    AREA_NAMES = ["area_" + str(i) for i in range(cfg.NUM_CLASSES)]
    BOUNDARY_NAMES = ["boundary_" + str(i) for i in range(cfg.NUM_CLASSES - 1)]
    GSGRAD = 1

    eval_imdb = imdb.ImageDatabase(
        images=test_images,
        labels=test_labels,
        segs=test_segments,
        image_names=test_image_names,
        boundary_names=BOUNDARY_NAMES,
        area_names=AREA_NAMES,
        fullsize_class_names=AREA_NAMES,
        num_classes=cfg.NUM_CLASSES,
        filename=cfg.TEST_DATASET_FILE,
        mode_type='fullsize'
    )

    aug_fn_arg = (aug.no_aug, {})

    eval_helper.evaluate_network(
        eval_imdb,
        model_file_path,
        is_evaluate,
        save_parameters.SaveParameters(
            pngimages=False,
            raw_image=True,
            raw_labels=True,
            temp_extra=True,
            boundary_maps=True,
            area_maps=True,
            comb_area_maps=True,
            seg_plot=True
        ),
        save_foldername=output_path.absolute(),
        gsgrad=GSGRAD,
        aug_fn_arg=aug_fn_arg,
        eval_mode='both',
        boundaries=True,
        boundary_errors=True,
        dice_errors=True,
        col_error_range=None,
        normalise_input=True,
        transpose=False
    )
