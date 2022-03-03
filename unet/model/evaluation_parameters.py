from __future__ import annotations

from pathlib import Path
import numpy as np
import tensorflow as tf

from unet.model import augmentation as aug
from unet.model import common
from unet.model import save_parameters as sparams


class PredictionDataset:
    def __init__(
        self,
        prediction_images: np.array,
        prediction_images_names: list[Path],
        prediction_images_output_dirs: list[Path]
    ):
        self.prediction_images = prediction_images
        self.prediction_images_names = prediction_images_names
        self.prediction_images_output_dirs = prediction_images_output_dirs


class EvaluationParameters:
    """Parameters for evaluation of trained network.
        _________

        model_filename: filename of Keras model to be loaded
        _________

        data_filename: filename of .hdf5 or .h5 file to be used to load images, masks, segs and names.
        Data files should contain 5 datasets.

        'images' with shape: (number of images, width, height) (dtype = 'uint8')
        'mask_labels' with shape: (number of images, width, height) (dtype = 'uint8')
        'patch_labels' with shape: (number of images, width, height) (dtype = 'uint8')
        'segs' with shape: (number of images, number of boundaries, width) (dtype = 'uint16')
        'image_names' with shape: (number of images,) (dtype = 'S' - fixed length strings)
        'boundary_names' with shape: (number of boundaries,) (dtype = 'S' - fixed length strings)
        'area_names' with shape: (number of boundaries + 1,) (dtype = 'S' - fixed length strings)
        _________

        aug_fn_args: 2 tuple includes: function used to augment each image,
         tuple of arguments to be passed to the augmentation function. Default: (None, None), do not use augmentation.
        _________

        graph_structure: graph neighbour structure to be used for evaluations
        _________

        eval_mode

        both: predict using network and construct probability maps AND segment maps with graph search
        network: predict using network and construct probability maps ONLY
        gs: segment maps with graph search ONLY
        _________

        save_filename

        file/folder used to load and save predicted maps, delineations, errors and other information.
        Usage depends on mode and output_type.

        when output_type = 'file':

        when mode is:
        both: will save prob maps, delineations and errors to file
        network: will save probs maps to file
        gs: will load prob maps from file and save delineations and errors to file

        savefiles contain a number of datasets for each image and information associated with each:

        'predictions' with shape (number of boundaries, width) (dtype = 'uint16')
        'errors' with shape (number of boundaries, width) (dtype = 'int16')     (SIGNED INTEGER FOR NEGATIVE VALUES)
        'boundary_maps' with shape (number of boundaries, width, height)  (dtype = 'uint8')
        'area_maps' with shape (number of boundaries + 1, width, height) (dtype = 'uint8') (semantic segmentation ONLY)
        'datafile_name' string with name of the datafile used for evaluation
        'model_name' string with name of the model used for evaluation
        'augmentation' string with a description of the augmentation function used for evaluation
        'patch_size' string with the size of patches used for evaluation
        'error_range' string with the column range used for error calculations
        _________

        patch_size: size of patches to use with shape: (patch width, patch height)
        _________

        col_error_range: range of columns to calculate errors. Default: None calculates error for all columns.
        _________

        save_params: parameters using to determine what to save from the evaluation
        _________

        verbosity: level of verbosity to use when printing output to the console

        0: no additional output
        1: network evaluation progress
        2  network evaluation and output progress
        3: network evaluation and output progress and results
        _________

        binar_boundary_maps: whether to binarize to boundary probability maps or not
        _________

        """
    def __init__(
        self,
        model_file_path: Path,
        prediction_dataset: PredictionDataset | Path,
        is_evaluate: bool,
        col_error_range,
        save_foldername: Path,
        eval_mode='both',
        aug_fn_arg=(aug.no_aug, {}),
        patch_size=None,
        save_params=sparams.SaveParameters(),
        transpose=False,
        normalise_input=True,
        verbosity=3,
        gsgrad=1,
        comb_pred=False,
        recalc_errors=False,
        boundaries=True,
        boundary_errors=True,
        trim_maps=False,
        trim_ref_ind=0,
        trim_window=(0, 0),
        collate_results=True,
        dice_errors=True,
        flatten_image=False,
        flatten_ind=0,
        flatten_poly=False,
        binarize=True,
        binarize_after=False,
        bg_ilm=True,
        bg_csi=False,
        flatten_pred_edges=False,
        flat_marg=0,
        use_thresh=False,
        thresh=0.5
    ):
        self.model_file_path = model_file_path
        self.prediction_dataset = prediction_dataset
        self.is_evaluate = is_evaluate
        self.binarize = binarize
        self.binarize_after = binarize_after

        self.eval_mode = eval_mode
        self.aug_fn_arg = aug_fn_arg
        self.patch_size = patch_size
        self.col_error_range = col_error_range
        self.save_params = save_params
        self.transpose = transpose
        self.normalise_input = normalise_input
        self.verbosity = verbosity
        self.comb_pred = comb_pred
        self.recalc_errors = recalc_errors
        self.boundaries = boundaries
        self.boundary_errors = boundary_errors
        self.trim_maps = trim_maps
        self.trim_ref_ind = trim_ref_ind
        self.trim_window = trim_window
        self.collate_results = collate_results
        self.dice_errors = dice_errors
        self.flatten_image = flatten_image
        self.flatten_ind = flatten_ind
        self.flatten_poly = flatten_poly

        self.flatten_pred_edges = flatten_pred_edges
        self.flat_marg = flat_marg

        self.bg_ilm=bg_ilm
        self.bg_csi=bg_csi

        self.use_thresh=use_thresh
        self.thresh=thresh

        self.aug_fn = aug_fn_arg[0]
        self.aug_arg = aug_fn_arg[1]
        self.aug_desc = self.aug_fn(None, None, None, self.aug_arg, desc_only=True)
        self.gsgrad = gsgrad

        self.save_foldername = save_foldername

        self.loaded_model = common.load_model(model_file_path)
        self.num_classes = self.loaded_model.output.shape[-1]

        if self.verbosity >= 1:
            self.predict_verbosity = 1
        else:
            self.predict_verbosity = 0
