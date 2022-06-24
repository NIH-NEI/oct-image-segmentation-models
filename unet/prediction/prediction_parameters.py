from pathlib import Path
from typeguard import typechecked

from unet.common import utils
from unet.common.dataset import Dataset


@typechecked
class PredictionSaveParams:
    def __init__(
        self,
        predicted_labels: bool=True,
        categorical_pred: bool=False,
        png_images: bool=True,
        boundary_maps: bool=True,
        individual_raw_boundary_pngs: bool=False,
        individual_seg_plots: bool=False,
    ) -> None:
        self.predicted_labels = predicted_labels
        self.categorical_pred = categorical_pred
        self.png_images = png_images
        self.boundary_maps = boundary_maps
        self.individual_raw_boundary_pngs = individual_raw_boundary_pngs
        self.individual_seg_plots = individual_seg_plots


@typechecked
class PredictionParams:
    def __init__(
        self,
        model_path: Path,
        dataset: Dataset,
        config_output_dir: Path,
        save_params: PredictionSaveParams,
        flatten_image: bool=False,
        flatten_ind: int=0,
        flatten_poly: bool=False,
        flatten_pred_edges: bool=False,
        flat_marg: int=0,
        trim_maps: bool=False,
        trim_ref_ind: int=0,
        trim_window: tuple=(0, 0),
        col_error_range: tuple=None,
    ) -> None:
        self.model_path = model_path
        self.dataset = dataset
        self.loaded_model = utils.load_model(model_path)
        self.num_classes = self.loaded_model.output.shape[-1]
        self.config_output_dir = config_output_dir
        self.save_params = save_params
        self.flatten_image = flatten_image
        self.flatten_ind = flatten_ind
        self.flatten_poly = flatten_poly
        self.flatten_pred_edges = flatten_pred_edges
        self.flat_marg = flat_marg
        self.trim_maps = trim_maps
        self.trim_ref_ind = trim_ref_ind
        self.trim_window = trim_window

        self.col_error_range = col_error_range
        if col_error_range is None:
            self.col_error_range = range(dataset.images[0].shape[0]) # image_width


