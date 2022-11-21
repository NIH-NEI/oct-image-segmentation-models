from __future__ import annotations

from pathlib import Path
from typeguard import typechecked

from oct_image_segmentation_models.common import utils


@typechecked
class EvaluationSaveParams:
    def __init__(
        self,
        predicted_labels: bool = True,
        categorical_pred: bool = False,
        png_images: bool = True,
        boundary_maps: bool = True,
    ) -> None:
        self.predicted_labels = predicted_labels
        self.categorical_pred = categorical_pred
        self.png_images = png_images
        self.boundary_maps = boundary_maps


@typechecked
class EvaluationParameters:
    """Parameters for evaluation of trained network.
    _________

    model_path: filename of Keras model to be loaded
    _________

    dataset: Dataset (see 'Dataset' class)
    _________

    save_params: parameters using to determine what to save from the evaluation
    _________
    """

    def __init__(
        self,
        model_path: Path,
        mlflow_tracking_uri: str | None,
        test_dataset_path: Path,
        save_foldername: Path,
        save_params: EvaluationSaveParams,
        transpose=False,
        gsgrad=1,
        dice_errors=True,
        binarize=True,
        bg_ilm=True,
        bg_csi=False,
    ):
        self.model_path = model_path
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.test_dataset_path = test_dataset_path
        self.binarize = binarize

        self.save_params = save_params
        self.transpose = transpose
        self.dice_errors = dice_errors

        self.bg_ilm = bg_ilm
        self.bg_csi = bg_csi

        self.gsgrad = gsgrad

        self.save_foldername = save_foldername
        self.loaded_model = utils.load_model(
            model_path,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        self.num_classes = self.loaded_model.output.shape[-1]
