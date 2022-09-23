from pathlib import Path

from unet.model import augmentation as aug


class TrainingParams:
    """
    Parameters for training a network:

        aug_fn_args: tuple of two-tuples containing augmentation function and \
            argument pairs
        _________

        aug_mode: mode to use for augmentation

        none: no augmentations -> will just use what is in the images and \
        labels arrays as is
        one: for each image, one augmentation will be picked from the list of \
        possible augmentation functions chosen based on probabilities in \
        aug_probs.
        all: for each image, all augmentations will be performed creating a \
        new separate image for each

        note that for patch mode: augs are applied to the full size images \
        before being broken into patches
        _________

        aug_probs: probabilities used for selecting augmentations in 'one' \
        mode. Should be values between 0 and 1 which add to 1.
        _________

        aug_val: boolean used to apply the same augmentation policy to the \
        validation dataset.
        _________
    """

    def __init__(
        self,
        training_dataset_path: Path,
        training_dataset_name: str,
        initial_model: Path,
        results_location: Path,
        opt_con,
        opt_params,
        loss,
        metric,
        epochs,
        batch_size,
        aug_fn_args=((aug.no_aug, {}),),
        aug_mode="none",
        aug_probs=(),
        aug_fly=False,
        aug_val=True,
        shuffle=True,
        model_save_best=True,
        model_save_monitor=("val_acc", "max"),
        normalise=True,
        use_gen=True,
        use_tensorboard=False,
        class_weight=None,
        channels_last: bool = True,
    ):
        self.training_dataset_path = training_dataset_path
        self.training_dataset_name = training_dataset_name
        self.initial_model = initial_model
        self.results_location = results_location
        self.opt_con = opt_con
        self.opt_params = opt_params
        self.loss = loss
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size
        self.aug_fn_args = aug_fn_args
        self.aug_mode = aug_mode
        self.aug_probs = aug_probs
        self.aug_fly = aug_fly
        self.aug_val = aug_val
        self.shuffle = shuffle
        self.normalise = normalise
        self.model_save_best = model_save_best
        self.model_save_monitor = model_save_monitor
        self.use_gen = use_gen
        self.use_tensorboard = use_tensorboard
        self.class_weight = class_weight
        self.channels_last = channels_last

        if type(loss) is not str:
            self.loss_name = self.loss.__name__
        else:
            self.loss_name = self.loss

        if type(metric) is not str:
            self.metric_name = self.metric.__name__
        else:
            self.metric_name = self.metric

        if self.model_save_monitor[0] == "val_acc":
            self.model_save_monitor = [
                "val_" + self.metric_name,
                model_save_monitor[1],
            ]
