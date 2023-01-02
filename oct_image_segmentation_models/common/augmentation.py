import numpy as np
import time


def augment_dataset(images, masks, segs, aug_fn_arg):
    start_augment_time = time.time()

    num_images = len(images)
    aug_fn = aug_fn_arg[0]
    aug_arg = aug_fn_arg[1]

    augmented_images = np.zeros_like(images)
    augmented_masks = np.zeros_like(masks)
    augmented_segs = np.zeros_like(segs)

    for i in range(num_images):
        image = images[i, :, :]
        mask = masks[i, :, :]
        seg = segs[i, :, :]
        (
            augmented_images[i, :, :],
            augmented_masks[i, :, :],
            augmented_segs[i, :, :],
            _,
            _,
        ) = aug_fn(image, mask, seg, aug_arg)

    aug_desc = aug_fn(None, None, aug_arg, True)

    end_augment_time = time.time()
    total_aug_time = end_augment_time - start_augment_time

    return [
        augmented_images,
        augmented_masks,
        augmented_segs,
        aug_desc,
        total_aug_time,
    ]


def no_aug(image, mask, _aug_args, desc_only=False):
    if desc_only is False:
        return image, mask
    else:
        desc = "no aug"
        return desc


def flip_aug(image, mask, aug_args, desc_only=False):
    flip_type = aug_args["flip_type"]

    if flip_type == "up-down":
        axis = 0
    elif flip_type == "left-right":
        axis = 1

    if desc_only is False:
        aug_image = np.flip(image, axis=axis)
        if mask is not None:
            aug_mask = np.flip(mask, axis=axis)
        else:
            aug_mask = None

        return aug_image, aug_mask
    else:
        aug_desc = "flip aug: " + flip_type
        return aug_desc


augmentation_map = {
    "no_augmentation": no_aug,
    "flip": flip_aug,
}


def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))
