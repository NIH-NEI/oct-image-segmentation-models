import numpy as np
import h5py
from PIL import Image, ImageOps, ImageDraw

# images numpy array should be of the shape: (number of images, image width, image height, 1)
# segs numpy array should be of the shape: (number of images, number of boundaries, image width)

def load_training_data(hdf5_data_file):
    train_images = hdf5_data_file['train_images'][:]
    train_segs = hdf5_data_file['train_labels'][:]

    return train_images, train_segs

# fill in this function to load your data for the validation set with format/shape given above
def load_validation_data(hdf5_data_file):
    val_images = hdf5_data_file['val_images'][:]
    val_segs = hdf5_data_file['val_labels'][:]

    return val_images, val_segs

# Create a hdf5 dataset
def addhdf5_dataset(npimgarray, name, filename):
    h5f = h5py.File(filename, "a")
    try:
        npimgarray = np.asarray(npimgarray)
        if npimgarray.size == 0:
            # Create empty dataset
            h5f.create_dataset(name, dtype="f")
        else:
            h5f.create_dataset(name, data=npimgarray)
        h5data = h5f[name][()]
    finally:
        h5f.close()

    return h5data


# Add channel dimension as 4th dimension to image and labels array, e.g. RGB colors
'''
def channels_last_reshape(images, channels):
    dim = images.ndim
    if dim == 3:
        x, y, z = images.shape
        image_array = images.reshape(x, y, z, channels)
    else:
        image_array = []

    return image_array
'''



# Add 4th dim and create hdf5 dataset
def reshape(images, description, h5filename, channels):

    dataset = np.asarray(images)
    dataset = channels_last_reshape(dataset, channels)
    h5dataset = addhdf5_dataset(dataset, description, h5filename)
    return h5dataset


# Black out areas with scanner name and legend
def black_out(im):

    draw = ImageDraw.Draw(im)
    draw.rectangle(parameters.BLACKOUT_COORDS_LEFT, fill=0)
    draw.rectangle(parameters.BLACKOUT_COORDS_RIGHT, fill=0)
    return im


# Change greyscale values to 1,2,3, ... values
def mask_categorical(dset1):

    uniques = np.unique(dset1)

    for i in range(len(uniques)):
        x = uniques[i]
        dset1 = np.where(dset1 == x, i, dset1)

    return dset1

# When segs contains area labels, just copy over, and add 4th dim for RGB channels
'''
def create_all_area_masks(segs):
    all_masks = channels_last_reshape(segs, 1)
    all_masks = np.array(all_masks)

    return all_masks
'''