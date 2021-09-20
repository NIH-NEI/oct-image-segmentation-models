import numpy as np
import h5py
from PIL import Image, ImageOps, ImageDraw

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