# U-Net for image Segmentation

This repository contains the code to train and evaluate a U-net model. The code in this repository is 
based on the paper "Automatic choroidal segmentation in OCT images using supervised deep learning methods"

Link: https://www.nature.com/articles/s41598-019-49816-4

If the code and methods here are useful to you and aided in your research, please consider citing the paper.

The code in this repository is based on the repository hosted at: www.github.com/bioteam/ML-Image-Segmentation.git

# Dependencies 

The `requirements.txt` file contains the dependencies.

After creating a virtual environment run:

`pip3 install -r requirements.txt`

# Instructions

The code in this repository only contains model code and doesn't contain any code related to images preprocessing.

# Model API

## Training

In order to train this model a HDF5 file needs to be provided with the following contents (e.g. see "mouse-image-segmentation" or "porcine-image-segmentation" repositories for instructions on how to generate the datasets):
- train_images: It should contain a 3D matrix with the of the shape: (number of images, image width, image height, 1). These images will be used for training.
- train_segs: It should contain a 3D matrix with the boundaries corresponding to the `train_images`. The shape of the matrix should be: (number of images, number of boundaries, image width)
- val_images: It should contain a 3D matrix with the of the shape: (number of images, image width, image height, 1). These images will be used for validation.
- val_segs: It should contain a 3D matrix with the "segmentation maps" corresponding to the `val_images`. The shape of the matrix should be: (number of images, number of boundaries, image width)

To train the model:



## Model Evaluation

### Test Dataset

### Evaluation

To evaluate the model the function `eval_model()` int `unet/model/evaluation.py` should be invoked. This function takes a `EvalParameters` object (see `unet/model/evaluation_parameters.py`). The `evaluation-scripts` directory contains an example on how to evaluate a model. Usage:

`python3 evaluate_model.py <path/to/model/file> </path/to/test/dataset/file> </output/dir/path>`

For example:

```
python3 evaluate_model.py ~/models/model_epoch52.hdf5 ~/datasets/test_dataset.hdf5 ~/eval-results
```

## Instructions
