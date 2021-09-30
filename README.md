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
1. In `unet/training_config.py`:
    - Point `TRAINING_DATA` to the location of the hdf5 dataset that should be used for training
    - Modify rest of parameters accordingly

2. 

## Evaluation


## Instructions



Extract images and segmentation files from example_data.hdf5 with `hdf5readimages.py`

## Run with example_data.hdf5

1. For a quick test, set `EPOCHS = 100` in `parameters.py`.
2. Create `results` and `data` directories at the root level directory of
   this repository.  
3. In `parameters.py`, update the `RESULTS_LOCATION` and `DATA_LOCATION` to
   match the paths of the two newly created directories. Examples:
   * `DATA_LOCATION = '/Users/user1/git/ML-Image-Segmentation/data/'`
   * `RESULTS_LOCATION = '/Users/user1/git/ML-Image-Segmentation/results/'`
4. Set `BIOTEAM = 0` in `parameters.py` and save the file.
5. Install a conda environment and all the necessary dependecies by running:

`conda env create --name ml_env --file environment.yml`

6. Activate the conda environment with `conda activate ml_env` 

7. Run `python train_script_semantic_general.py`

### What to expect

* During training, the dice_coef (Dice Coefficient) increases as tensorflow
  converges on a better model
* The results directory has one `config.hdf5` file and several hdf5 files
  asigned to an `epoch` number 
* To read any hdf5 file from the `results` directory, run `hdf5scan.py`

## BioTeam version: Train by reading images from a directory

1. Create a `remlmaterials` directory at the root level directory of this
   repository with the following sub-directories: `train_images`, `train_segs`,
   `val_images`, `val_segs`, `test_images`, `test_segs`.
    * Example data has been added at the root level that you can use for
      testing in a directory called `remlmaterials`.
2. Copy the files into the corresponding directories. __A minimum of 3 train and val files is required for training__
3. In `parameters.py`, update the `INPUT_LOCATION` to match the directory
   created in step one. (Example: `INPUT_LOCATION =
   '/Users/user1/git/ML-Image-Segmentation/remlmaterials/'`)
4. Set `BIOTEAM = 1` in `parameters.py` and save the file.
5. As before,  activate the conda environment with `conda activate ml_env`.
6. Run `python train_script_semantic_general.py`

### What to expect
1. readdirimages.py will create an hdf5 file `img_data.hdf5` with the images
   and segs files in the same format as the `example_data.hdf5`
2. `img_data.hdf5` can be read with `hdf5readimages.py`
3. `img_data.hdf5` cannot be used as input to the original Kugelman et al code
   (BIOTEAM = 0) because it contains areas, not boundaries
4. All other files and results are in the same format as before

# Training a model (patch-based)
1. Modify *load_training_data* and *load_validation_data* functions in
   *train_script_patchbased_general.py* to load your training and validation
   data (see comments in code). [see example data file and load functions]
2. Choose one of the following and pass as first training parameter as shown in code:
    * *model_cifar* (Cifar CNN)
    * *model_complex* (Complex CNN) [default]
    * *model_rnn* (RNN)
3. Can change the desired patch size (*PATCH_SIZE*) as well as the name of your
   dataset (*DATASET_NAME*).
4. Run *train_script_patchbased_general.py*
5. Training results will be saved in the location defined by *parameters.RESULTS_LOCATION*. Each new training run will be saved in a new seperate folder named with the format: 
(*TIMESTAMP*) _ (*MODEL_NAME*) _ (*DATASET_NAME*). Each folder will contain the following files:
    * *config.hdf5* (summary of parameters used for training)
    * *stats_epoch#.hdf5* (training and validation results for each epoch up to epoch #)
    * one or more *model_epoch&.hdf5* files containing the saved model at each epoch &
  
# Training a model (semantic)
1. Modify *load_training_data* and *load_validation_data* functions in *train_script_semantic_general.py* to load your training and validation data (see comments in code). [see example data file and load functions]
2. Choose one of the following and pass as first training parameter as shown in code:
    * *model_residual* (Residual U-Net)
    * *model_standard* (Standard U-Net) [default]
    * *model_sSE* (Standard U-Net with sSE blocks)
    * *model_cSE* (Standard U-Net with cSE blocks)
    * *model_scSE* (Standard U-Net with scSE blocks)
3. Can change the name of your dataset (*DATASET_NAME*).
4. Run *train_script_semantic_general.py*
5. Training results will be saved in the location defined by *parameters.RESULTS_LOCATION*. Each new training run will be saved in a new seperate folder named with the format: 
(*TIMESTAMP*) _ (*MODEL_NAME*) _ (*DATASET_NAME*). Each folder will contain the following files:
    * *config.hdf5* (summary of parameters used for training)
    * *stats_epoch#.hdf5* (training and validation results for each epoch up to epoch #)
    * one or more *model_epoch&.hdf5* files containing the saved model at each epoch &
  
# Evaluating a model (patch-based)
1. Modify *load_testing_data* function in *eval_script_patchbased_general.py*
   to load your testing data (see comments in code). [see example data file and
   load function]
2. Specify trained network folder to evaluate.
3. Specify filename of model to evaluate within the chosen folder:
   *model_epoch&.hdf5*
4. Run *eval_script_patchbased_general.py*
5. Evaluation results will be saved in a new folder (with the name
   *no_aug_(DATASET_NAME).hdf5*) within the specified trained network folder.
   Within this, a folder is created for each evaluated image containing a range
   of .png images illustrating the results qualitatively as well as an
   *evaluations.hdf5* file with all quantitative results. A new *config.hdf5*
   file is created in the new folder as well as *results.hdf5* and
   *results.csv* files summarising the overall results after all images have
   been evaluated.
  
# Evaluating a model (semantic)
1. Update `MODEL_LOCATION` in `parameters.py` to point to the sub-directory
   generated during the training within the `results` folder. (Example:
   `MODEL_LOCATION = '/2021-06-21 17_02_56 U-net exampledata/'`)
2. Update `MODEL_NAME` in `parameters.py` to point to the largest epoch file
   generated during the training contained within the `MODEL_LOCATION`
   sub-directory rreferenced in the previous step. (Example: `MODEL_NAME =
   'model_epoch100.hdf5'`)
4. Run *eval_script_semantic_general.py*
5. Evaluation results will be saved in a new folder (with the name
   *no_aug_(DATASET_NAME).hdf5*) within the specified trained network folder.
   Within this, a folder is created for each evaluated image containing a range
   of .png images illustrating the results qualitatively as well as an
   *evaluations.hdf5* file with all quantitative results. A new *config.hdf5*
   file is created in the new folder as well as *results.hdf5* and
   *results.csv* files summarising the overall results after all images have
   been evaluated.

# Still to be added
* Code and instructions for preprocessing using contrast enhancement (Girard filter)
