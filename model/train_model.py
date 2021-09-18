import augmentation as aug
import custom_losses
import custom_metrics
import image_database as imdb
import training_config as cfg
import readdirimages
import model
import training
import training_parameters as tparams

from keras.utils import to_categorical
import h5py
import keras.optimizers

if cfg.CHANNELS_LAST:
    keras.backend.set_image_data_format('channels_last')

training_hdf5_file = h5py.File(cfg.TRAINING_DATA, 'r')

# images numpy array should be of the shape: (number of images, image width, image height, 1)
# labels numpy array should be of the shape: (number of images, image width, image height, 1)
train_images, train_labels = readdirimages.load_training_data(training_hdf5_file)
val_images, val_labels = readdirimages.load_validation_data(training_hdf5_file)

train_labels = to_categorical(train_labels, cfg.NUM_CLASSES)
val_labels = to_categorical(val_labels, cfg.NUM_CLASSES)

train_imdb = imdb.ImageDatabase(images=train_images, labels=train_labels, name=cfg.TRAINING_DATASET_NAME, filename=cfg.TRAINING_DATA, mode_type='fullsize', num_classes=cfg.NUM_CLASSES)
val_imdb = imdb.ImageDatabase(images=val_images, labels=val_labels, name=cfg.TRAINING_DATASET_NAME, filename=cfg.TRAINING_DATA, mode_type='fullsize', num_classes=cfg.NUM_CLASSES)

model_standard = model.unet(8, 4, 2, (3, 3), (2, 2), input_channels=cfg.INPUT_CHANNELS, output_channels=cfg.NUM_CLASSES)

opt_con = keras.optimizers.Adam
opt_params = {}     # default params
loss = custom_losses.dice_loss
metric = custom_metrics.dice_coef
epochs = cfg.EPOCHS
batch_size = 10

aug_fn_args = [(aug.no_aug, {}), (aug.flip_aug, {'flip_type': 'left-right'})]

aug_mode = 'one'
aug_probs = (0.5, 0.5)
aug_val = False
aug_fly = True

train_params = tparams.TrainingParams(model_standard, opt_con, opt_params, loss, metric, epochs, batch_size, model_save_best=True, aug_fn_args=aug_fn_args, aug_mode=aug_mode,
                                      aug_probs=aug_probs, aug_val=aug_val, aug_fly=aug_fly)

training.train_network(train_imdb, val_imdb, train_params)
