import augmentation as aug
import custom_losses
import custom_metrics
import image_database as imdb
import training_config as cfg
import dataset_loader
import dataset_construction
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
# segments numpy array should be of the shape: (number of images, number of boundaries, image width)
train_images, train_segs = dataset_loader.load_training_data(training_hdf5_file)
val_images, val_segs = dataset_loader.load_validation_data(training_hdf5_file)

train_labels = dataset_construction.create_all_area_masks(train_images, train_segs)
val_labels = dataset_construction.create_all_area_masks(val_images, val_segs)

num_classes = train_segs.shape[1] + 1

train_labels = to_categorical(train_labels, num_classes)
val_labels = to_categorical(val_labels, num_classes)

train_imdb = imdb.ImageDatabase(images=train_images, labels=train_labels, name=cfg.TRAINING_DATASET_NAME, filename=cfg.TRAINING_DATA, mode_type='fullsize', num_classes=num_classes)
val_imdb = imdb.ImageDatabase(images=val_images, labels=val_labels, name=cfg.TRAINING_DATASET_NAME, filename=cfg.TRAINING_DATA, mode_type='fullsize', num_classes=num_classes)

model_standard = model.unet(8, 4, 2, (3, 3), (2, 2), input_channels=cfg.INPUT_CHANNELS, output_channels=num_classes)

opt_con = keras.optimizers.Adam
opt_params = {}     # default params
loss = custom_losses.dice_loss
metric = custom_metrics.dice_coef
epochs = cfg.EPOCHS
batch_size = cfg.BATCH_SIZE

if batch_size > train_images.shape[0]:
    print(f"The batch size ({batch_size}) cannot be larger than the number of training samples ({train_images.shape[0]})")
    exit(1)

if batch_size > val_images.shape[0]:
    print(f"The batch size ({batch_size}) cannot be larger than the number of validation samples ({val_images.shape[0]})")
    exit(1)

aug_fn_args = [(aug.no_aug, {}), (aug.flip_aug, {'flip_type': 'left-right'})]

aug_mode = 'one'
aug_probs = (0.5, 0.5)
aug_val = False
aug_fly = True

train_params = tparams.TrainingParams(model_standard, opt_con, opt_params, loss, metric, epochs, batch_size, model_save_best=True, aug_fn_args=aug_fn_args, aug_mode=aug_mode,
                                      aug_probs=aug_probs, aug_val=aug_val, aug_fly=aug_fly)

training.train_network(train_imdb, val_imdb, train_params)
