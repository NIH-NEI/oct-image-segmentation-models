def load_training_data(hdf5_data_file):
    train_images = hdf5_data_file['train_images'][:]
    train_segs = hdf5_data_file['train_labels'][:]

    return train_images, train_segs

def load_validation_data(hdf5_data_file):
    val_images = hdf5_data_file['val_images'][:]
    val_segs = hdf5_data_file['val_labels'][:]

    return val_images, val_segs

def load_testing_data(hdf5_data_file):
    test_images = hdf5_data_file['test_images'][:]
    test_segs = hdf5_data_file['test_labels'][:]
    test_image_names = [f"image_{i}" for i in range(len(test_images))]

    return test_images, test_segs, test_image_names
