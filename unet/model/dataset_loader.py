def load_training_data(hdf5_data_file):
    train_images = hdf5_data_file["train_images"][:]
    train_labels = hdf5_data_file["train_labels"][:]
    train_segs = hdf5_data_file.get("train_segs")
    if train_segs:
        train_segs = train_segs[:]

    return train_images, train_labels, train_segs


def load_validation_data(hdf5_data_file):
    val_images = hdf5_data_file["val_images"][:]
    val_labels = hdf5_data_file["val_labels"][:]
    val_segs = hdf5_data_file.get("val_segs")
    if val_segs:
        val_segs = val_segs[:]

    return val_images, val_labels, val_segs


def load_testing_data(hdf5_data_file):
    test_images = hdf5_data_file['test_images'][:]
    test_labels = hdf5_data_file["test_labels"][:]
    test_segs = hdf5_data_file.get("test_segs")
    if test_segs:
        test_segs = test_segs[:]
    test_image_names = [f"image_{i}" for i in range(len(test_images))]

    return test_images, test_labels, test_segs, test_image_names
