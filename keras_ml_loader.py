"""
Loader for Brain Tumor MRI dataset using pure Keras
"""

from keras.utils import image_dataset_from_directory

def load_data(dataset_dir, img_size=(128, 128), batch_size=32):
    """
    Loads train and validation datasets from a directory.

    Args:
        dataset_dir (str): Path to the dataset folder containing class subfolders.
        img_size (tuple): Image height and width to resize to.
        batch_size (int): Number of images per batch.

    Returns:
        train_dataset, val_dataset: Keras Dataset objects
    """
    train_dataset = image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_dataset = image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    # Normalize pixel values from 0-255 to 0-1
    train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
    val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))

    return train_dataset, val_dataset
