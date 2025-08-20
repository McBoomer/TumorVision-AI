"""
Defines the CNN model for Brain Tumor MRI classification
"""

from keras import layers, models

def build_model(input_shape=(128,128,3), num_classes=2):
    """
    Builds and compiles a simple CNN model.

    Args:
        input_shape (tuple): Shape of input images (H, W, C)
        num_classes (int): Number of output classes

    Returns:
        model: Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),  # Fully defined input shape
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
