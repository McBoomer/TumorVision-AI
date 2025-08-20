"""
Main program to train Brain Tumor MRI classifier

Muadh Khan
08/20/2025
Markham, Ontario
Dataset Sourse https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data
"""

from keras_ml_loader import load_data
from model import build_model


# Config

dataset_dir = "dataset/Brain Tumor MRI Dataset"  # path to folder containing subfolders for each class
img_size = (128, 128)                            # resize all images to 128x128
batch_size = 32                                  # number of images per batch
epochs = 10                                      # number of training epochs


# Load dataset

train_gen, val_gen = load_data(dataset_dir, img_size=img_size, batch_size=batch_size)

# Automatically get number of classes
num_classes = train_gen.element_spec[1].shape[-1]


# Build model

model = build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)


# Train model

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)


# Save trained model

model.save("brain_tumor_model.h5")
print("✅ Training complete. Model saved as brain_tumor_model.h5")
