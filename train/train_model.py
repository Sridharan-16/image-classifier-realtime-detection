import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow.keras import layers, regularizers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Dataset Path
dataset_path = "give_your_dataset_path"

# Image & Batch Size
img_size = (180, 180)
batch_size = 16

# Load dataset with train-validation split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names  # Save class names before mapping
print(f"Classes: {class_names}")

# Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Normalize dataset
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Define CNN model
model = keras.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001), input_shape=(180, 180, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dropout(0.5),  # Reducing Overfitting
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")  # Output Layer
])

# Compile model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Model Checkpointing
checkpoint = keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss", mode="min")

# Train model
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint])

# Save final model
model.save("image_classifier.h5")
print("Model saved as image_classifier.h5")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
