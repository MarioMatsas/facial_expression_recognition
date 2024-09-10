import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import fer_data_prep as fdp
import fer_model_architecture as fma

# Constants
WIDTH = 48
HEIGHT = 48
NUM_CLASSES = 7
EPOCHS = 60

# Prepare the data
#fdp.prepare_data()

# Load training data
train_images, train_labels = fdp.load_data(HEIGHT, WIDTH, NUM_CLASSES, os.path.join("dataset_affect_trainable", "train"))

# Load validation data
val_images, val_labels = fdp.load_data(HEIGHT, WIDTH, NUM_CLASSES, os.path.join("dataset_affect_trainable", "val"))

# Load test data
test_images, test_labels = fdp.load_data(HEIGHT, WIDTH, NUM_CLASSES, os.path.join("dataset_affect_trainable", "test"))

model = fma.create_model(HEIGHT, WIDTH, 1, NUM_CLASSES)

model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.0005), metrics=["accuracy"])

# Reduce learning rate when accuracy has stopped improving
lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.2,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

datagen = ImageDataGenerator(
        zoom_range=0.2,          
        rotation_range=10,       
        width_shift_range=0.1,   
        height_shift_range=0.1,  
        horizontal_flip=True,    
        vertical_flip=False      
)     

# Train model
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=64),
    epochs=EPOCHS,
    validation_data=(val_images, val_labels),
    callbacks=[lr_scheduler]
)

# Evaluate model
evaluation = model.evaluate(test_images, test_labels)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

# Save model
model.save('aff_model.keras')

# Visualize results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

# Plot training and validation accuracy
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

