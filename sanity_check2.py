import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import sys
#from sklearn.preprocessing import OneHotEncoder

# Constants
WIDTH = 48
HEIGHT = 48
NUM_CLASSES = 7
EPOCHS = 80

# Load and preprocess data
csv_path = "fer2013.csv"
data = pd.read_csv(csv_path)

# Split data into training and testing sets
train_data = data[data['Usage'] == 'Training']
test_data = data[data['Usage'].isin(['PublicTest', 'PrivateTest'])]
#print(len(train_data) + len(test_data))


# Convert pixel data to numpy arrays and reshape
def process_pixels(pixels):
    pixels = np.array(pixels.split(), dtype=np.float32)
    return pixels.reshape(WIDTH, HEIGHT, 1) / 255.0

train_images = np.array([process_pixels(pixels) for pixels in train_data['pixels']])
test_images = np.array([process_pixels(pixels) for pixels in test_data['pixels']])

#print(len(train_images) + len(test_images))

# Convert labels to one-hot encoding
train_labels = np.array([int(emotion) for emotion in train_data['emotion']])
test_labels = np.array([int(emotion) for emotion in test_data['emotion']])
#print(len(train_labels) + len(test_labels))
#sys.exit()
# Create tf.data.Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(64)

# Create model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(WIDTH, HEIGHT, 1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(NUM_CLASSES, activation="softmax"))

"""model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3), activation="relu", input_shape=(WIDTH, HEIGHT, 1)))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation="softmax"))"""
"""model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(WIDTH, HEIGHT, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation="softmax"))"""

model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])

# Reduce learning rate when a metric has stopped improving
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

checkpoint_callback = ModelCheckpoint(
    filepath="checkpoint.keras",
    save_weights_only=False,
    save_best_only=True,
    save_freq="epoch",
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.2,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    early_stopping,
    checkpoint_callback,
    lr_scheduler,
]


# Train model
model_info = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=callbacks,
)

# Evaluate model
evaluation = model.evaluate(test_dataset)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

# Save model
model.save('my_model_mario8.keras')

model_json = model.to_json()
with open("mmm8.json", "w") as f:
    f.write(model_json)

model.save_weights("mmm8.weights.h5")

