from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Input
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import sys

# Constants
WIDTH = 48
HEIGHT = 48
NUM_CLASSES = 7
EPOCHS = 80

# Preprocess images
train_data_gen = ImageDataGenerator(rescale=1.0/255)
test_data_gen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_data_gen.flow_from_directory(
    os.path.join("data", "train"),
    target_size = (WIDTH, HEIGHT),
    color_mode = "grayscale",
    class_mode = "categorical" # To create the categories
)

test_generator = test_data_gen.flow_from_directory(
    os.path.join("data", "test"),
    target_size = (WIDTH, HEIGHT),
    color_mode = "grayscale",
    class_mode = "categorical"
)

#sys.exit()

# Create model
model = Sequential()
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
model.add(Dense(NUM_CLASSES, activation="softmax"))

"""
model = Sequential([
Input(shape=(48, 48, 1)),
Conv2D(512, (3,3), activation="relu", padding="same"),
BatchNormalization(),
Conv2D(256, (3,3), activation="relu", padding="same"),
BatchNormalization(),
MaxPool2D(2),
Dropout(0.5),
Conv2D(128, (3,3), activation="relu", padding="same"),
BatchNormalization(),
Conv2D(64, (3,3), activation="relu", padding="same"),
BatchNormalization(),
MaxPool2D(2),
Dropout(0.5),
Conv2D(32, (3,3), activation="relu", padding="same"),
MaxPool2D(2),
Dropout(0.5),
Flatten(),
Dense(NUM_CLASSES, activation="softmax")
])"""

model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])

# Reduce learning rate when a metric has stopped improving
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    early_stopping,
    lr_scheduler,
]

# Train model
model_info = model.fit(
    train_generator,
    #steps_per_epoch = 21875 // 64,
    epochs = EPOCHS,
    validation_data = test_generator,
    callbacks = callbacks,
    #validation_steps = 6036 // 64
)

evaluation = model.evaluate(test_generator)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

# Save model
model.save('my_model_mario4.keras')

model_json = model.to_json()
with open("mmm4.json", "w") as f:
    f.write(model_json)

model.save_weights("mmm4.weights.h5")


