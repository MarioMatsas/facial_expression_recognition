import cv2
import tensorflow as tf
import os
import numpy as np
"""# Function to create augmented images
def augment_image(image):
    augmented_images = []
    
    # Original image
    augmented_images.append(image)
    
    # Rotation
    for angle in [15, -15]:
        M = cv2.getRotationMatrix2D((WIDTH // 2, HEIGHT // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (WIDTH, HEIGHT))
        augmented_images.append(rotated)
    
    # Horizontal shift
    for shift in [0.1, -0.1]:
        M = np.float32([[1, 0, WIDTH * shift], [0, 1, 0]])
        shifted = cv2.warpAffine(image, M, (WIDTH, HEIGHT))
        augmented_images.append(shifted)
    
    # Vertical shift
    for shift in [0.1, -0.1]:
        M = np.float32([[1, 0, 0], [0, 1, HEIGHT * shift]])
        shifted = cv2.warpAffine(image, M, (WIDTH, HEIGHT))
        augmented_images.append(shifted)
    
    return augmented_images"""


# Image size
HEIGHT = 96
WIDTH = 96
NUM_CATEGORIES = 7

# Get working and data directory
cur_work_dir = os.getcwd()
data_path = "archive"
data_dir = os.path.join(cur_work_dir, data_path)

# Mapping categories to integer labels
category_to_label = {'happy': 0, 'sad': 1, 'angry': 2, 'neutral': 3, 'fear': 4, 'disgust': 5, 'surprise': 6}

# Initialize lists for data
training_inputs = []
training_labels = []
testing_inputs = []
testing_labels = []

# Go through the testing and training data sets
for directory in os.listdir(data_dir):
    for category in os.listdir(os.path.join(data_dir, directory)):
        for img in os.listdir(os.path.join(data_dir, directory, category)):
            # Translate the images into a numpy multidimensional array
            image = cv2.imread(os.path.join(data_dir, directory, category, img))
            # Resize to ensure consistency
            image = cv2.resize(image, (WIDTH, HEIGHT))
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Add channel dimension
            image = np.expand_dims(image, axis=-1)
            
            # Create augmented images
            #augmented_images = augment_image(image)
            
            # Append data
            #for aug_img in augmented_images:
            # Append data
            if directory == "test":
                testing_inputs.append(image)
                testing_labels.append(category_to_label[category])
            else:
                training_inputs.append(image)
                training_labels.append(category_to_label[category])  

# Convert lists to NumPy arrays
training_inputs = np.array(training_inputs)
training_labels = np.array(training_labels)
testing_inputs = np.array(testing_inputs)
testing_labels = np.array(testing_labels)

# Verify shapes
print(f"Training inputs shape: {training_inputs.shape}")
print(f"Testing inputs shape: {testing_inputs.shape}")

# Normalize the image data
training_inputs = training_inputs / 255.0
testing_inputs = testing_inputs / 255.0

# Convert labels to one-hot encoding
training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=NUM_CATEGORIES)
testing_labels = tf.keras.utils.to_categorical(testing_labels, num_classes=NUM_CATEGORIES)

# Define the model to recognize facial expressions
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(HEIGHT, WIDTH, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TODO Data augmentation

# Fit the model with data augmentation
history = model.fit(training_inputs, training_labels, epochs=20)

# Save the model
model.save("emotion_recognition_model.keras")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(testing_inputs, testing_labels)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

print("FIN")

