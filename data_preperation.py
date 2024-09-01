import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# *************** PART 1 ***************** 
def prepare_data():
    os.makedirs(os.path.join("affectnet_data", "train"), exist_ok=True)
    os.makedirs(os.path.join("affectnet_data", "test"), exist_ok=True)
    os.makedirs(os.path.join("affectnet_data", "val"), exist_ok=True)

    # 1->anger, 2->happy, 3->sad, 4->fear, 5->neutral, 6->disgust, 7->surprise 
    # Paths
    source_dir = 'affectnet_dataset'
    target_dir = 'affectnet_data'
    
    # Make sure subdirectories for expressions exist in train, val, and test directories
    expressions = ['1', '2', '3', '4', '5', '6', '7']
    for expr in expressions:
        os.makedirs(os.path.join(target_dir, 'train', expr), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'val', expr), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'test', expr), exist_ok=True)

    # Split the data
    for expr in expressions:
        expr_folder = os.path.join(source_dir, expr)
        images = os.listdir(expr_folder)

        # Split into train, val, and test (80%, 10%, 10%)
        train_images, test_val_images = train_test_split(images, test_size=0.2, random_state=42)
        val_images, test_images = train_test_split(test_val_images, test_size=0.5, random_state=42)

        # Copy images to train directory
        for img in train_images:
            src = os.path.join(expr_folder, img)
            dst = os.path.join(target_dir, 'train', expr, img)
            shutil.copy2(src, dst)

        # Copy images to val directory
        for img in val_images:
            src = os.path.join(expr_folder, img)
            dst = os.path.join(target_dir, 'val', expr, img)
            shutil.copy2(src, dst)

        # Copy images to test directory
        for img in test_images:
            src = os.path.join(expr_folder, img)
            dst = os.path.join(target_dir, 'test', expr, img)
            shutil.copy2(src, dst)

# *********************************** PART 2 **********************************

def load_data(height, width, batch_size):
    target_dir = 'affectnet_data'
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(target_dir, 'train'),
        labels='inferred',
        label_mode='categorical',
        image_size=(height, width),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(target_dir, 'val'),
        labels='inferred',
        label_mode='categorical',
        image_size=(height, width),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(target_dir, 'test'),
        labels='inferred',
        label_mode='categorical',
        image_size=(height, width),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Data Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomHeight(0.1),
        tf.keras.layers.RandomWidth(0.1)
    ])

    # Apply data augmentation only to training data
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Prefetch to improve performance
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
    """train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,  # Reduce rotation range
    width_shift_range=0.1,  # Reduce width shift range
    height_shift_range=0.1,  # Reduce height shift range
    shear_range=0.1,  # Reduce shear range
    zoom_range=0.1,  # Reduce zoom range
    horizontal_flip=True,
    fill_mode='nearest')

    # Path
    target_dir = 'affectnet_data'

    # ImageDataGenerator for validation and testing (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(target_dir, 'train'),
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Validation generator
    val_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(target_dir, 'val'),
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Test generator
    test_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(target_dir, 'test'),
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator, test_generator"""
