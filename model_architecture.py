import tensorflow as tf
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Dense # Layers we will use
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import os
import matplotlib.pyplot as plt

def conv_block(input, filters, conv_kernel=(4, 4), pooling_kernel=(2, 2)):
    x = Conv2D(filters, kernel_size=conv_kernel, activation="relu", padding="same")(input)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size=conv_kernel, activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=pooling_kernel)(x)
    x = Dropout(0.5)(x)
    return x

def create_model(height=96, width=96, channels=3, classes=7):
    # Input layer
    input = Input(shape=(height, width,channels))
    
    # Conv block #1
    x = conv_block(input, 64)
    
    # Conv block #2
    x = conv_block(x, 128)
    
    # Conv block #3
    x = conv_block(x, 256)
    
    # Conv block #4
    #x = conv_block(x, 256)
    
    # Conv block #5
    x = conv_block(x, 512)

    x = Flatten()(x)
    output = Dense(classes, activation="softmax")(x)

    model = Model(inputs=input, outputs=output)
    return model

def train_model(model, train, val, test, eps):
    # Create callbacks
    early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=15,
    verbose=1,
    restore_best_weights=True,
    )

    checkpoint_callback = ModelCheckpoint(
        filepath="fer_checkpoint.keras",
        monitor='val_accuracy',
        save_weights_only=False,
        save_best_only=True,
        save_freq="epoch",
        verbose=1
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.2,
        patience=10, 
        min_lr=0.00001
    )

    tensorboard_callback = TensorBoard(log_dir="log_dir", histogram_freq=1)
    # Train model
    history = model.fit(train, epochs=eps, validation_data=val, callbacks=[early_stopping, checkpoint_callback, lr_scheduler, tensorboard_callback])

    # Evaluate model
    evaluation = model.evaluate(test)
    print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

    # Save model
    model.save('affectnet_model.keras')

    model_json = model.to_json()
    with open("affectnet_model.json", "w") as f:
        f.write(model_json)

    model.save_weights("affectnet_model.weights.h5")

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
