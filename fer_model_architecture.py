from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Input

def conv_block(input, filters, conv_kernel=(4, 4), pooling_kernel=(2, 2)):
    x = Conv2D(filters, kernel_size=conv_kernel, activation="relu", padding="same")(input)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size=conv_kernel, activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=pooling_kernel)(x)
    x = Dropout(0.4)(x)
    return x

def create_model(height=48, width=48, channels=1, classes=7):
    # Input layer
    input = Input(shape=(height, width,channels))
    
    # Conv block #1
    x = conv_block(input, 64)
    
    # Conv block #2
    x = conv_block(x, 128)
    
    # Conv block #3
    x = conv_block(x, 256)
    
    # Conv block #4
    x = conv_block(x, 512)

    x = Flatten()(x)
    output = Dense(classes, activation="softmax")(x)

    model = Model(inputs=input, outputs=output)
    return model