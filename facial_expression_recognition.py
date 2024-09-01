import data_preperation
import model_architecture
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Constants
WIDTH = 96
HEIGHT = 96
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 80

# Get all the data in the correct files
#data_preperation.prepare_data()

# Create the training, validation and testing partitions
train_data, val_data, test_data = data_preperation.load_data(HEIGHT, WIDTH, BATCH_SIZE)

# Create the model
model = model_architecture.create_model()

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0005), metrics=["accuracy"])

# Train and save the model
model_architecture.train_model(model, train_data, val_data, test_data, EPOCHS)

print("DONE!")