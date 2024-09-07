import os
import shutil
import pandas as pd
from tensorflow.keras.preprocessing import image
import random
import numpy as np

# *************** PART 1 ***************** 
def prepare_data(): 
    # Load the CSV file
    csv_file = 'labels.csv'
    data = pd.read_csv(csv_file)

    # Define the root directory for moving images
    root_dir = 'dataset_affect'

    # Iterate over each row in the CSV
    for _, row in data.iterrows():
        img_path = "affectnet_dataset/"+row['pth']
        label = str(row['label'])

        # Create the directory if it doesn't exist
        label_dir = os.path.join(root_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        # Define the destination path
        dest_path = os.path.join(label_dir, os.path.basename(img_path))

        # Copy the image to the new folder
        try:
            shutil.copy(img_path, dest_path)
        except FileNotFoundError:
            print(f"File {img_path} not found. Skipping.")
        except Exception as e:
            pass

    # Rename the folders as per this # 1->anger, 2->happy, 3->sad, 4->fear, 5->neutral, 6->disgust, 7->surprise
    # So we can use sparse categorical crossentropy as our loss function

    folder_mapping = {
    'anger': '1',
    'happy': '2',
    'sad': '3',
    'fear': '4',
    'neutral': '5',
    'disgust': '6',
    'surprise': '7'
    }

    # Rename the folders
    for old_name, new_name in folder_mapping.items():
        old_folder_path = os.path.join(root_dir, old_name)
        new_folder_path = os.path.join(root_dir, new_name)

        os.rename(old_folder_path, new_folder_path)


# *********************************** PART 2 **********************************

def load_data(height, width, num_classes, dir):
    images = []
    labels = []
    
    for label in range(1, num_classes + 1):
        label_dir = os.path.join(dir, str(label))
        
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = image.load_img(img_path, target_size=(height, width), color_mode="grayscale")
            img_array = image.img_to_array(img) / 255.0  # Normalize the image
            images.append(img_array)
            labels.append(label - 1)  # Class labels will be 0-indexed

    images = np.array(images)
    labels = np.array(labels)

    # Shuffle the data while keeping image-label pairs together
    data = list(zip(images, labels))
    random.shuffle(data)
    images, labels = zip(*data)
        
    return np.array(images), np.array(labels)
