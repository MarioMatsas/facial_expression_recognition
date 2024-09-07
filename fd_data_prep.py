import os
from PIL import Image
import shutil

def load_data_from_csv(in_label_file, in_image_dir, out_label_dir, out_image_dir):
    # Create the output directories if they don't exist
    os.makedirs(out_label_dir, exist_ok=True)
    os.makedirs(out_image_dir, exist_ok=True)

    current_image = None
    image_counter = 1  # Start numbering from 1

    with open(in_label_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith('#'):
                # New image label
                current_image = line[1:].strip()
                image_path = os.path.join(in_image_dir, current_image)

                # Get the image dimensions
                with Image.open(image_path) as img:
                    image_width, image_height = img.size

                # Define the output filenames
                output_image_file = f"{image_counter}.jpg"
                output_label_file = f"{image_counter}.txt"

                # Save the image in the output directory with the new name
                output_image_path = os.path.join(out_image_dir, output_image_file)
                shutil.copy(image_path, output_image_path)

                # Create the corresponding label file
                output_label_path = os.path.join(out_label_dir, output_label_file)

                # Increment the counter for the next image
                image_counter += 1

                with open(output_label_path, 'w') as out_label:
                    continue

            else:
                # Process the bounding box
                coords = list(map(int, line.split()))
                x_min, y_min, x_max, y_max = coords

                # Convert to YOLO format
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height

                # Prepare the YOLO label line
                yolo_label = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

                # Write the label to the corresponding file
                with open(output_label_path, 'a') as out_label:
                    out_label.write(yolo_label)

def split_data(images_source_dir, labels_source_dir, images_train_dir, images_val_dir, labels_train_dir, labels_val_dir):
    # Define the source and destination directories
    images_source_dir = 'yolo_images'  # Path to your images folder
    labels_source_dir = 'yolo_labels'  # Path to your labels folder
    images_train_dir = 'data/images/train'
    images_val_dir = 'data/images/val'
    labels_train_dir = 'data/labels/train'
    labels_val_dir = 'data/labels/val'

    # Create the destination directories if they don't exist
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    # List all images and labels and sort them
    images = sorted(os.listdir(images_source_dir))
    labels = sorted(os.listdir(labels_source_dir))

    # Ensure that the number of images matches the number of labels
    assert len(images) == len(labels), "Number of images and labels must be the same."

    # Ensure that each image has a corresponding label by matching filenames
    for image in images:
        # Remove the extension and add '.txt' to get the corresponding label name
        image_name = os.path.splitext(image)[0]
        label_name = image_name + '.txt'

        # Ensure the corresponding label exists
        if label_name not in labels:
            raise ValueError(f"Label file {label_name} not found for image {image}.")

    # Split the data: first 150 go to validation, the rest to training
    for i, image in enumerate(images):
        label = os.path.splitext(image)[0] + '.txt'

        if i < 150:
            shutil.copy(os.path.join(images_source_dir, image), os.path.join(images_val_dir, image))
            shutil.copy(os.path.join(labels_source_dir, label), os.path.join(labels_val_dir, label))
        else:
            shutil.copy(os.path.join(images_source_dir, image), os.path.join(images_train_dir, image))
            shutil.copy(os.path.join(labels_source_dir, label), os.path.join(labels_train_dir, label))