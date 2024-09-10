import os
from ultralytics import YOLO
import fd_data_prep as fddp
import sys

# Preapre all the data according to the yaml file
#in_labels = os.path.join("Dataset_FDDB", "label.txt")
#in_images = os.path.join("Dataset_FDDB", "images")
#fddp.load_data_from_csv(in_labels, in_images, "yolo_labels", "yolo_images")
#fddp.split_data("yolo_images", "yolo_labels", "fd_data/images/train", "fd_data/images/val", "fd_data/labels/train", "fd_data/labels/val")

model = YOLO("yolov8n.pt")

results = model.train(data=os.path.join("yolov8_config.yaml"), epochs=40)