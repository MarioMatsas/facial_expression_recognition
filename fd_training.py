import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(data=os.path.join("PATH", "yolov8_config.yaml"), epochs=40)