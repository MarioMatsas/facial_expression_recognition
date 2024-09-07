import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO

# Load the YOLOv8 model for face detection
yolo_model = YOLO('models/yolov8_nano.pt')

# Load the AffectNet model for emotion recognition
emotion_model = load_model('models/aff_model_test_2.keras')

# 1->anger, 2->happy, 3->sad, 4->fear, 5->neutral, 6->disgust, 7->surprise 
emotion_labels = ['Angry', 'Happy', 'Sad', 'Fear', 'Neutral', 'Disgust', 'Surprise']

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLOv8 to detect faces
    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy.numpy()  # Extract bounding boxes

    for box in boxes:
        # Extract face region
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue
        
        # Preprocess face for emotion recognition
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert BGR (OpenCV format) to RGB (model format)
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0  # Normalize

        # Predict emotion
        emotion_prob = emotion_model.predict(face)
        emotion_label = emotion_labels[np.argmax(emotion_prob)]

        # Draw bounding box and emotion label on the original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('fd & fer', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()