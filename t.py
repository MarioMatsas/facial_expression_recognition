import cv2
import tensorflow as tf
import numpy as np
from keras.models import model_from_json

# Load the trained model
model = tf.keras.models.load_model("my_model_mario4.keras") #model.h5 my_model_mario2.keras
"""with open('mmm2.json', 'r') as json_file:
    model_json = json_file.read()

# Recreate the model from JSON data
model = model_from_json(model_json)
# Load the weights into the model
model.load_weights('mmm2.weights.h5')""" " Neutral, "

#class_names = ["Angry", "Disgust", "Fear", "Happy","Sad", "Surprise"]  # Replace with your actual class names
class_names = ["Surprise","Fear","Disgust","Happy","Sad","Angry","Neutral"]
#class_names = ["Angry", "Happy", "Sad", "Surprise"]
#class_ck = ["anger","disgust","fear","happy","sad","surprise","neutral","contempt"]

# Load the pre-trained Haar Cascade face detector
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to preprocess the face image
def preprocess(frame):
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = cv2.resize(frame, (48, 48))  # Resize to model input size
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=-1)  # Add a channel dimension

    return frame


def detect_face(frame):
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return gray, frame, faces
# Start capturing video from the webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()  # Read frame from webcam

    if ret:
        gray, color_frame, faces = detect_face(frame)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]  # Extract the face
            face = preprocess(face)  # Preprocess the face

            # Expand dimensions and predict emotion
            face_exp = np.expand_dims(face, axis=0)
            predictions = model.predict(face_exp)
            idx = np.argmax(predictions)
            confidence = predictions[0][idx]
            
            if confidence > 0.3:  # Only display if confidence is above a threshold
                class_name = class_names[idx]
                cv2.putText(color_frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(color_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the result
        cv2.imshow("Facial Expression Detection", color_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
video.release()
cv2.destroyAllWindows()
