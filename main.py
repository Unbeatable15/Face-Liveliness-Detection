import cv2
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import model_from_json

# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/antispoofing_model_mobilenet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load antispoofing model weights
model.load_weights('antispoofing_models/antispoofing_model_40-0.995714.h5')
print("Model loaded from disk")


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y - 5:y + h + 5, x - 5:x + w + 5]
        resized_face = cv2.resize(face, (160, 160))
        resized_face = resized_face.astype("float") / 255.0
        resized_face = np.expand_dims(resized_face, axis=0)

        # pass the face ROI through the trained liveness detector
        # model to determine if the face is "real" or "fake"
        preds = model.predict(resized_face)[0]

        if preds > 0.5:
            label = 'spoof'
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            label = 'real'
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


def main():
    st.title("Face Spoof Detection App")

    video = cv2.VideoCapture(0)

    while True:
        try:
            ret, frame = video.read()
            processed_frame = process_frame(frame)

            # Display the processed frame using st.image
            st.image(processed_frame, channels="BGR", use_column_width=True, caption="Face Spoof Detection")

        except Exception as e:
            pass


if __name__ == "__main__":
    main()
