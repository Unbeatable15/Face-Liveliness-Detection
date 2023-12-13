from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
json_file = open('antispoofing_models/antispoofing_model_mobilenet.json', 'r')
loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights('antispoofing_models/antispoofing_model_40-0.995714.h5')
print("Model loaded from disk")

video = None  # Initialize video capture instance

def process_frame(frame):
    global video  # Access the global video object
    if video is None:  # Check if video capture instance is not initialized
        video = cv2.VideoCapture(0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y - 5:y + h + 5, x - 5:x + w + 5]
        resized_face = cv2.resize(face, (160, 160))
        resized_face = resized_face.astype("float") / 255.0
        resized_face = np.expand_dims(resized_face, axis=0)

        preds = model.predict(resized_face)[0]

        label = 'spoof' if preds > 0.5 else 'real'
        color = (0, 0, 255) if label == 'spoof' else (0, 255, 0)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame

def generate_frames():
    while True:
        success, frame = video.read()
        if not success:
            break
        processed_frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global video
    if video is not None:
        video.release()
    return 'Server shutting down...'

if __name__ == "__main__":
    app.run(debug=True)
