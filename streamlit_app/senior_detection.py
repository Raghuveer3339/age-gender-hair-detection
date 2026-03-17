import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import load_model

# Load model
model = load_model("../models/Age_Sex_Detection.h5", compile=False)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Preprocess
        face_resized = cv2.resize(face, (48, 48))
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        # Predict
        preds = model.predict(face_resized)

        age = int(np.round(preds[1][0]))
        gender = "Female" if preds[0][0] > 0.5 else "Male"

        # Senior logic
        if age > 60:
            status = "Senior Citizen"
        else:
            status = "Normal"

        # Label
        label = f"{gender}, {age}, {status}"

        # Draw box
        if age > 60:
         color = (0, 0, 255)   # Red for Senior
        else:
         color = (0, 255, 0)   # Green for Normal
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Save data
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if len(data) == 0 or data[-1][0] != age:
            data.append([age, gender, status, time_now])

    cv2.imshow("Senior Citizen Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save to CSV
df = pd.DataFrame(data, columns=["Age", "Gender", "Status", "Time"])
df.to_csv("senior_detection_results.csv", index=False)

cap.release()
cv2.destroyAllWindows()