import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import os

st.title("✋ Sign Language Detection")

# ================= TIME RESTRICTION =================
current_hour = datetime.now().hour

if False:  # change later to (18 <= current_hour <= 22)
    st.error("⏰ App only works between 6 PM to 10 PM")
    st.stop()

# ================= MEDIAPIPE SETUP =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3   # 🔥 reduced for better detection
)
mp_draw = mp.solutions.drawing_utils

# ================= SIGN LOGIC =================
def predict_sign(hand_landmarks):
    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
        return "Hello 👋"
    elif hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
        return "Yes 👍"
    elif hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        return "No ❌"
    else:
        return "Thanks 🙏"

# ================= SAVE FUNCTION =================
def save_data(sign):
    os.makedirs("streamlit_app", exist_ok=True)

    data = {
        "Sign": sign,
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    df = pd.DataFrame([data])
    file_path = "streamlit_app/sign_detection_results.csv"

    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

# ================= IMAGE UPLOAD =================
st.subheader("📷 Upload Image")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # 🔥 Improve brightness & contrast (fix grayscale issue)
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=30)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            sign = predict_sign(hand_landmarks)
            st.success(f"Detected Sign: {sign}")

            save_data(sign)
    else:
        st.warning("No hand detected")

    st.image(image, channels="BGR")

# ================= WEBCAM =================
st.subheader("🎥 Real-Time Detection")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    success, frame = cap.read()
    if not success:
        st.error("Camera not working")
        break

    frame = cv2.flip(frame, 1)

    # 🔥 Improve webcam visibility
    frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            sign = predict_sign(hand_landmarks)

            cv2.putText(frame, f"{sign}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            save_data(sign)

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()