import streamlit as st
import numpy as np
import cv2
import joblib
import mediapipe as mp
from datetime import datetime

# MUST BE FIRST
st.set_page_config(page_title="ML App", layout="centered")

# ---------------- SIGN LANGUAGE ----------------
st.header("✋ Sign Language Detection (ML)")

# Time restriction
current_hour = datetime.now().hour
if not (18 <= current_hour <= 22):
    st.error("❌ Works only between 6 PM to 10 PM")
else:
    if st.button("Start Sign Detection"):

        model = joblib.load("sign_model.pkl")

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        mp_draw = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    features = []
                    for lm in hand_landmarks.landmark:
                        features.extend([lm.x, lm.y])

                    prediction = model.predict([features])[0]

                    cv2.putText(frame, f"Sign: {prediction}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            stframe.image(frame, channels="BGR")

        cap.release()

    # -------- IMAGE UPLOAD --------
    st.subheader("📸 Detect Sign from Image")

    uploaded_file = st.file_uploader("Upload Hand Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        model = joblib.load("sign_model.pkl")

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        mp_draw = mp.solutions.drawing_utils

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y])

                prediction = model.predict([features])[0]

                cv2.putText(frame, f"Sign: {prediction}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            st.image(frame, channels="BGR")
        else:
            st.warning("No hand detected")


# ---------------- HAIR MODEL ----------------
st.header("💇 Hair Based Gender Detection")

hair_model = joblib.load("hair_model.pkl")

age_input = st.slider("Select Age", 1, 100, 25)
hair = st.selectbox("Hair Length", ["Short", "Long"])

hair_value = 1 if hair == "Long" else 0

if st.button("Predict Gender"):
    prediction = hair_model.predict([[age_input, hair_value]])[0]
    st.success(f"Predicted Gender: {prediction}")