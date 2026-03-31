import streamlit as st

# MUST BE FIRST
st.set_page_config(page_title="AI Detection App", layout="centered")

from PIL import Image
import numpy as np
import cv2
import joblib
import mediapipe as mp
from datetime import datetime
from keras.models import load_model

# ============================
# ✋ SIGN LANGUAGE DETECTION
# ============================

st.header("✋ Sign Language Detection (ML)")

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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# ============================
# 💇 HAIR MODEL (ML)
# ============================

st.header("💇 Hair-based Gender Detection")

hair_model = joblib.load("hair_model.pkl")

age_input = st.slider("Select Age", 1, 100, 25)
hair = st.selectbox("Hair Length", ["Short", "Long"])

hair_value = 1 if hair == "Long" else 0

if st.button("Predict Gender"):
    prediction = hair_model.predict([[age_input, hair_value]])[0]
    st.success(f"Predicted Gender: {prediction}")

# ============================
# 👤 AGE & GENDER (CNN)
# ============================

st.header("👤 Age & Gender Detection")

@st.cache_resource
def load_models():
    try:
        age_model = load_model("Age_Sex_Detection.h5", compile=False)
        return age_model
    except:
        return None

model = load_models()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    img = image.resize((48, 48))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)

    age = int(np.round(preds[1][0]))
    gender = "Female" if preds[0][0] > 0.5 else "Male"

    st.success(f"Age: {age}")
    st.success(f"Gender: {gender}")

# ============================
# FOOTER
# ============================

st.markdown("### 🚀 Powered by Your ML Project")