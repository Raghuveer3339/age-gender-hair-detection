import streamlit as st
import librosa
import numpy as np
import pandas as pd
from datetime import datetime
import os
import joblib

st.title("🎤 Voice Age & Emotion Detection (ML Powered)")

# ===================== LOAD MODEL =====================
model = joblib.load("voice_model.pkl")

# Upload audio
audio_file = st.file_uploader("Upload Voice", type=["wav", "mp3"])


# ===================== FEATURE EXTRACTION =====================
def extract_features(file):
    y, sr = librosa.load(file, sr=None, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0)


# ===================== BASIC LOGIC =====================
def predict_gender(features):
    if np.mean(features) > -20:
        return "Male"
    else:
        return "Female"


def predict_age(features):
    mean_val = np.mean(features)

    if mean_val > -15:
        return 65
    elif mean_val > -25:
        return 50
    else:
        return 30


# ===================== MAIN =====================
if audio_file is not None:

    st.write("🔄 Processing audio...")

    features = extract_features(audio_file)

    gender = predict_gender(features)

    if gender == "Female":
        st.error("❌ Upload male voice only")

    else:
        age = predict_age(features)

        st.success(f"Gender: {gender}")
        st.info(f"Estimated Age: {age}")

        # ===================== ML PREDICTION =====================
        if age > 60:
            emotion = model.predict([features])[0]

            st.warning("👴 Senior Citizen Detected")
            st.success(f"Predicted Emotion: {emotion}")
        else:
            emotion = "N/A"
            st.info("Emotion detection not required (Age ≤ 60)")

        # ===================== SAVE DATA =====================
        os.makedirs("streamlit_app", exist_ok=True)

        data = {
            "Gender": gender,
            "Age": age,
            "Emotion": emotion,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        df = pd.DataFrame([data])

        file_path = "streamlit_app/voice_detection_results.csv"

        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

        st.success("✅ Data saved successfully!")