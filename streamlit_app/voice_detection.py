import streamlit as st
import librosa
import numpy as np
import pandas as pd
from datetime import datetime
import os

st.title("🎤 Voice Age & Emotion Detection")

# Upload audio
audio_file = st.file_uploader("Upload Voice", type=["wav", "mp3"])


# ===================== FEATURE EXTRACTION =====================
def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)


# ===================== PREDICTION FUNCTIONS =====================
def predict_gender(features):
    # Dummy logic
    if np.mean(features) > -20:
        return "Male"
    else:
        return "Female"


def predict_age(features):
    mean_val = np.mean(features)

    if mean_val > -20:
        return np.random.randint(45, 75)  # older voices
    else:
        return np.random.randint(20, 45)  # younger voices


def predict_emotion(features):
    emotions = ["Happy", "Sad", "Neutral", "Angry"]
    return np.random.choice(emotions)


# ===================== MAIN LOGIC =====================
if audio_file is not None:

    st.write("Processing audio...")

    features = extract_features(audio_file)

    gender = predict_gender(features)

    if gender == "Female":
        st.error("❌ Upload male voice only")

    else:
        age = predict_age(features)

        st.success(f"Gender: {gender}")
        st.info(f"Age: {age}")

        if age > 60:
            emotion = predict_emotion(features)
            st.warning("Senior Citizen Detected")
            st.success(f"Emotion: {emotion}")
        else:
            emotion = "N/A"
            st.info("Emotion detection not required")

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

        st.write("Saving to:", file_path)  # debug

        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)