import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import tensorflow as tf
from keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="Age & Gender Detector",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown(
    """
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 2rem;
}

.sub-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2563EB;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

.result-text {
    font-size: 1.5rem;
    font-weight: 500;
    padding: 0.75rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
}

.image-container {
    margin-bottom: 2rem;
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: rgba(237,242,247,0.5);
}

.app-footer {
    text-align: center;
    margin-top: 2rem;
    opacity: 0.7;
}
</style>
""",
    unsafe_allow_html=True,
)

# Load models
@st.cache_resource
def load_models():
    try:
        age_gender_model = load_model(r"D:\age_gender_code\models\Age_Sex_Detection.h5", compile=False)
        hair_model = load_model(r"D:\age_gender_code\models\hair_model.h5", compile=False)
        return age_gender_model, hair_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# Preprocess image for age/gender model
def preprocess_image(uploaded_image):
    if uploaded_image.mode != "RGB":
        uploaded_image = uploaded_image.convert("RGB")

    image = uploaded_image.resize((48, 48))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# Age and gender prediction
def predict_age_gender(model, image_array):

    predictions = model.predict(image_array)

    predicted_age = int(np.round(predictions[1][0]))

    gender_prob = predictions[0][0]
    predicted_gender = "Female" if gender_prob > 0.5 else "Male"

    gender_confidence = gender_prob if predicted_gender == "Female" else 1 - gender_prob

    return predicted_age, predicted_gender, float(gender_confidence)


# Main function
def main():

    st.markdown(
        '<div class="main-header">Age and Gender Detector</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model..."):
        age_gender_model, hair_model = load_models()

    if age_gender_model is None or hair_model is None:
        st.warning("Model files not found.")
        return

    st.markdown('<div class="sub-header">Upload Images</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose one or more images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Detect Age & Gender", key="detect_button"):

        with st.spinner("Analyzing images..."):

            for i, uploaded_file in enumerate(uploaded_files):

                with st.container():

                    st.markdown('<div class="image-container">', unsafe_allow_html=True)

                    st.markdown(f"<h3>Image {i+1}</h3>", unsafe_allow_html=True)

                    col1, col2 = st.columns([1, 1])

                    image = Image.open(uploaded_file)

                    col1.image(
    image,
    caption=f"Image {i+1}: {uploaded_file.name}",
    width="stretch",
)

                    processed_image = preprocess_image(image)

                    age, gender, confidence = predict_age_gender(
                        age_gender_model, processed_image
                    )

                    # Hair prediction
                    hair_img = image.resize((128,128))
                    hair_img = np.array(hair_img) / 255.0
                    hair_img = np.expand_dims(hair_img, axis=0)

                    hair_pred = hair_model.predict(hair_img)

                    hair = "Long" if hair_pred[0][0] > 0.5 else "Short"

                    # Internship rule
                    if 20 <= age <= 30:
                        if hair == "Long":
                            final_gender = "Female"
                        else:
                            final_gender = "Male"
                    else:
                        final_gender = gender

                    col2.markdown(
                        '<div class="sub-header">Results:</div>',
                        unsafe_allow_html=True,
                    )

                    col2.markdown(
                        f'<div class="result-text">Age: {age}</div>',
                        unsafe_allow_html=True,
                    )

                    col2.markdown(
                        f'<div class="result-text">Hair Length: {hair}</div>',
                        unsafe_allow_html=True,
                    )

                    col2.markdown(
                        f'<div class="result-text">Final Gender: {final_gender}</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("</div>", unsafe_allow_html=True)

                    if i < len(uploaded_files) - 1:
                        st.markdown("<hr>", unsafe_allow_html=True)

    

    st.markdown(
        '<div class="app-footer">Powered by NULLCLASS🧑‍💻</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()