# 🧠 Age, Gender, Voice & Sign Language Detection System

## 📌 Project Overview  
This project is a **multi-modal AI-based system** that performs detection using **image, voice, and hand gestures**. It combines **computer vision, audio processing, and machine learning concepts** into a single interactive application built using **Streamlit**.

The system includes:
- Age & Gender Detection from images  
- Senior Citizen Detection (real-time webcam)  
- Voice-based Age & Emotion Detection  
- Sign Language Detection using hand gestures  

---

## 🚀 Features  

### 👤 Age & Gender Detection
- Detects age and gender from facial images  
- Uses a **pretrained CNN model**  
- Includes image preprocessing for better results  

---

### 👴 Senior Citizen Detection
- Real-time detection using webcam  
- Marks individuals as **Senior Citizen (age > 60)**  
- Stores results with timestamp  

📁 Output file: `streamlit_app/senior_detection_results.csv`

---

### 🎤 Voice Age & Emotion Detection
- Accepts audio input (`.wav`, `.mp3`)  
- Extracts features using **MFCC (librosa)**  
- Detects:
  - Gender (male only allowed)
  - Age (estimated)
  - Emotion (for senior citizens)

📁 Output file: `streamlit_app/voice_detection_results.csv`

---

### ✋ Sign Language Detection
- Detects hand gestures using **MediaPipe (ML-based framework)**  
- Supports:
  - Image upload detection  
  - Real-time webcam detection  

- Uses hand landmark extraction  
- Applies rule-based logic to classify gestures:
  - Hello 👋  
  - Yes 👍  
  - No ❌  
  - Thanks 🙏  

📁 Output file: `streamlit_app/sign_detection_results.csv`

---

## 🧠 Machine Learning Concepts Used  

- Convolutional Neural Networks (CNN)  
- Feature extraction (MFCC for audio)  
- Hand landmark detection using MediaPipe  
- Rule-based classification  
- Data logging using CSV  

---

## 🖥️ Tech Stack  

- Python  
- Streamlit  
- OpenCV  
- NumPy, Pandas  
- Librosa  
- MediaPipe  

---

## 📂 Project Structure  


AGE_GENDER_CODE/
│
├── dataset/
├── hair_dataset/
├── models/
├── notebook/
├── plots/ # Screenshots
│
├── streamlit_app/
│ ├── app.py
│ ├── senior_detection.py
│ ├── voice_detection.py
│ ├── sign_language.py
│ ├── senior_detection_results.csv
│ ├── voice_detection_results.csv
│ ├── sign_detection_results.csv
│
├── requirements.txt
├── README.md


---

## 🛠 Installation & Usage  

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Raghuveer3339/age-gender-hair-detection.git
cd AGE_GENDER_CODE
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run applications
▶ Age & Gender Detection
streamlit run streamlit_app/app.py
▶ Voice Detection
streamlit run streamlit_app/voice_detection.py
▶ Sign Language Detection
streamlit run streamlit_app/sign_language.py
📊 Results

Outputs are stored in CSV files

Screenshots are stored in the plots/ folder

Real-time predictions are shown in Streamlit UI

📌 Future Improvements

Train a custom deep learning model for gesture recognition

Improve accuracy using larger datasets

Add advanced classifiers (SVM, CNN, etc.)

Enhance real-time performance

🤝 Contributions

Feel free to fork, improve, and contribute to this project.

📬 Conclusion

This project demonstrates the integration of computer vision, audio processing, and interactive UI, providing a strong foundation for real-world AI applications.