# Age and Gender Detection with Deep Learning  

## 📌 Project Overview  
This project is a **deep learning-based application** that detects a person's **age and gender** from images using a **Convolutional Neural Network (CNN)**. It includes **data preprocessing, model training, evaluation, and a user-friendly UI built with Streamlit**.  

---

## 🚀 Features  
- Pretrained CNN model for accurate predictions  
- Image preprocessing for better input quality  
- Streamlit UI for an interactive experience  
- Visualization of training and validation performance  

---

## 👴 Senior Citizen Detection (Webcam)  

This module performs **real-time detection** using a webcam.

- Detects age and gender in real-time  
- Marks individuals as **Senior Citizen** if age > 60  
- Stores results in a CSV file with timestamp  
- Uses OpenCV for webcam capture  

📁 Output file:  

streamlit_app/senior_detection_results.csv


---

## 🎤 Voice Age & Emotion Detection  

This module detects a person's **age, gender, and emotion from voice input** using audio feature extraction.

### 🔍 Features  
- Upload `.wav` or `.mp3` audio file  
- Detects **gender (only male allowed)**  
- Predicts **age from voice features**  
- Identifies **Senior Citizens (age > 60)**  
- Detects **emotion (Happy, Sad, Neutral, Angry)**  
- Stores results in CSV file with timestamp  

📁 Output file:  

streamlit_app/voice_detection_results.csv


---
## 📂 Project Structure  


age-gender-hair-detection/
│
├── dataset/
├── hair_dataset/
├── models/
├── notebook/
├── plots/
│ ├── voice result-1.png
│ ├── voice result-2.png
│ ├── voice result-3.png
│ ├── voice result-4.png
│
├── streamlit_app/
│ ├── app.py
│ ├── senior_detection.py
│ ├── voice_detection.py
│ ├── senior_detection_results.csv
│ ├── voice_detection_results.csv
│
├── README.md
├── requirements.txt


---

## 🛠 Installation & Usage  

1. **Clone the repository**  
```bash
git clone https://github.com/Raghuveer3339/age-gender-hair-detection.git
cd age-gender-hair-detection

Install dependencies

pip install -r requirements.txt

Run Image Detection App

streamlit run streamlit_app/app.py

Run Voice Detection App

python -m streamlit run streamlit_app/voice_detection.py
📌 Model & Optimization

This is a basic model, and there is scope for improvement:

Improve CNN architecture for better accuracy

Use data augmentation to enhance dataset quality

Apply transfer learning for better performance

Replace dummy logic in voice detection with real ML models

🤝 Contributions

Feel free to fork, improve, and contribute!
