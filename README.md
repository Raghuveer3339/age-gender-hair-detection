# 🚀 Sign Language Detection & Hair-Based Gender Prediction (ML Project)

## 📌 Project Overview

This project is a **Machine Learning-based Multi-Feature System** developed using Python and Streamlit.
It focuses on **real-time sign language detection** along with an additional **hair-based gender prediction module**.

The system supports both:

* 📷 **Real-time webcam detection**
* 🖼️ **Image upload detection**

---

## 🎯 Features

### ✋ 1. Sign Language Detection (Main Task)

* Uses **MediaPipe Hands** for hand tracking
* Extracts **hand landmark features (x, y coordinates)**
* Uses a **Machine Learning model (Decision Tree)** for prediction
* Recognizes gestures:

  * Hello 👋
  * Yes 👍
  * No ❌
  * Thanks 🙏

#### ✅ Functionalities:

* Real-time webcam detection
* Image upload detection
* Displays prediction on screen

---

### ⏰ 2. Time Restriction

* The system only works between:

  * **6 PM to 10 PM**
* Ensures controlled execution as per task requirement

---

### 💇 3. Hair-Based Gender Detection (Bonus Feature)

* Uses a **Decision Tree ML model**
* Inputs:

  * Age
  * Hair Length (Short / Long)
* Predicts:

  * Gender

---

## 🧠 Technologies Used

* Python 🐍
* Streamlit (GUI)
* OpenCV (Computer Vision)
* MediaPipe (Hand Tracking)
* Scikit-learn (ML Models)
* NumPy

---

## 📂 Project Structure

```
age-gender-hair-detection/
│
├── streamlit_app/
│   ├── app.py
│   ├── sign_model.pkl
│   ├── hair_model.pkl
│
├── README.md
├── requirements.txt
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository

```
git clone https://github.com/Raghuveer3339/age-gender-hair-detection.git
cd age-gender-hair-detection/streamlit_app
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Run the Application

```
python -m streamlit run app.py
```

---

## 📊 Output

* Real-time sign detection using webcam
* Image-based sign detection
* Hair-based gender prediction
* Interactive GUI using Streamlit

---

## 🎓 Key Highlights

* Real-time computer vision system
* Integration of **MediaPipe + Machine Learning**
* Multi-input system (Webcam + Image Upload)
* User-friendly UI using Streamlit
* Follows task constraints (time restriction)

---

## 📌 Future Improvements

* Add more gesture classes
* Improve model accuracy with larger dataset
* Deploy application online (Streamlit Cloud)
* Add voice feedback system

---

## 👨‍💻 Author

**Raghuveer Singh**

---

## ⭐ Acknowledgement

This project was developed as part of an **AI/ML Internship**, focusing on real-world implementation of machine learning and computer vision concepts.

---
