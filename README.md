# 🤟 Real-Time Sign Language Translator

<p align="center">

### 🌟 SignEase — Real-Time Sign Language Translator

**Turning Hand Signs into Meaningful Text using AI & Computer Vision**

</p>

---

## 📌 Project Overview

This project is a **camera-based Sign Language Translator** that uses **Computer Vision** and **Machine Learning** to recognize selected hand signs and convert them into readable text.

The system detects hand landmarks using **MediaPipe** and uses a trained **K-Nearest Neighbors (KNN)** machine learning model to predict the sign.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔐 **Login System** | Simple user login |
| 📷 **Camera Capture** | Capture hand signs using the camera |
| ✋ **Hand Detection** | Detect hand landmarks using MediaPipe |
| 🧠 **AI Prediction** | Predict signs using Machine Learning |
| 💬 **Text Translation** | Convert recognized signs into text |
| 🏠 **Home Page** | Project introduction and instructions |
| 📖 **Supported Signs** | View currently supported signs |
| ℹ️ **About Project** | Information about the project |
| 🚪 **Logout** | Securely exit the application |

---

## 🔄 How It Works

```text
        📷 CAMERA
            ↓
     ✋ HAND DETECTION
            ↓
   📍 21 HAND LANDMARKS
            ↓
    ⚙️ FEATURE EXTRACTION
            ↓
      🧠 KNN MODEL
            ↓
     🔍 SIGN PREDICTION
            ↓
       💬 TEXT OUTPUT
       
       
       🛠️ Technologies Used

🐍 Python
Main programming language.

🎨 Streamlit
Used to create the web application interface.

👁️ OpenCV
Used for image and camera processing.

✋ MediaPipe
Used for detecting hand landmarks

🧠 Scikit-learn
Used for the KNN machine learning model.

🔢 NumPy
Used for numerical data processing.

💾 Pickle
Used to save and load the trained model.



📁 Project Structure

text
🤟 SignLanguageTranslator/
│
├── 📄 app.py
├── 📄 camera.py
├── 📄 collect_data.py
├── 📄 train_model.py
├── 📄 predict.py
├── 🧠 sign_model.pkl
├── ✋ hand_landmarker.task
│
├── 📂 data/
│   ├── hello.csv
│   ├── yes.csv
│   ├── no.csv
│   ├── stop.csv
│   └── thankyou.csv
│
├── 📄 .gitignore
└── 📖 README.md