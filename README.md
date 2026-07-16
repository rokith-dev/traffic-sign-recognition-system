# 🚦 AI-Powered Traffic Sign Recognition System

An intelligent Deep Learning-based Traffic Sign Recognition System that automatically detects and classifies traffic signs using **TensorFlow**, **MobileNetV2**, and **Streamlit**. This project demonstrates the application of Computer Vision and Transfer Learning for road safety and intelligent transportation systems.

---

## 📌 Project Overview

This project classifies traffic sign images into **58 different classes** using a trained MobileNetV2 model. It includes data preprocessing, model training, evaluation, prediction, an interactive Streamlit web application, and Grad-CAM explainability.

---

## ✨ Features

- 📂 Dataset Loading and Preprocessing
- 📊 Exploratory Data Analysis (EDA)
- 🧹 Dataset Quality Analysis
- 🧠 Custom CNN Model
- 🚀 MobileNetV2 Transfer Learning
- 📈 Model Training Pipeline
- 📋 Model Evaluation
- 🎯 Single Image Prediction
- 🏆 Top-5 Predictions
- 🌐 Streamlit Web Application
- 🔥 Grad-CAM Explainability
- 📷 Webcam Detection Support
- 📊 Model Performance Comparison

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Plotly
- Scikit-learn
- Streamlit
- Git & GitHub

---

## 📂 Project Structure

```text
traffic-sign-recognition-system/
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   └── config.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── prediction/
│   ├── explainability/
│   ├── webcam/
│   ├── visualization/
│   └── utils/
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- Total Images: **4,145**
- Classes: **58**
- Image Size: **224 × 224**
- Train/Test Split: **80/20**

---

## 🧠 Model

### Version 1
- Custom CNN
- Validation Accuracy: **91.56%**

### Version 2
- MobileNetV2 (Transfer Learning)
- Validation Accuracy: **99.76%**

---

## 📈 Model Performance

| Metric | CNN | MobileNetV2 |
|---------|-----|-------------|
| Validation Accuracy | 91.56% | **99.76%** |
| Precision | 91% | **99%** |
| Recall | 92% | **99%** |
| F1-Score | 90% | **99%** |

---

## 🚀 How to Run

### Clone the Repository

```bash
git clone https://github.com/your-username/traffic-sign-recognition-system.git
cd traffic-sign-recognition-system
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python -m src.training.trainer
```

### Evaluate the Model

```bash
python -m src.evaluation.evaluate_mobilenet
```

### Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 📷 Application Features

- Upload Traffic Sign Images
- Predict Traffic Sign Class
- Confidence Score
- Top-5 Predictions
- Grad-CAM Visualization
- Webcam Detection
- Interactive Streamlit Dashboard

---

## 🎯 Learning Outcomes

This project helped in learning:

- Computer Vision
- Deep Learning
- Transfer Learning
- Convolutional Neural Networks (CNN)
- TensorFlow & Keras
- Image Processing
- Model Evaluation
- Explainable AI (Grad-CAM)
- Streamlit Development
- Git & GitHub

---

## 🔮 Future Improvements

- EfficientNet Model
- Batch Image Prediction
- Prediction History
- PDF Prediction Reports
- Cloud Deployment
- REST API Integration
- Mobile Application

---

## 👨‍💻 Author

**Rokith**

B.Tech Artificial Intelligence and Data Science

Karunya Institute of Technology and Sciences

---

## ⭐ Acknowledgements

- TensorFlow
- Keras
- OpenCV
- Streamlit
- Scikit-learn
- MobileNetV2
- Kaggle Traffic Sign Dataset

---

## 📄 License

This project is developed for educational and research purposes.