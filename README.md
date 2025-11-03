# 🧠 DermalScan AI — Skin Condition Classifier

### 🔍 Overview

**DermalScan AI** is a deep learning–based dermatological analysis system designed to automatically classify **facial skin conditions** — including *Clear Face*, *Dark Spots*, *Puffy Eyes*, and *Wrinkles*.
The system uses **DenseNet121** with transfer learning and integrates **OpenCV** for facial detection, enabling real-time prediction through an interactive **Streamlit web interface**.
All backend processing — including preprocessing, face detection, and model inference — is implemented **within a single Streamlit app (`app.py`)** for simplified deployment and easy execution.

---

## 🎯 Objectives

* Build a robust AI model capable of detecting and classifying multiple facial skin conditions.
* Provide real-time classification and annotated visualization through a clean web interface.
* Achieve **>95 % training and validation accuracy** using an optimized deep learning model.
* Ensure faster and lightweight performance using **TensorFlow Lite optimization**.

---

## ⚙️ Technologies Used

| Category             | Libraries                               |
| :------------------- | :-------------------------------------- |
| **Data Handling**    | `os`, `glob`, `numpy`, `pandas`         |
| **Visualization**    | `matplotlib`, `seaborn`, `PIL`          |
| **Image Processing** | `opencv-python`                         |
| **Model Training**   | `tensorflow`, `keras`, `scikit-learn`   |
| **Web Interface**    | `streamlit`                             |
| **Optimization**     | `tensorflow-lite`, `psutil`, `datetime` |

---

## 🏗️ Project Structure

```
DermalScanAI/
│
├── app.py                 # Streamlit app (includes backend + UI)
├── Documentation/         # Project documentation & reports
├── model/                 # Trained and optimized model files
├── results/               # Evaluation graphs, confusion matrices
├── README.md              # Project overview
└── requirements.txt       # Library dependencies
```

> 🧩 *Note:* All data processing, model loading, and prediction logic are included directly inside **`app.py`** — no external backend files are required.

---

## 🚀 Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Springboard-Mentor/DermalScan-AI.git
cd DermalScan-AI
```

### 2️⃣ Setup Environment

Ensure Python 3.10+ is installed. Then install all the required libraries listed in the **requirements.txt** file.

### 3️⃣ Add the Trained Model

Copy your trained model file (e.g., `best_model.h5` or `model.tflite`) into the project folder.

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

Once executed, the local server (typically `http://localhost:8501`) will open the **DermalScan AI** web interface.

---

## 🧩 Model Details

* **Architecture:** DenseNet121 (Transfer Learning, pretrained on ImageNet)
* **Optimizer:** Adam (lr = 0.001)
* **Loss Function:** Categorical Crossentropy
* **Callbacks Used:** EarlyStopping · ReduceLROnPlateau · ModelCheckpoint
* **Input Size:** 224 × 224 × 3
* **Output Labels:** Clear Face | Dark Spots | Puffy Eyes | Wrinkles

---

## 📊 Key Results

| Metric                                 | Performance                 |
| :------------------------------------- | :-------------------------- |
| **Training Accuracy**                  | 90.9 %                      |
| **Validation Accuracy**                | 84.1 %                      |
| **Fine-Tuned Accuracy**                | > 95 % (after optimization) |
| **Detection Accuracy (Pre-Validated)** | 100 %                       |
| **Inference Speed**                    | < 1.5 sec per image         |

---

## 🧠 Step-by-Step Development Summary

1. **Dataset Preparation:** Cleaned and organized the dataset into four classes, verifying dimensions and image quality.
2. **Preprocessing:** Resized all images to 224×224 and normalized pixel intensity.
3. **Augmentation:** Applied rotation, flipping, and brightness variation to improve generalization.
4. **Model Training:** Used DenseNet121 with transfer learning and three callbacks to achieve high validation accuracy.
5. **Performance Evaluation:** Generated confusion matrices, accuracy/loss plots, and classification reports.
6. **Prediction Pipeline:** Integrated OpenCV Haar-Cascade for automatic face detection within app.py.
7. **Streamlit Deployment:** Developed an end-to-end Streamlit web app that handles image upload, face detection, model inference, and result display seamlessly.

---

## 🖼️ Visualization and UI Snapshots

### 📊 Model Visualization

* Dataset distribution charts
* Accuracy vs Loss performance curves
* Confusion matrix heatmaps
* Prediction grids with detected faces

### 🖥️ Streamlit Web Interface

Below are sample interface screenshots of the deployed **DermalScan AI** web app 👇

---
<img width="1920" height="1240" alt="screencapture-localhost-8501-2025-11-02-11_59_02" src="https://github.com/user-attachments/assets/bea7988a-2f97-4d4b-a353-b077371c8d54" />
<img width="1920" height="2287" alt="screencapture-localhost-8501-2025-11-02-14_03_28" src="https://github.com/user-attachments/assets/b3715a20-0a76-4e02-afad-1b700ea1c005" />


## 📦 Output Files

| File Name               | Description                                  |
| :---------------------- | :------------------------------------------- |
| `best_model.h5`         | Trained DenseNet121 model weights            |
| `model.tflite`          | Optimized lightweight version for inference  |
| `detection_results.png` | Annotated predictions with detected faces    |
| `confusion_matrix.png`  | Model accuracy and performance visualization |
| `app.py`                | Integrated Streamlit + Backend code          |
| `prediction_logs.csv`   | Logs of predictions and confidence scores    |

---

## 🧩 requirements.txt

All dependencies for running **DermalScan AI** are listed below with exact version numbers for reproducibility.

```
# Core Frameworks
tensorflow==2.15.0
keras==2.15.0

# Web Interface
streamlit==1.39.0

# Image Processing
opencv-python==4.9.0.80
Pillow==10.3.0

# Data Handling & Analysis
numpy==1.26.4
pandas==2.2.2

# Visualization
matplotlib==3.8.4
seaborn==0.13.2

# Model Optimization & Utilities
tensorflow-lite==2.15.0
psutil==5.9.8

# Additional Tools
scikit-learn==1.5.1
tqdm==4.66.4
```

---

## 👨‍💻 Developer

**Boini Pramod Kumar**

📅 Project Year: 2025

💬 *“DermalScan AI combines deep learning and dermatology to make intelligent skin analysis accessible to everyone.”*

---

✅ **End of Project README**

---
