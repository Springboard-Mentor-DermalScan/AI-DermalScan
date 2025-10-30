# 🧠 AI Dermal Scan

*From pixels to precision — intelligent skin analysis reimagined.*

---

## 📋 Overview

**AI Dermal Scan** is an intelligent deep learning–based application that analyzes facial skin images to detect **common dermal conditions** (like *dark spots, wrinkles, puffy eyes,* or *clear skin*) and **predicts the user’s estimated age**.

The project integrates a **Haar Cascade face detector** for region extraction, a **fine-tuned MobileNetV2 model** for condition classification, and a **Streamlit-based UI** for real-time predictions.

It demonstrates a complete **end-to-end AI pipeline** — from image preprocessing to model inference and interactive visualization.

---

## 🚀 Features

* 🧩 **AI-Powered Skin Analysis** — Predicts skin conditions using MobileNetV2.
* 👁️ **Face Detection** — Uses OpenCV Haarcascade to locate and crop facial regions.
* 📊 **Prediction Confidence** — Displays confidence score (%) for each prediction.
* 🎯 **Age Estimation** — Predicts an estimated age range based on the condition.
* 🖼️ **Bounding Boxes** — Draws detection boxes around faces with labels.
* 🧾 **Session Log** — Saves image name, prediction results, bounding box coordinates, and processing time.
* ⚡ **Optimized Inference** — End-to-end processing time ≤ 5 seconds per image.
* 💾 **Downloadable Results** — Users can download annotated output images.

---

## 🧩 System Architecture

```
Frontend (Streamlit)
       │
       ▼
Backend Pipeline
 - Image Preprocessing (Resize, Normalize)
 - Face Detection (Haarcascade)
 - Condition Prediction (MobileNetV2)
 - Age Estimation
       │
       ▼
Model Inference Results → Streamlit UI (Annotated Image + Log Table)
```

---

## 🧱 Module Breakdown

### **Module 1:** Data Collection & Preprocessing

* Collected and labeled facial images across 4 categories: *puffy_eyes, darkspots, wrinkles, clear_face*.
* Resized images to `(224 × 224)` for model input.

### **Module 2:** Model Training

* Fine-tuned **MobileNetV2** with transfer learning.
* Achieved **82.07% accuracy** and well-balanced validation metrics.

### **Module 3:** Face Detection

* Integrated **Haarcascade** classifier to localize faces.
* Extracted bounding boxes `(x, y, width, height)` for precise ROI prediction.

### **Module 4:** Model Inference Pipeline

* Designed a modular `backend.py` that loads the model, preprocesses inputs, and returns results.
* Outputs:

  * `condition`, `confidence (%)`, `predicted_age`, and bounding box coordinates.

### **Module 5:** Frontend Integration (Streamlit)

* Built `frontend.py` with an interactive user interface.
* Supports image upload, result display, and downloadable annotated output.

### **Module 6:** Logging & Performance

* Added CSV-based session logging (`session_logs.csv`).
* Ensured ≤ 5 seconds average processing time per image.

---

## ⚙️ Tech Stack

| Category             | Tools Used                                             |
| -------------------- | ------------------------------------------------------ |
| **Language**         | Python 3.10+                                           |
| **Frameworks**       | TensorFlow / Keras, Streamlit                          |
| **Image Processing** | OpenCV, Pillow, NumPy                                  |
| **Model**            | MobileNetV2 (Pretrained on ImageNet)                   |
| **Logging**          | Pandas                                                 |
| **Frontend UI**      | Streamlit                                              |
| **Deployment-ready** | Can be hosted on Streamlit Cloud or HuggingFace Spaces |

---

## 📁 Project Structure

```
AI_Dermal_Scan/
│
├── backend.py                # Model inference and preprocessing logic
├── frontend.py               # Streamlit UI for user interaction
├── MobileNetV2_best_model.h5 # Trained model file
├── session_logs.csv          # CSV log of predictions
├── default_face.jpg          # Default image (if no upload)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 💻 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Dermal-Scan.git
cd AI-Dermal-Scan
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run frontend.py
```

### 4. Upload an Image

Upload a **front-facing facial image** (PNG/JPG ≤10MB).
Results will show the detected condition, confidence, age, and bounding box.

---

## 📊 Sample Output

**Predicted Condition:** Darkspots
**Confidence:** 84.52%
**Predicted Age:** 34 yrs
**Bounding Box:** X=180, Y=240, W=110, H=130

*Visual output:*

```
🖼️ Original Image    |    🔍 Processed Output (with bounding box + label)
```

---

## 🧠 Model Details

* **Base Model:** MobileNetV2
* **Input Size:** 224×224×3
* **Optimizer:** Adam
* **Loss:** Categorical Crossentropy
* **Accuracy:** 82.07%
* **Classes:** `['puffy_eyes', 'darkspots', 'clear_face', 'wrinkles']`

---



---

Would you like me to include a **requirements.txt** file (with exact dependencies) for this README too?
