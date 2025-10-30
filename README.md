# 💎 AI DermalScan – Facial Aging Sign Detection

**AI DermalScan** is a deep learning–based web application that detects **facial aging signs** such as *wrinkles*, *dark spots*, and *puffy eyes*, while also estimating the **approximate age** of the person using a fine-tuned **DenseNet121** model.  
The app is built with **Streamlit** for an intuitive and modern user interface.

---

## 🚀 Features

✅ Real-time **Face Detection** using OpenCV Haar Cascade  
✅ Accurate **Aging Sign Classification** (`clear face`, `darkspots`, `puffy eyes`, `wrinkles`)  
✅ Logical **Age Estimation** based on detected facial condition  
✅ **Streamlit-based UI** with dark gradient theme  
✅ **Downloadable Results** – annotated image + prediction CSV  
✅ **Automatic CSV Logging** of all predictions  
✅ Average **processing time under 5 seconds**

---

## ⚙️ Project Structure

```bash
AI_DermalScan/
│
├── Documentation/
│   └── Naman Kapoor(AI_DermalScan) Documentation.pdf
│
├── Naman Kapoor(AI_DermalScan).ipynb   # Model training & experimentation notebook
│
├── app.py                     # Streamlit frontend 
├── backend.py                 # Backend pipeline 
│
├── haarcascade_frontalface_default.xml  # Face detection model
│
├── requirements.txt           # All required dependencies
├── prediction_log.csv         # (Auto-generated) prediction records
└── README.md                  # You are here
```
---

## 🧠 Model Overview

| Parameter | Details |
|------------|----------|
| **Base Architecture** | DenseNet121 (Transfer Learning) |
| **Input Size** | 224 × 224 pixels |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy |
| **Framework** | TensorFlow / Keras |
| **Augmentation** | Rotation, Zoom, Flip, Shift |

---

## 🧩 Tech Stack

| Layer | Technology |
|--------|-------------|
| **Frontend** | Streamlit |
| **Backend** | TensorFlow / Keras |
| **Detection** | OpenCV Haar Cascade |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Streamlit |
| **Logging** | CSV via Pandas |

---

## 💻 Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan.git
cd AI-DermalScan
git checkout Naman
```
### 2️⃣ Create a Virtual Environment(Optional)
```bash
python -m venv venv
venv\Scripts\activate      # For Windows
source venv/bin/activate   # For macOS / Linux
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 📊 Output Example

**Uploaded Image → Annotated Result**

✅ **Detected Sign:** Wrinkles  
📊 **Confidence:** 92.5%  
🎯 **Estimated Age:** 68 years  
⚡ **Processing Time:** 3.42 seconds  

The model successfully identifies visible facial aging signs and overlays bounding boxes with predicted class, confidence score, and estimated age.

---

## 👥 Contributors

- **Intern:** Naman Kapoor  
- **Mentor:** Praveen (Infosys Springboard)

