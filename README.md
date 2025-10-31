# 🧠 AI Dermal Scan

*From pixels to precision — Smart AI that understands your skin.*

---

## 🌟 Overview

**AI Dermal Scan** is a simple yet powerful AI project that can **analyze your face image** and predict:

* Your **skin condition** (like *dark spots, wrinkles, puffy eyes,* or *clear skin*), and
* Your **estimated age**.

The system uses **OpenCV’s deep learning face detector** to find your face in the image and a **MobileNetV2 model** (trained with deep learning) to identify your skin condition.

The complete project runs through **Streamlit**, which makes it easy for anyone to use through a clean web interface.

---

## 🚀 Key Features

✅ **AI Skin Detection** – Detects skin issues using a trained deep learning model.
✅ **Face Detection** – Uses **OpenCV DNN face detector** for accurate face localization.
✅ **Age Prediction** – Estimates your approximate age from the image.
✅ **Confidence Score** – Shows how confident the AI is in its prediction.
✅ **Multiple Face Detection** – Can detect and label more than one face in a photo.
✅ **Fast & Lightweight** – Gives results in just a few seconds.
✅ **Easy to Use** – Works through a friendly web app built in Streamlit.

---

## ⚙️ Tools and Technologies

| Category                 | Tools Used         |
| ------------------------ | ------------------ |
| **Programming Language** | Python 3.10+       |
| **Deep Learning**        | TensorFlow, Keras  |
| **Face Detection**       | OpenCV DNN         |
| **Frontend**             | Streamlit          |
| **Data Handling**        | NumPy, Pandas      |
| **Visualization**        | Matplotlib, Pillow |

---

## 🧩 Face Detection Setup (Required Files)

Since this project uses **OpenCV’s DNN-based face detector**, you need to download two files before running the app:

1️⃣ [deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
2️⃣ [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830)

**Steps to set up:**

* Download both files.
* Place them in your project folder (same folder as `backend.py`).
* That’s it! The app will automatically load them when it runs.

---

## 📁 Project Structure

```
AI_Dermal_Scan/
│
├── backend.py                # Handles AI model and face detection
├── frontend.py               # Streamlit web app for user interface
├── MobileNetV2_best_model.h5 # Trained deep learning model
├── requirements.txt          # Required libraries
└── README.md                 # Project guide (this file)
```

---

## 💻 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Dermal-Scan.git
cd AI-Dermal-Scan
```

### Step 2: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 3: Add the Face Detector Files

Place these files in your project folder:

* `deploy.prototxt`
* `res10_300x300_ssd_iter_140000.caffemodel`

### Step 4: Run the App

```bash
streamlit run frontend.py
```

### Step 5: Upload an Image

* Upload a **clear front-facing image** (JPG/PNG ≤10MB).
* Wait for a few seconds to see the predicted **skin condition**, **confidence**, and **estimated age**.

---

## 🧠 Model Details

| Parameter             | Description                                 |
| --------------------- | ------------------------------------------- |
| **Model Used**        | MobileNetV2                                 |
| **Input Image Size**  | 224 × 224 × 3                               |
| **Optimizer**         | Adam                                        |
| **Loss Function**     | Categorical Crossentropy                    |
| **Accuracy Achieved** | 82.07%                                      |
| **Classes**           | puffy_eyes, darkspots, clear_face, wrinkles |




