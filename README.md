# 🌸 DermalScan – AI Facial Aging Detection System

**DermalScan** is an intelligent AI-based web application that analyzes facial images to detect signs of aging such as **wrinkles**, **dark spots**, and **puffy eyes**, while also identifying **clear skin**.  
The project combines **deep learning (InceptionV3)** and **computer vision** to provide accurate visual and tabular insights into facial aging patterns.

---

## 💡 Objective
To build a user-friendly system capable of detecting and analyzing visible facial aging indicators using an efficient convolutional neural network (CNN).

---

## 🧠 Key Features
- 🧍 Automatic **face detection** using OpenCV Haar Cascade.  
- 🧠 Deep learning-based classification of aging signs.  
- 📊 Downloadable **CSV report** containing predictions and confidence scores.  
- 🖼️ Annotated output images with bounding boxes and labels.  
- ⚙️ Intuitive Streamlit-based web interface.

---

## ⚙️ Technologies Used
| Category | Tools & Libraries |
|-----------|------------------|
| Deep Learning | TensorFlow, Keras (InceptionV3) |
| Web Framework | Streamlit |
| Image Processing | OpenCV, Pillow |
| Data Handling | Pandas, NumPy |

---

## 📁 Project Structure
```
DermalScan/
│
├── app.py                       # Streamlit UI application
├── backend.py                   # Model loading, preprocessing, and prediction logic
├── Documentation
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🧩 Installation Guide

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan.git
cd AI-DermalScan
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add the trained model
Download or place the model file `best_inceptionv3_model2.h5` in the project root directory.

### 4️⃣ Run the app
```bash
streamlit run app.py
```

Once running, open the local URL displayed in the terminal (usually `http://localhost:8501`).

---

## 🧬 Model Summary
- **Architecture:** InceptionV3 pretrained on ImageNet  
- **Input Size:** 224 × 224 × 3  
- **Output Classes:** 4  
  - `clear face`
  - `darkspots`
  - `puffy eyes`
  - `wrinkles`
- **Classification Output:** Condition label + confidence percentage  
- **Estimated Age Range:**  
  - Clear Face → 18–30 years  
  - Dark Spots → 30–40 years  
  - Puffy Eyes → 40–55 years  
  - Wrinkles → 56–75 years  

---

## 🧾 requirements.txt
```
streamlit==1.50.0
tensorflow==2.20.0
opencv-python==4.12.0
pandas==2.3.3
numpy==2.2.6
Pillow==11.3.0
```

---

## 🚀 Future Scope
- Integration of **real-time camera capture**
- Implementation of **skin tone & texture analysis**
- **Cloud-based API** deployment for faster inference
- Model optimization for **mobile and web inference**
- Inclusion of **dermatological recommendations**

---

## 🧑‍💻 Developer
**Battula Bhulakshmi**  
B.Tech in Computer Science and Engineering  
Rajiv Gandhi University of Knowledge Technologies, Ongole  


© 2025 DermalScan | AI-Powered Dermatological Intelligence
