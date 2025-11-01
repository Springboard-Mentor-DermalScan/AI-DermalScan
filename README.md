🧠 DermalScan – AI Facial Skin Aging Detection

From pixels to precision — AI that understands your skin.

🌟 Overview

DermalScan is a deep learning–powered AI application that analyzes your face image to detect skin conditions (like wrinkles, dark spots, puffy eyes, or clear face) and predict your approximate age.

Using OpenCV’s DNN-based face detector and a fine-tuned MobileNetV2 model, it processes facial images in real-time — delivering accurate predictions with confidence scores.

The complete system runs on Streamlit, offering a smooth and interactive web experience for everyone.

🚀 Key Features
Feature	Description
🧠 AI Skin Detection	Detects multiple facial skin issues using a trained MobileNetV2 model.
👁️ Face Detection	Uses OpenCV DNN for precise face localization.
🎂 Age Estimation	Predicts the approximate age for each detected face.
📊 Confidence Score	Displays model confidence for every prediction.
🧍‍♀️ Multi-Face Support	Detects and labels more than one face per image.
⚡ Fast & Lightweight	Inference takes less than 5 seconds per image.
💻 Web UI (Streamlit)	Simple drag-and-drop interface for all users.
⚙️ Tech Stack
Category	Tools / Libraries
Programming Language	Python 3.10+
Deep Learning	TensorFlow, Keras
Face Detection	OpenCV DNN
Frontend (Web UI)	Streamlit
Data Handling	NumPy, Pandas
Visualization	Matplotlib, Pillow
🧩 Face Detection Setup

This project uses OpenCV’s DNN-based face detector.
Download these two required files before running the app:

1️⃣ deploy.prototxt
2️⃣ res10_300x300_ssd_iter_140000.caffemodel

📁 Setup Instructions:

Download both files.

Place them in your project root folder (same location as backend.py).

The app will automatically detect and use them during runtime.

📁 Project Structure
DermalScan/
│
├── app.py                       
├── backend.py                   
├── DermalSkin_MobileNetV2_Finetuned.h5  
├── requirements.txt              
├── Documents/
│   ├── Infosys Documentation   
├── results
├── uploads
└── README.md                    

💻 How to Run the Project
Step 1️⃣: Clone the Repository
git clone https://github.com/rasoolbaig-ai/DermalScan.git
cd DermalScan

Step 2️⃣: Install Dependencies
pip install -r requirements.txt

Step 3️⃣: Add Face Detector Files

Place the following in your project folder:

deploy.prototxt

res10_300x300_ssd_iter_140000.caffemodel

Step 4️⃣: Run the Application
streamlit run app.py

Step 5️⃣: Upload an Image

Upload a clear front-facing facial image (JPG/PNG ≤ 10MB).

Wait for the model to predict skin condition, confidence score, and estimated age.

🧠 Model Summary
Parameter	Description
Model Used	MobileNetV2 (Fine-Tuned)
Input Shape	224 × 224 × 3
Optimizer	Adam
Loss Function	Categorical Crossentropy
Accuracy Achieved	99.07%
Predicted Classes	Clear Face, Dark Spots, Puffy Eyes, Wrinkles
👨‍💻 Developed By

Rasool Baig
🎓 B.Tech – Computer Science & Engineering
SESHADRI RAO GUDLAVALLERU ENGINEERING COLLEGE
🏫 Jawaharlal Nehru Technological University, Kakinada (JNTUK)
🤝 Infosys Springboard Virtual Internship Project – DermalSkin Analyzer
