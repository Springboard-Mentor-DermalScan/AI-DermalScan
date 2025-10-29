DermalScan: AI Facial Skin Aging Detection App
🧠 Overview

DermalScan is an AI-powered application that analyzes facial images to predict skin type and age indicators using a deep learning model trained with EfficientNetB0 and image processing through OpenCV.
It helps users visualize aging signs and understand their skin health using annotated predictions and class probabilities.

🚀 Features

📸 Upload single or multiple facial images

🧩 Detect and visualize skin type, age, and facial features

🧠 Pre-trained deep learning model (EfficientNetB0) for classification

🎯 Annotated visualization with bounding boxes and class confidence

💾 Export results as annotated images and CSV files

🧍‍♀️ Web UI built with Streamlit for smooth interaction

🧰 Tech Stack
Area	Tools / Libraries
Image Ops	OpenCV, NumPy, Haarcascade
Model	TensorFlow/Keras, EfficientNetB0
Frontend	Streamlit
Backend	Python, Modularized Inference
Evaluation	Accuracy, Loss, Confusion Matrix
Exporting	CSV, Annotated Image, PDF (optional)
📂 Repository Structure
AI-DermalScan/
│
├── app.py                    # Streamlit Frontend (UI)
├── backend.py                # Backend Logic & Model Integration
├── Shivani_AI_DermalScan.ipynb  # Model Training / Testing Notebook
│
├── Documents/
│   ├── AI_DermalScan.pdf
│   ├── DermalScan_Documentation.docx
│   └── Requirement.txt.txt
│
├── uploads/                  # Sample Test Images
├── results/                  # Annotated Images & CSV Exports
│
├── LICENSE                   # License Information
└── README.md                 # Project Overview

⚙️ Installation

1️⃣ Clone this repository

git clone https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan.git
cd AI-DermalScan


2️⃣ Install dependencies

pip install -r Documents/Requirement.txt.txt


3️⃣ Run the application

streamlit run app.py

📈 Milestone Summary

Milestone 3 (Modules 5–6): Integrated Frontend & Backend with Streamlit UI.

Enabled image upload and real-time preview.

Displayed multiple annotated predictions with class confidence.

Added result export (CSV + annotated image).

👩‍💻 Intern
Akkala Shivani Reddy
🎓 Malla Reddy Engineering College
🤝 Infosys Springboard Mentor – Praveen
