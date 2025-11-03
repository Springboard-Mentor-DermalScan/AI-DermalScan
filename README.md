🧠 DermalScan: AI Facial Skin Aging Detection App
📌 Overview

DermalScan AI is a deep learning–based web application designed to analyze facial skin and detect signs of aging such as wrinkles, dark spots, puffy eyes, and clear skin.
Developed using TensorFlow (MobileNetV2) and Streamlit, the system allows users to upload facial images and receive annotated predictions, estimated age, and confidence scores through an intuitive web interface.

🎯 Project Objectives

Detect and classify visible facial aging indicators using deep learning.

Identify key categories: wrinkles, dark_spots, puffy_eyes, clear_face.

Estimate approximate age through categorical and regression-based approaches.

Provide annotated outputs with confidence and bounding boxes.

Maintain detailed session logs and enable export options for analysis.

⚙️ Tech Stack
Component	Tools / Libraries
Frontend	Streamlit, HTML, CSS
Backend / Model	TensorFlow / Keras (MobileNetV2)
Image Processing	OpenCV, NumPy, PIL
Logging & Export	Pandas, CSV
Visualization	Matplotlib, Streamlit Components
🧩 Features Implemented

✅ AI Model Integration:
Integrated MobileNetV2 for efficient multi-class facial skin condition detection with low latency and high accuracy.

✅ Frontend Enhancements:
Built an interactive Streamlit-based futuristic UI featuring gradient backgrounds, glow effects, and neon-styled data tables.

✅ Image Upload & Annotation:
Enables users to upload facial images and view real-time AI inference results with annotated bounding boxes and predicted features.

✅ Session Log:
Tracks every analysis with detailed attributes including:

Image name

Predicted skin condition

Confidence percentage

Estimated age

Processing time

Bounding box coordinates

✅ Export Options:

Download annotated output images directly.

Export session logs to CSV for documentation or further evaluation.

🧪 Testing Summary

Extensively tested with diverse facial images, including real-time captures and multi-face photographs.

Verified the accuracy of bounding boxes, feature predictions, and estimated age consistency.

Optimized table and font colors for clear visibility in dark-themed UI.

Confirmed export functionality for annotated results and session logs.

🚀 How to Run
1️⃣ Clone Repository
git clone https://github.com/<your-username>/DermalScan-AI.git
cd DermalScan-AI

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Add Your Model

Place your trained MobileNetV2 model file in the project directory:

C:\MACHINE LEARNING\Desktop\AI_DERMAL_SCAN\dermal_model.h5


Or rename it to:

model.h5

4️⃣ Run the Application
streamlit run app.py

🖼️ Output Preview

Annotated Prediction Example:

Bounding box highlighting detected face region

Top predicted condition with confidence

Estimated age displayed beside label

Session Log Example:

Image Name	Predicted Condition	Confidence (%)	Predicted Age	Processing Time (s)	Bounding Box
face1.jpg	wrinkles	92.3	56 yrs	1.42	(210,120,410,410)
📄 Project Structure
DermalScan-AI/
│
├── app.py                   # Streamlit web interface
├── dermal_model.h5          # MobileNetV2 trained model
├── requirements.txt         # Dependencies list
├── README.md                # Project documentation
├── AI_DermalScan.pdf        # Detailed report
└── main.ipynb               # Model integration and training notebook

🧩 Modules Implemented
Milestone	Module	Key Deliverable
1	Dataset Setup & Preprocessing	Organized and augmented facial image dataset
2	MobileNetV2 Integration	Optimized CNN model for skin feature classification
3	Face Detection Pipeline	Extraction and detection of facial regions
4	Frontend & Backend Integration	Streamlit-based interactive web interface
5	Export & Logging	Session logs and annotated outputs
6	Final Documentation	Project report and deployment-ready application
💡 Recent Additions

Added download functionality for annotated output images.

Implemented a session log system capturing bounding box coordinates, confidence, prediction time, and estimated age.

Enabled CSV export for session logs.

Conducted tests with real-time and multiple-face images to validate robustness and accuracy.

🧔‍♂️ Developer Notes

This project represents a complete integration of deep learning and interactive web UI to provide real-time facial skin analysis.
The use of MobileNetV2 ensures fast inference and scalability for edge and web deployment.
Ideal for research, skincare analytics, and healthcare applications.

📬 Author

Developer: Mohammed Mujahid Ahmed
Project: DermalScan AI
Technology Stack: TensorFlow, Streamlit, OpenCV, MobileNetV2
Year: 2025
