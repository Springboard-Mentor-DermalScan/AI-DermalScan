🧠 DermalScan: AI Facial Skin Aging Detection App
🌸 Overview

DermalScan is an AI-powered web application that detects facial skin conditions and estimates age from an uploaded image.
Built using MobileNetV2 with Streamlit frontend, it can analyze multiple faces in a single frame, highlight each with bounding boxes, and predict the condition with confidence levels.

🚀 Features

✅ Upload face images (.jpg, .jpeg, .png)
✅ Detect multiple faces in a single image
✅ Predict skin conditions → Clear, Dark Spots, Puffy Eyes, Wrinkles
✅ Estimate Age for each detected face
✅ Display confidence scores and bounding box coordinates
✅ Download annotated output image and CSV report
✅ End-to-end processing within ≤ 5 seconds per image

🧩 Tech Stack
Area	Tools / Libraries
Frontend	Streamlit
Backend	Python
Model	MobileNetV2 (Keras/TensorFlow)
Image Processing	OpenCV, NumPy, Haarcascade
Data Handling	Pandas
Visualization	Matplotlib, Seaborn
Exporting	CSV, Annotated Image
Accuracy (Validation)	~87%
📂 Project Structure
📁 DermalScan/
│
├── app.py                # Streamlit frontend (UI)
├── backend.py            # Model inference and preprocessing
├── best_model.h5         # Trained MobileNetV2 model
├── uploads/              # Temporary image storage
├── results/              # Generated annotated images and CSVs
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation

⚙️ Installation

1️⃣ Clone the repository

git clone https://github.com/<your-username>/DermalScan.git
cd DermalScan


2️⃣ Install dependencies

pip install -r requirements.txt


3️⃣ Run the Streamlit app

streamlit run app.py

🖼️ Usage

Launch the Streamlit app.

Upload an image with one or multiple faces.

Wait for model inference (approx. 3–5 seconds).

View:

Left: Input image

Right: Annotated output with bounding boxes

Below: Prediction table with labels, confidence, age, and coordinates

Download the annotated image or CSV results.

📊 Sample Output
Class ID	Condition	Confidence	Estimated Age	x	y	width	height	Time (s)
0	Clear	97.89%	28	112	80	120	98	3.21
📈 Results

Model Used: MobileNetV2 (trained on labeled facial dataset)

Validation Accuracy: 87%

Inference Time: ≤ 5 seconds per image

Output Format: Annotated image + CSV report

🧰 Requirements
keras==3.11.3  
tensorflow==2.20.0  
numpy==2.2.6  
opencv-python==4.12.0.88  
scipy==1.16.2  
pandas==2.2.2  
matplotlib==3.10.6  
seaborn==0.13.2  
streamlit==1.50.0  

📚 Milestone Reference

Milestone 3: Frontend and Backend Integration

Module 5: Web UI for Image Upload & Visualization

Module 6: Backend Pipeline for Model Inference

👩‍💻 Developer

Shivani Reddy
B.Tech, Artificial Intelligence & Machine Learning
Malla Reddy Engineering College
