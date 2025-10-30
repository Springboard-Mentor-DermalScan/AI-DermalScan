🌸 DermalScan – AI Facial Aging Detection System

DermalScan is an AI-powered web application designed to analyze facial images and detect signs of aging such as wrinkles, dark spots, puffy eyes, and clear skin.
The system integrates Deep Learning (InceptionV3) and Computer Vision (OpenCV) to deliver accurate predictions with annotated visual feedback.

💡 Objective

To develop a user-friendly and intelligent facial analysis platform that detects aging patterns and provides interpretable AI insights for dermatological applications.

🧠 Key Features

🧍 Automatic Face Detection using OpenCV Haar Cascade.

🤖 Deep Learning Classification of four facial conditions.

🧾 Downloadable Reports (CSV and annotated images).

🧠 Heuristic Age Estimation based on detected skin conditions.

🌐 Interactive Streamlit Web Interface for easy use.

⚙️ Technologies Used
Category	Tools / Libraries
Deep Learning	TensorFlow, Keras (InceptionV3)
Web Framework	Streamlit
Image Processing	OpenCV, Pillow
Data Handling	Pandas, NumPy
Visualization & Docs	Jupyter Notebook, MS Word




📁 Project Structure
├── app.py                        # Streamlit frontend web application
├── backend.py                    # Model loading, preprocessing, and prediction logic
├── DermalScan_Mode.ipynb         # Jupyter notebook for model training & testing
├── Project_Documentation.docx    # Project report/documentation file
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation




🧩 Installation Guide
1️⃣ Clone the Repository
https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan.git
cd AI-DermalScan

2️⃣ Install Dependencies

Make sure you have Python 3.8 or later installed, then run:

pip install -r requirements.txt

3️⃣ Add Model File

Ensure the trained model file best_inceptionv3_model2.h5 is present in the project folder.

4️⃣ Launch the Application
streamlit run app.py


Open the browser window at the provided local URL (e.g. http://localhost:8501).

🧬 Model Summary

Base Architecture: InceptionV3 (pretrained on ImageNet)

Input Size: 224 × 224 × 3

Output Classes: 4

Clear Face

Dark Spots

Puffy Eyes

Wrinkles

Heuristic Age Range Mapping:

Clear Face → 18–30 years

Dark Spots → 30–40 years

Puffy Eyes → 40–55 years

Wrinkles → 56–75 years

🧾 requirements.txt
streamlit
tensorflow
opencv-python
pandas
numpy
Pillow

🚀 Future Enhancements

Real-time webcam-based facial aging detection

Integration of skin tone & texture analysis

Deployment as a REST API or cloud web app

Integration with dermatological recommendation systems

Improved model generalization through larger, diverse datasets

🧑‍💻 Developer

Battula Bhulakshmi
B.Tech in Computer Science & Engineering
Rajiv Gandhi University of Knowledge Technologies, Ongole
