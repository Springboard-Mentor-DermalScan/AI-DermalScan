AI-Powered Facial Skin & Age Analysis System
Built by Shreya Bhat 

📘 Overview
DermalScan is an AI-based dermatological analysis application that leverages Deep Learning (DenseNet121) and Facial Detection (MTCNN) to analyze facial skin conditions and estimate biological age.
The system is designed with a futuristic Tech-Lab theme interface using Streamlit, providing real-time visual insights and professional-grade annotated image results.

🚀 Key Features

🧬 AI-Powered Skin Analysis — Detects facial features such as:
Wrinkles
Dark spots
Puffy eyes
Clear facial texture

📊 Biological Age Estimation — Predicts approximate age range based on detected skin features.

⚙️ Deep Learning Backbone — Fine-tuned DenseNet121 trained on curated facial datasets.

🔍 Real-Time Face Detection — Powered by MTCNN for robust and multi-face detection.

🧠 Interactive Streamlit Interface —

Tech-inspired dark UI with animated scan sequence

Upload image → AI scan → Download annotated output & CSV results

📁 Modular Design

Dataset cleaning & augmentation pipeline
Model training and fine-tuning workflow

Streamlit-based deployment frontend

🏗️ Project Structure
AI-DermalScan/
│
├── app.py                         # Streamlit Frontend (Tech-Lab Themed)
├── backend.py                     # Image processing + AI inference logic
├── densenet121_best_optimized.h5  # Trained model weights
│
├── documentation/                 # Project documentation and reports
├── test_images/                   # Sample images for testing
│
├── dataset_extraction.ipynb       # Jupyter notebook for dataset cleanup
├── requirements.txt               # All dependencies
└── README.md                      # Project overview (this file)

🧰 Tech Stack
Component	Technology
Frontend	Streamlit (Python)
Backend	TensorFlow / Keras
Model	DenseNet121 (Transfer Learning)
Detection	MTCNN (Multi-task Cascaded Convolutional Networks)
Image Processing	OpenCV, Pillow
Data Handling	Pandas, NumPy
Visualization	Matplotlib

⚙️ Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/<your-username>/AI-DermalScan.git
cd AI-DermalScan

2️⃣ Create a Virtual Environment
python -m venv dermalscan_env
source dermalscan_env/bin/activate   # (Mac/Linux)
dermalscan_env\Scripts\activate      # (Windows)

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit App
streamlit run app.py

🧬 Model Training (Optional)

To retrain or fine-tune the model:
Place your dataset ZIP file in the root directory.
Run the Jupyter Notebook or dataset extraction script:
python dataset_extraction.py

Train the model:
python backend.py


The best model weights are automatically saved as
densenet121_best_optimized.h5

🧪 Usage Instructions

Launch the app with Streamlit.
Upload a front-facing, clear image (JPG/PNG).
Wait for the animated scanning sequence to finish.

View:

Annotated image with bounding boxes
AI-generated feature insights
Biological age estimation
Download results as:

📊 CSV file

🖼️ Annotated image

🧠 Example Output
Feature	Confidence (%)	Estimated Age
Wrinkles	               87.45	68
Dark Spots	             65.23	34

Annotated image includes bounding boxes and feature labels.

📈 Model Details

Architecture: DenseNet121 (Pre-trained on ImageNet)
Optimizer: AdamW with cosine learning rate scheduling
Loss Function: Categorical Crossentropy (label smoothing = 0.1)
Augmentation: Random flip, zoom, rotation, brightness, contrast
Validation Accuracy: ~95% (fine-tuned model)

🧾 License

This project is released under the MIT License.
You may use, modify, and distribute it with proper attribution.

⚠️ Disclaimer

DermalScan is not a medical diagnostic tool.
It is developed purely for research and educational purposes.
For professional medical advice, always consult a certified dermatologist.

👩‍💻 Author
Shreya Bhat
