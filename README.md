# 🧬 AI-Powered Facial Skin & Age Analysis System  
### Built by **Shreya Bhat**

---

## 📘 Overview

**DermalScan** is an AI-based dermatological analysis application that leverages Deep Learning (**DenseNet121**) and Facial Detection (**MTCNN**) to analyze facial skin conditions and estimate biological age.  
It features a futuristic **Tech-Lab themed Streamlit interface**, providing real-time visual insights and professional-grade annotated image results.

---

## 🚀 Key Features

- 🧠 **AI-Powered Skin Analysis**
  - Detects facial features such as:
    - Wrinkles  
    - Dark spots  
    - Puffy eyes  
    - Clear skin texture  

- 📊 **Biological Age Estimation**
  - Predicts approximate age range based on skin features.

- ⚙️ **Deep Learning Backbone**
  - Fine-tuned **DenseNet121** model trained on curated facial datasets.

- 🔍 **Real-Time Face Detection**
  - Uses **MTCNN** for multi-face and robust feature detection.

- 💻 **Interactive Streamlit Interface**
  - Tech-inspired dark UI with animated scan sequence.
  - Upload → AI Scan → Download annotated results & CSV insights.

---

## 🏗️ Project Structure

AI-DermalScan/
│
├── app.py # Streamlit frontend (Tech-Lab themed)
├── backend.py # Image processing + AI inference logic
├── densenet121_best_optimized.h5 # Trained model weights
│
├── documentation/ # Project documentation and reports
├── test_images/ # Sample input images for testing
│
├── dataset_extraction.ipynb # Dataset cleanup and preprocessing notebook
├── requirements.txt # Dependency list
└── README.md # Project overview (this file)

yaml
Copy code

---

## 🧰 Tech Stack

| Component       | Technology |
|-----------------|-------------|
| Frontend        | Streamlit |
| Backend         | TensorFlow / Keras |
| Model           | DenseNet121 (Transfer Learning) |
| Detection       | MTCNN (Multi-task Cascaded CNN) |
| Image Handling  | OpenCV, Pillow |
| Data Handling   | Pandas, NumPy |
| Visualization   | Matplotlib |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/AI-DermalScan.git
cd AI-DermalScan
2️⃣ Create a Virtual Environment
bash
Copy code
python -m venv dermalscan_env
# Activate environment
dermalscan_env\Scripts\activate   # (Windows)
source dermalscan_env/bin/activate   # (Mac/Linux)
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Streamlit App
bash
Copy code
streamlit run app.py
🧬 Model Training (Optional)
If you wish to retrain or fine-tune the model:

Place your dataset ZIP file in the root directory.

Run the dataset extraction:
bash
Copy code
python dataset_extraction.py
Train the model:

bash
Copy code
python backend.py
The best model weights are automatically saved as
densenet121_best_optimized.h5.

🧪 Usage Instructions
Launch the app using Streamlit.

Upload a clear, front-facing image (JPG/PNG).

Wait for the animated AI scanning sequence.

You’ll receive:

✅ Annotated image with bounding boxes
✅ AI-generated feature insights
✅ Estimated biological age

Download results as:

📊 CSV file

🖼️ Annotated image

📈 Model Details
Parameter	Description
Architecture	DenseNet121 (pre-trained on ImageNet)
Optimizer	AdamW + cosine learning rate scheduler
Loss	Categorical Crossentropy (Label Smoothing = 0.1)
Augmentation	Random flip, zoom, rotation, brightness, contrast
Validation Accuracy	~95% (Fine-tuned model)

⚙️ Performance Tips
Use @st.cache_resource to cache models and detectors.

Resize input images ≤ 1024px before processing.

For GPU acceleration:

bash
Copy code
pip install tensorflow[and-cuda]
Convert model to TFLite for smaller size & faster inference.

🧾 License
Released under the MIT License.
You may use, modify, and distribute this project with proper attribution.

⚠️ Disclaimer
DermalScan is not a medical diagnostic tool.
It is for research and educational purposes only.
For professional medical advice, always consult a certified dermatologist.

Author
Shreya Bhat — aspiring Computer Science student
🌐 Building AI that cares for your skin.
