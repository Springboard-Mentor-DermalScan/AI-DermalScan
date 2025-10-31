# DermalScan AI: Skin Aging & Condition Detector

✨ DermalScan AI is a web application that uses deep learning to analyze facial images for signs of aging and various skin conditions. Upload an image, and the AI will detect the face, classify conditions like wrinkles, dark spots, or puffy eyes, and provide an estimated biological age range.

![Demo Screenshot](https://user-images.githubusercontent.com/86633388/194758512-b869d387-2646-4467-8b2e-6f7f3a8b8328.png) 
*(Note: The image above is a representative example of a Streamlit application interface.)*

---

## 🚀 Features

- **AI-Powered Skin Analysis**: Utilizes a DenseNet model built with TensorFlow/Keras to classify skin conditions.
- **Face Detection**: Automatically identifies faces in uploaded images using OpenCV's Haar Cascade classifier.
- **Condition Classification**: Detects and categorizes conditions into:
  - `clear face`
  - `darkspots`
  - `puffy eyes`
  - `wrinkles`
- **Biological Age Estimation**: Provides a plausible age range based on the detected skin condition.
- **Interactive Web UI**: A user-friendly interface built with Streamlit for easy image uploads and clear result visualization.
- **Results History**: Keeps a running log of all predictions made during the session.
- **Data Export**: Allows users to download the annotated image and a CSV file containing the complete prediction history.

---

## ⚙️ How It Works

1.  **Image Upload**: The user uploads a clear, front-facing facial image via the Streamlit sidebar.
2.  **Face Detection**: The application uses an OpenCV Haar Cascade model to locate the face in the image. A small padding is added to ensure the entire facial region is captured.
3.  **Image Preprocessing**: The detected face is cropped, resized to 224x224 pixels, and preprocessed to match the model's input requirements.
4.  **AI Prediction**: The preprocessed image is passed to the pre-trained DenseNet model, which predicts the primary skin condition and calculates a confidence score.
5.  **Age Estimation**: Based on the predicted condition, a random age is generated from a predefined range associated with that condition.
6.  **Visualization**: The original image is displayed alongside the processed image, which features a bounding box around the detected face and an overlay with the analysis results (condition, confidence, estimated age).
7.  **History Logging**: The results of the analysis are added to a persistent table for the current session.

---

## 🛠️ Tech Stack

- **Backend & ML**: Python, TensorFlow, Keras, OpenCV, Scikit-learn
- **Frontend**: Streamlit
- **Data Handling**: Pandas, NumPy

---

## 📦 Installation & Setup

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Dermal_Scan_project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file with the following content:
    ```txt
    streamlit
    tensorflow
    opencv-python-headless
    pandas
    numpy
    ```
    Then, run the installation command:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Model:**
    Make sure you have the `best_densenet_model.h5` file. The application currently looks for it at `E:\New_Projects\streamlit\best_densenet_model.h5`. You should update the `model_path` in `frontend.py` to point to its correct location on your system.

5.  **Run the Streamlit App:**
    ```bash
    streamlit run frontend.py
    ```

6.  Open your web browser and navigate to `http://localhost:8501`.

---

## 📂 Project Structure

```
Dermal_Scan_project/
├── main.py               # Main Streamlit application script
├── best_densenet_model.h5    # Pre-trained Keras model
├── requirements.txt          # Python dependencies
├── uploads/                  # Directory for user-uploaded images (created at runtime)
├── results/                  # Directory for annotated images and CSV logs (created at runtime)
└── README.md                 # Project documentation
```