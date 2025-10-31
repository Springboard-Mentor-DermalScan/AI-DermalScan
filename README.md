
# 🧠 DermalScan AI – Facial Skin Aging Detection System

### An intelligent deep learning system that reads your face to analyze skin aging signs.

---

## 🌟 Introduction – What Is DermalScan AI?

DermalScan AI is an artificial intelligence project designed to automatically *detect, analyze, and classify facial skin aging signs* using deep learning.

It focuses on identifying four major skin conditions from facial images:

* 🌞 *Dark Spots* – patches of uneven pigmentation, often caused by sunlight exposure or hormonal changes.
* 🕳 *Puffy Eyes* – swelling or puffiness around the eyes due to stress, fatigue, or aging.
* 💧 *Clear Skin* – balanced, healthy skin with minimal aging features.
* 🪞 *Wrinkles* – visible fine lines that appear with age and reduced skin elasticity.

The project uses *MobileNetV2, a lightweight yet powerful deep learning architecture, combined with **MTCNN* face detection to analyze facial regions precisely.

By connecting AI, dermatology, and visualization, DermalScan AI becomes more than just a project — it’s an example of how technology can be applied in health and skincare analysis.

---

## 🧰 Why This Project Was Created

The motivation behind DermalScan AI lies in combining *modern deep learning methods* with *human-centered design*.
The project was built to:

* Make *AI-based facial skin analysis* accessible to everyone, not just technical users.
* Educate students and beginners about how an AI project is developed from scratch.
* Demonstrate how *transfer learning* allows accurate results even with smaller datasets.
* Explore *real-time facial image classification* in a simple, deployable application.

In short, DermalScan AI bridges the gap between *AI research* and *real-world application*.

---

## 🧱 The Complete Project Pipeline

The workflow of DermalScan AI is carefully structured into seven modules.
Each module represents one essential part of the system — like a guided tour that takes you from raw data to a working web application.

| Step | Module                             | Goal                                                                |
| ---- | ---------------------------------- | ------------------------------------------------------------------- |
| ⿡  | Dataset Setup                      | Collect, clean, and organize facial images into 4 clear categories. |
| ⿢  | Image Preprocessing & Augmentation | Prepare and enrich images to make the model learn better.           |
| ⿣  | Model Design                       | Build a robust deep learning model using MobileNetV2.               |
| ⿤  | Model Training                     | Train the model and fine-tune it using augmented data.              |
| ⿥  | Evaluation                         | Test model accuracy and visualize results using graphs.             |
| ⿦  | Prediction & Visualization         | Detect new faces, predict the class, and annotate images.           |
| ⿧  | Deployment                         | Create a Streamlit web app for live testing and user interaction.   |

This pipeline ensures the project is modular, easy to understand, and adaptable for future improvements.

---

## 🖥 Environment Setup

### 💡 Why Python and VS Code?

* *Python 3.9+* is widely used in machine learning because it supports all essential libraries like TensorFlow, Keras, and OpenCV.
* *VS Code* offers a clean, beginner-friendly interface, built-in Git support, and an integrated terminal for smooth execution.

This combination makes it easy to write, run, and debug AI projects efficiently.

---

## 📦 Required Libraries and Their Purpose

DermalScan AI uses a collection of Python libraries, each serving a unique purpose within the system.

| Library          | Purpose                 | Why It’s Important                                      |
| ---------------- | ----------------------- | ------------------------------------------------------- |
| *TensorFlow*   | Deep learning framework | Core of model training and inference.                   |
| *Keras*        | Neural network API      | Simplifies model creation and training.                 |
| *OpenCV*       | Image processing        | Handles image resizing, cropping, and color conversion. |
| *MTCNN*        | Face detection          | Detects faces precisely before classification.          |
| *NumPy*        | Numerical computing     | Efficiently handles large image data arrays.            |
| *Pandas*       | Data organization       | Stores predictions and logs results in structured form. |
| *Matplotlib*   | Data visualization      | Creates training accuracy and loss plots.               |
| *Scikit-learn* | Model evaluation        | Generates performance metrics like confusion matrix.    |
| *Pillow (PIL)* | Image annotation        | Adds prediction labels and confidence scores to images. |
| *Streamlit*    | Web application         | Converts the AI model into an interactive web app.      |

Each of these tools is like a puzzle piece — together, they form the complete DermalScan AI system.

---

## 📁 Module 1 – Dataset Setup and Image Labeling

This module lays the foundation for the entire project.

Here, facial images are carefully collected and divided into four labeled folders — each representing one of the target skin conditions:
Clear Face, Dark Spots, Puffy Eyes, and Wrinkles.

A well-structured dataset ensures that the model can learn properly without confusion.
Each class has a balanced number of images (around 100 per category), preventing bias toward any one skin type.

The dataset is visually inspected to ensure clarity and correctness — any noisy or mislabeled image is removed.
Finally, sample visualizations are displayed to confirm that images are correctly grouped under each label.

---

## 🧼 Module 2 – Image Preprocessing and Augmentation

Before an AI model can understand images, they must be prepared in a consistent, machine-readable format.

In this stage:

* All images are resized to *224x224 pixels* to match the MobileNetV2 input format.
* Pixel values are normalized (converted to a 0–1 range) for faster learning.
* Data augmentation techniques are applied — introducing random transformations like rotation, flipping, zooming, brightness change, and shifting.

This process effectively “teaches” the model how to recognize faces in many lighting conditions, angles, and skin tones, improving generalization.

By the end of this module, the dataset becomes diverse, balanced, and ready for training.

---

## 🧠 Module 3 – Model Design (Using MobileNetV2)

This module builds the *core intelligence* of DermalScan AI.

Instead of training a model entirely from scratch, MobileNetV2 — a pre-trained model from Google — is used as the *base feature extractor*.
It already understands general visual features like edges, textures, and patterns, which makes training faster and more accurate.

On top of MobileNetV2, custom layers are added:

* *Global Average Pooling* to compress features efficiently.
* *Batch Normalization* to stabilize learning.
* *Dropout Layers* to prevent overfitting.
* *Dense Layers* to connect the learned features to final skin classes.
* *Softmax Output Layer* to provide probabilities for each skin category.

This layered structure gives DermalScan AI a balance of *speed, accuracy, and compactness* — perfect for running on both local systems and web servers.

---

## 🧮 Module 4 – Model Training and Validation

Once the architecture is ready, the model enters the learning phase.

Training occurs in two main stages:

*Phase 1 – Initial Training:*
Only the newly added top layers are trained, while the pre-trained MobileNetV2 base remains frozen.
This helps the new layers learn skin-related patterns without disturbing the original learned features.

*Phase 2 – Fine-tuning:*
Some deeper layers of MobileNetV2 are unfrozen and retrained with a smaller learning rate to fine-tune their parameters.
This helps the model adapt more precisely to the specific dataset.

During training, various callbacks like *ModelCheckpoint* and *ReduceLROnPlateau* are used to automatically save the best-performing model and adjust the learning rate.

After about 60–80 minutes of training, the model achieves a validation accuracy of around *86–89%*, with minimal overfitting — a strong performance for a small dataset.

---

## 📈 Module 5 – Model Evaluation and Metrics

Once trained, the model’s performance is carefully evaluated on unseen test images.

The results are measured using metrics such as accuracy, precision, recall, and F1-score.
The confusion matrix visualizes how well each class was identified.

Typical outcomes include:

* Clear Face – 93% accuracy
* Dark Spots – 85% accuracy
* Puffy Eyes – 83% accuracy
* Wrinkles – 88% accuracy

The overall model accuracy stabilizes between *85.94% and 89.06%*, showing consistent predictions across categories.

This proves that the model has successfully learned the unique patterns of each skin type.

---

## 🖼 Module 6 – Prediction and Result Visualization

Here, the model is applied to new, real-world images.

When a user uploads a photo, the *MTCNN face detector* first identifies the face area and crops it.
The cropped region is then analyzed by the trained MobileNetV2 model.
The model predicts the most likely skin condition and provides a confidence percentage.

To make the output meaningful, an *age range mapping* is included:

| Skin Condition | Estimated Age Range |
| -------------- | ------------------- |
| Clear Face     | 10–29 years         |
| Dark Spots     | 30–59 years         |
| Puffy Eyes     | 30–59 years         |
| Wrinkles       | 60+ years           |

These mappings make the results easy for non-technical users to interpret.
Annotated output images display bounding boxes, predicted labels, and confidence values clearly.

---

## 📑 Module 7 – Web Deployment and Automated Reporting

After achieving strong performance, the system is turned into a *Streamlit-based web application*.

Through the web app:

* Users can upload their facial image.
* The AI model instantly predicts the detected skin condition.
* Results are displayed visually with annotations and confidence scores.
* Logs are automatically saved in CSV and JSON formats.

This deployment makes the AI system interactive, user-friendly, and ready for demonstration or presentation.

The deployed app runs both locally and publicly through *Streamlit Cloud*, offering real-time accessibility and high uptime performance.

---

## 📊 Final Results Summary

* Total Dataset: 397 images
* Model Used: MobileNetV2 (transfer learning from ImageNet)
* Training Accuracy: 90.34%
* Validation Accuracy: 85.94–89.06%
* Testing Accuracy: 85.94%
* Model Size: approximately 9.8 MB
* Average Prediction Time: 1.7 seconds per image
* Deployment: Streamlit Cloud with 99.5% uptime

These results demonstrate both the efficiency and reliability of the system in real-world use.

---

## 🧭 Conclusion – The Full Journey

DermalScan AI is a complete, end-to-end artificial intelligence system designed to detect and classify facial skin aging conditions.

It combines *MTCNN* for face detection, *MobileNetV2* for classification, and *Streamlit* for deployment — offering a powerful yet easy-to-understand example of applied deep learning.

Beyond its technical achievements, this project stands out for its *clarity, educational value, and accessibility*.
Anyone — even without AI knowledge — can read this documentation, follow the workflow, and understand exactly how the system works.

DermalScan AI proves that artificial intelligence doesn’t have to be complicated to be effective.

---

👨‍💻 *Author:* Boini Pramod Kumar
