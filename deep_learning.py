# module_4.py — corrected and runnable (saves model as .keras)
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle as sk_shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # pyright: ignore[reportMissingImports]
from tensorflow.keras.utils import to_categorical  # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications import MobileNetV2  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization  # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Model  # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam  # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # pyright: ignore[reportMissingImports]

# ==============================
# CONFIGURATION
# ==============================
DATASET_PATH = r"C:\Users\hp\Downloads\archive (2)\DATASET"
CLASSES = ['wrinkles', 'dark spots', 'puffy eyes', 'clear face']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = (224, 224)
MODEL_PATH = "dermalscan_model.keras"   # use .keras for Keras 3+
OUTPUT_DIR = "outputs"
TEST_IMAGES_DIR = "TEST_IMAGES"

# Age group estimation (heuristic)
AGE_MAPPING = {
    'wrinkles': '50+',
    'dark spots': '40 or 50',
    'puffy eyes': '30 or 40',
    'clear face': '20 or 30'
}

# Test images list - Fixed paths
TEST_IMAGES = [
    r"C:\Users\hp\Downloads\archive (2)\DATASET\dark spots\101.jpg",
    r"C:\Users\hp\Downloads\archive (2)\DATASET\wrinkles\Image_179.jpg",
    r"C:\Users\hp\Downloads\archive (2)\DATASET\clear face\82.jpg",
    r"C:\Users\hp\Downloads\archive (2)\DATASET\puffy eyes\Image_66.jpg"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

# ==============================
# 1. DATA LOADING & PREPROCESSING
# ==============================
def load_and_preprocess_data():
    print(" Loading dataset from:", DATASET_PATH)
    images, labels = [], []

    for idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f" Missing folder: {class_path}")
        print(f"  Loading '{class_name}'...")
        count = 0
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(idx)
                count += 1
            except Exception as e:
                print(f" Skipping {img_path}: {e}")
        print(f"    → {count} images loaded")

    images = np.array(images)
    labels = np.array(labels)
    print(f"\n Total images loaded: {len(images)}")
    print(f" Images shape: {images.shape}")
    print(f" Labels shape: {labels.shape}")
    print(f" Number of classes: {NUM_CLASSES}")
    print(f" Image size: {IMG_SIZE}")
    return images, labels

# Execute DATA LOADING
print(" Starting DermalScan Pipeline...")
print("=" * 50)
images_all, labels_all = load_and_preprocess_data()

# ==============================
# 2. MODEL BUILDING
# ==============================
def build_model():
    print("\n Building model...")

    # Base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    # Custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(" Model built successfully!")
    print(f" Model layers: {len(model.layers)}")
    print(f" Output activation: softmax")
    print(f" Output shape: {model.output_shape}")
    print(f" Number of parameters: {model.count_params():,}")
    return model

# Build once (you may rebuild inside training too)
print("\n" + "=" * 50)
_base_model_preview = build_model()

# ==============================
# 3. MODEL TRAINING + EVALUATION
# ==============================
import seaborn as sns

def train_and_evaluate(images, labels):
    print(" Starting training process...")

    if len(images) == 0:
        raise ValueError("No images found. Check DATASET_PATH and subfolders.")

    # Shuffle images and labels together (important for stratify)
    images, labels = sk_shuffle(images, labels, random_state=42)

    # One-hot encode labels
    labels_cat = to_categorical(labels, NUM_CLASSES)

    # Split with stratify on integer labels
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels_cat, test_size=0.2, random_state=42, stratify=labels
    )

    # For second split we need the integer labels for stratify — recreate from y_temp
    strat_labels_temp = np.argmax(y_temp, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=strat_labels_temp
    )

    print(f"\n Data Split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")

    # Build model fresh for training
    model = build_model()

    # Callbacks — ensure filepath ends with .keras (Keras 3)
    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_PATH,             # .keras path
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Data augmentation (optional)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.02,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    batch_size = 16
    epochs = 10  # you can raise this

    # Fit using generator to support augmentation
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)

    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(X_train) // batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\n Test Accuracy: {test_accuracy:.2%}")
    print(f" Test Loss: {test_loss:.4f}")

    # Plot training history
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Training Accuracy', marker='o')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy', marker='s')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Training Loss', marker='o')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss', marker='s')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=200)
    plt.show()

    # Predictions for confusion matrix
    y_true = np.argmax(y_test, axis=1)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
    plt.show()

    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # Save final model (already saved by ModelCheckpoint best; but save final as well)
    print(f"\n Saving final model to: {MODEL_PATH}")
    model.save(MODEL_PATH)
    print(" Model saved successfully!")

    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f" Model file: {MODEL_PATH} ({file_size:.2f} MB)")

    return model, history

# Execute MODEL TRAINING
print("\n" + "=" * 50)
model, history = train_and_evaluate(images_all, labels_all)

# ==============================
# 4. FACE DETECTION (MULTI-FACE SUPPORT)
# ==============================
def detect_faces(image_path):
    print(f"\n Detecting faces in: {os.path.basename(image_path)}")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    if img is None:
        print(" Could not read image")
        return None, None, []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        print(" No faces detected")
        return None, img, []

    cropped_faces = []
    bboxes = []
    for (x, y, w, h) in faces:
        face_crop = img[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, IMG_SIZE)
        face_normalized = face_resized.astype(np.float32) / 255.0
        cropped_faces.append(face_normalized)
        bboxes.append((x, y, w, h))

    print(f" Detected {len(faces)} face(s)")

    # Show the original image with face detection
    img_with_faces = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f'Face Detection - {os.path.basename(image_path)}', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.show()

    return cropped_faces, img, bboxes

# ==============================
# 5. BATCH INFERENCE WITH ANNOTATION
# ==============================
def run_inference_on_test_images():
    print(f"\n Running inference on test images...")

    # Use TEST_IMAGES directly instead of copying to test_images directory
    test_images = []
    for img_path in TEST_IMAGES:
        if os.path.exists(img_path):
            test_images.append(img_path)
            print(f" Found: {os.path.basename(img_path)}")
        else:
            print(f" Missing: {img_path}")

    if not test_images:
        print(f" No test images available")
        return

    print(f" Found {len(test_images)} test images")

    # Load the trained model
    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]
        model_loaded = load_model(MODEL_PATH)
        print(" Loaded trained model for inference")
    else:
        print(" No trained model found. Please train the model first.")
        return

    for img_path in test_images:
        img_name = os.path.basename(img_path)
        print(f"\n Processing: {img_name}")

        cropped_faces, full_img, bboxes = detect_faces(img_path)

        if cropped_faces is None or len(cropped_faces) == 0:
            print(f" No face detected in {img_name}")
            continue

        print(f" Processing {img_name}:")
        print(f"  Faces detected: {len(cropped_faces)}")

        # Create annotated image
        annotated_img = full_img.copy()

        # Get image dimensions for text size calculation
        img_height, img_width = annotated_img.shape[:2]

        # Calculate dynamic text size based on image dimensions
        base_text_scale = max(0.5, min(1.5, img_width / 1000))
        base_font_scale = base_text_scale * 0.8
        base_thickness = max(1, int(base_text_scale * 2))
        base_rectangle_thickness = max(2, int(base_text_scale * 3))

        for i, (face, (x, y, w, h)) in enumerate(zip(cropped_faces, bboxes)):
            # Make prediction using the trained model
            input_tensor = np.expand_dims(face, axis=0)
            predictions = model_loaded.predict(input_tensor, verbose=0)
            pred_class_idx = np.argmax(predictions[0])
            pred_class = CLASSES[pred_class_idx]
            confidence = predictions[0][pred_class_idx]
            age_group = AGE_MAPPING.get(pred_class, "Unknown")

            print(f"  Face {i+1}:")
            print(f"    Bounding box: ({x}, {y}, {w}, {h})")
            print(f"    Prediction: {pred_class}")
            print(f"    Confidence: {confidence:.2%}")
            print(f"    Age Group: {age_group}")

            # Calculate text size based on face size
            face_size_factor = max(w, h) / 200
            font_scale = base_font_scale * face_size_factor
            thickness = base_thickness
            rectangle_thickness = base_rectangle_thickness

            # Ensure minimum and maximum limits
            font_scale = max(0.3, min(font_scale, 1.2))
            thickness = max(1, min(thickness, 3))
            rectangle_thickness = max(2, min(rectangle_thickness, 5))

            # Annotate the image
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), rectangle_thickness)
            label = f"{pred_class}: {confidence:.2%} ({age_group})"

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Adjust font scale if text is too wide
            max_text_width = w * 0.9
            if text_width > max_text_width:
                font_scale = font_scale * (max_text_width / text_width)
                font_scale = max(0.3, font_scale)
                thickness = max(1, int(thickness * 0.8))

            # Recalculate text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Calculate text position
            text_x = x
            text_y = y - 10 if y - 10 > text_height + 5 else y + text_height + 15

            # Add background for text
            cv2.rectangle(annotated_img,
                         (text_x, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + 5),
                         (0, 0, 0), -1)

            # Add text
            cv2.putText(annotated_img, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        # Show the annotated image
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_rgb)
        plt.title(f'Prediction Results - {img_name}\nDetected {len(cropped_faces)} face(s)',
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Save annotated image
        output_path = os.path.join(OUTPUT_DIR, f"annotated_{img_name}")
        cv2.imwrite(output_path, annotated_img)
        print(f" Annotated image saved: {output_path}")

# Execute INFERENCE
print("\n" + "=" * 50)
run_inference_on_test_images()

print("\n" + "=" * 50)
print(" DermalScan Pipeline Completed!")
print("=" * 50)
