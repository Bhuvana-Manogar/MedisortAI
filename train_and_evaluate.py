import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Paths Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "MedBin_dataset.v7i.multiclass")
if not os.path.exists(DATASET_DIR):
    DATASET_DIR = r"C:\Users\admin\Downloads\MedBin_dataset.v7i.multiclass"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

print("Dataset Directory:", DATASET_DIR)
print("Train Directory  :", TRAIN_DIR)
print("Valid Directory  :", VALID_DIR)
print("Test Directory   :", TEST_DIR)

# 2. Load Datasets
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nDiscovered {num_classes} Classes: {class_names}")

# Performance optimization: prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds_perf = train_ds.prefetch(buffer_size=AUTOTUNE)
valid_ds_perf = valid_ds.prefetch(buffer_size=AUTOTUNE)
test_ds_perf = test_ds.prefetch(buffer_size=AUTOTUNE)

# 3. Model Architecture with Transfer Learning
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = layers.Lambda(preprocess_input)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="MediSort_MobileNetV2")
model.summary()

# 4. Compile Model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Train Model
EPOCHS = 5
print(f"\n--- Training MediSort AI for {EPOCHS} Epochs ---")
history = model.fit(
    train_ds_perf,
    validation_data=valid_ds_perf,
    epochs=EPOCHS
)

# 6. Save Model
model_save_path = os.path.join(BASE_DIR, "medical_waste_model.h5")
model.save(model_save_path)
print(f"\nModel saved successfully to: {model_save_path}")

alt_save_path = r"C:\Users\admin\medisort\medical_waste_model.h5"
try:
    model.save(alt_save_path)
    print(f"Model also synced to: {alt_save_path}")
except Exception as e:
    print(f"Could not sync to {alt_save_path}: {e}")

# 7. Plot & Save Training Curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", color="#2ecc71", lw=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#3498db", lw=2)
plt.title("MediSort AI - Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", color="#e74c3c", lw=2)
plt.plot(history.history["val_loss"], label="Validation Loss", color="#e67e22", lw=2)
plt.title("MediSort AI - Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
training_plot_path = os.path.join(BASE_DIR, "training_curves.png")
plt.savefig(training_plot_path, dpi=300)
plt.close()
print(f"Training curves saved to: {training_plot_path}")

# 8. Evaluation on Held-Out Test Set
print("\n--- Evaluating on Test Dataset ---")
test_loss, test_accuracy = model.evaluate(test_ds_perf)
print(f"Test Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

y_true = []
y_pred = []
for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n" + "="*50)
print("           FINAL TEST PERFORMANCE METRICS")
print("="*50)
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall   : {rec*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")
print("="*50)
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# 9. Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("MediSort AI - Test Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
cm_plot_path = os.path.join(BASE_DIR, "confusion_matrix.png")
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"Confusion matrix saved to: {cm_plot_path}")

# 10. OpenCV Feature Pipeline: CLAHE Preprocessing & ROI Contour Detection
print("\n--- Generating OpenCV Visual Feature Demonstration ---")
def opencv_preprocess_and_detect_roi(image_path):
    """
    OpenCV Feature Pipeline:
    1. Read image via cv2
    2. Convert to LAB color space and apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. Bilateral filter for edge-preserving noise reduction
    4. Canny Edge Detection + Morphological Closing
    5. Find external contours to detect medical waste ROI & Bounding Box
    """
    bgr = cv2.imread(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # CLAHE contrast enhancement
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b_chan))
    enhanced_rgb = cv2.cvtColor(cv2.cvtColor(limg, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    
    # Edge detection & Contours
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw ROI bounding box on original image
    annotated = rgb.copy()
    if contours:
        # Find largest contour (likely the waste item)
        largest_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cnt) > 200:
            x, y, w, h = cv2.boundingRect(largest_cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(annotated, "Medical Waste ROI", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    return rgb, enhanced_rgb, edges, annotated

# Sample test image for OpenCV demo
sample_test_files = []
for cat in class_names:
    cat_folder = os.path.join(TEST_DIR, cat)
    if os.path.exists(cat_folder) and os.listdir(cat_folder):
        sample_test_files.append(os.path.join(cat_folder, os.listdir(cat_folder)[0]))

if sample_test_files:
    demo_img_path = sample_test_files[0]
    orig, enhanced, edges, roi_img = opencv_preprocess_and_detect_roi(demo_img_path)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(orig)
    axes[0].set_title("1. Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(enhanced)
    axes[1].set_title("2. OpenCV CLAHE Enhanced")
    axes[1].axis("off")
    
    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("3. OpenCV Canny Edges")
    axes[2].axis("off")
    
    axes[3].imshow(roi_img)
    axes[3].set_title("4. OpenCV ROI Localization")
    axes[3].axis("off")
    
    plt.tight_layout()
    opencv_demo_path = os.path.join(BASE_DIR, "opencv_feature_demo.png")
    plt.savefig(opencv_demo_path, dpi=300)
    plt.close()
    print(f"OpenCV Feature demo saved to: {opencv_demo_path}")

# 11. Visual Predictions on Test Set
for images, labels in test_ds.take(1):
    predictions = model.predict(images, verbose=0)
    plt.figure(figsize=(14, 9))
    num_display = min(9, len(images))
    for i in range(num_display):
        predicted_idx = np.argmax(predictions[i])
        conf = np.max(predictions[i]) * 100
        actual_name = class_names[labels[i]]
        pred_name = class_names[predicted_idx]
        
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        color = "green" if actual_name == pred_name else "red"
        plt.title(f"Actual: {actual_name}\nPred: {pred_name} ({conf:.1f}%)", color=color, fontsize=10)
        plt.axis("off")
        
    plt.tight_layout()
    pred_plot_path = os.path.join(BASE_DIR, "test_predictions.png")
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    print(f"Test sample predictions saved to: {pred_plot_path}")

print("\n>>> Pipeline execution completed successfully! <<<")
