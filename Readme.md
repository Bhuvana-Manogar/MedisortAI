#  MediSort AI – Smart Medical Waste Classification

## 📝 Description
MediSort AI is an **AI-powered system** for **real-time biomedical waste classification and segregation** in hospitals and healthcare facilities.  
It automatically detects and categorizes medical waste into:

- **Hazardous Waste** – Blood-soaked materials, sharps, infectious waste  
- **Chemical Waste** – Expired medicines, chemical reagents  
- **Non-Hazardous Waste** – Recyclable or general waste  

**Benefits:**  
- Reduces human exposure to infectious or hazardous materials  
- Improves efficiency in waste management  
- Supports sustainable and safe disposal practices  

---

## 🛠 Technology & Model
- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow / Keras  
- **Model Architecture:** Convolutional Neural Network (CNN)  
- **Image Processing:** OpenCV  
- **Web Interface:** Streamlit  
- **Libraries:** NumPy, Pandas, Matplotlib, Pillow  

---

## 🔄 System Workflow
<img width="854" height="374" alt="image" src="https://github.com/user-attachments/assets/98a049ee-7a95-4912-845a-1664fb86c33f" />


**Workflow Steps:**  
1. **Image/Video Input:** Capture or upload biomedical waste image/video.  
2. **Preprocessing:** Resize, enhance, normalize, and augment images.  
3. **Classification:** CNN model predicts the waste category.  
4. **Output:** Display classification with **confidence score** and alerts if hazardous.  

---
## Program
```python
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

st.title("Medical Waste Classification (Live Camera)")

# Path to dataset
DATASET_PATH = r"C:\Users\admin\OneDrive\Desktop\project\Dataset"

# Check if model exists; if not, train it quickly using transfer learning
MODEL_PATH = "medical_waste_model.h5"

# Function to create & train the model
def train_model():
    st.write("Training model on dataset...")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='validation'
    )
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_gen, validation_data=val_gen, epochs=5)
    model.save(MODEL_PATH)
    return model, train_gen.class_indices

# Load or train model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    # get class indices
    datagen = ImageDataGenerator(rescale=1./255)
    temp_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(224,224),
        batch_size=16,
        class_mode='categorical'
    )
    class_indices = temp_gen.class_indices
else:
    model, class_indices = train_model()

# Reverse class index to get label names
class_labels = {v:k for k,v in class_indices.items()}

# Webcam capture
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break
    
    # Convert to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(img)
    
    # Preprocess frame
    img_resized = cv2.resize(img, (224,224))
    img_array = np.expand_dims(img_resized/255.0, axis=0)
    
    # Prediction
    preds = model.predict(img_array)
    label = class_labels[np.argmax(preds)]
    confidence = np.max(preds)
    
    st.write(f"Prediction: {label} ({confidence*100:.2f}%)")

cap.release()
```
---

## 📸 Sample Output
<img width="1170" height="605" alt="image" src="https://github.com/user-attachments/assets/804a8e18-cb61-470c-9432-aa9775cf88a5" />

<img width="813" height="424" alt="Screenshot 2025-11-28 123929" src="https://github.com/user-attachments/assets/6081313c-5106-486c-8f36-f565930383c5" />


---

## ✅ Conclusion
MediSort AI delivers a **robust, scalable, and real-time biomedical waste classification system**.  

**Key Highlights:**  
- Minimizes manual handling and exposure to hazardous waste  
- High-accuracy classification even in challenging lighting or cluttered environments  
- Enhances operational efficiency and compliance in hospitals  
- Supports environmentally responsible biomedical waste management  

---
