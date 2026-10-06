import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="MediSort AI — Medical Waste Classification",
    page_icon="🏥",
    layout="wide"
)

# Constants & Class Definitions
CATEGORIES = [
    'General_and_Glass_Waste',
    'Infectious_Waste',
    'Pharmaceutical_Waste',
    'Plastic_Medical_Waste',
    'Sharps_Waste'
]

WHO_DISPOSAL_GUIDELINES = {
    'Infectious_Waste': {
        'bin_color': '🟡 Yellow Biohazard Bin',
        'badge_color': '#f1c40f',
        'symbol': '☣️ Infectious / Biohazard',
        'examples': 'Cotton swabs, gauze, bloody dressings, soiled bandages, surgical masks, exam gloves, covid test kits',
        'treatment': 'High-temperature incineration or double autoclaving before controlled sanitary landfill disposal.',
        'safety': 'Handle with thick protective gloves. Never compress or overfill yellow biohazard bags.'
    },
    'Sharps_Waste': {
        'bin_color': '⚪ White Puncture-Proof Container',
        'badge_color': '#e74c3c',
        'symbol': '🗡️ Sharps / Puncture Hazard',
        'examples': 'Hypodermic needles, syringes with needles, surgical scalpels, lancets, suture needles, razor blades',
        'treatment': 'Puncture-proof rigid container, needle destruction/cutting, shredding, and encapsulation in concrete.',
        'safety': 'NEVER recap needles. Drop directly into puncture-proof container immediately after single use.'
    },
    'Pharmaceutical_Waste': {
        'bin_color': '🟤 Brown Container / Bin',
        'badge_color': '#8e44ad',
        'symbol': '💊 Pharmaceutical / Chemical',
        'examples': 'Expired drugs, antibiotic capsules, tablets, vaccine residues, chemical reagents, topical unguents',
        'treatment': 'High-temperature rotary kiln incineration (>1200°C) or chemical encapsulation.',
        'safety': 'Keep original packaging where possible. Prevent discharge into general hospital plumbing or municipal sewage.'
    },
    'Plastic_Medical_Waste': {
        'bin_color': '🔴 Red Biohazard Bin',
        'badge_color': '#e67e22',
        'symbol': '🧪 Recyclable Contaminated Plastic',
        'examples': 'Intravenous (IV) fluid bottles, catheters, dialysis tubing, plastic pipettes, needleless syringes',
        'treatment': 'Autoclaving / hydroclaving chemical disinfection followed by energy recovery or licensed plastic recycling.',
        'safety': 'Separate plastic tubing from metal needles before placing in red containers.'
    },
    'General_and_Glass_Waste': {
        'bin_color': '🔵 Blue / Black Container',
        'badge_color': '#2980b9',
        'symbol': '📦 Glassware & Non-Hazardous',
        'examples': 'Medicine vials, ampoules, glass specimen bottles, paper boxes, clean external packaging',
        'treatment': 'Glassware disinfection & recycling; clean packaging sent to standard municipal recycling.',
        'safety': 'Ensure ampoules are completely empty. Avoid mixing broken glass with soft biohazard bags.'
    }
}

# 1. Load Trained Model
@st.cache_resource
def get_model():
    model_paths = [
        os.path.join(os.path.dirname(__file__), "medical_waste_model.h5"),
        r"C:\Users\admin\Downloads\medisort\medical_waste_model.h5",
        r"C:\Users\admin\medisort\medical_waste_model.h5"
    ]
    for p in model_paths:
        if os.path.exists(p):
            try:
                m = load_model(p)
                return m, p
            except Exception as e:
                continue
    return None, None

model, loaded_path = get_model()

# 2. OpenCV Feature Processing Functions
def opencv_process_pipeline(pil_image, clahe_clip=3.0, canny_thresh1=50, canny_thresh2=150):
    """
    OpenCV Computer Vision Pipeline:
    1. BGR / RGB conversion
    2. LAB Color Space CLAHE (Contrast-Limited Adaptive Histogram Equalization)
    3. Bilateral Filter for edge-preserving noise reduction
    4. Canny Edge Detection & Morphological Closing
    5. External Contour Detection & ROI Bounding Box extraction
    """
    rgb = np.array(pil_image.convert('RGB'))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 1. CLAHE Contrast Enhancement in LAB space
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged_lab = cv2.merge((cl, a, b_chan))
    enhanced_rgb = cv2.cvtColor(cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)

    # 2. Bilateral filtering + Canny Edge Detection
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

    # 3. Morphological closing to connect fragmented edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 4. Contour detection & ROI bounding box
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = rgb.copy()
    roi_detected = False

    if contours:
        # Sort by contour area
        sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_cnt = sorted_cnts[0]
        if cv2.contourArea(largest_cnt) > 250:
            x, y, w, h = cv2.boundingRect(largest_cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (39, 174, 96), 3)
            cv2.putText(
                annotated,
                "Medical Waste ROI",
                (x, max(22, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (39, 174, 96),
                2
            )
            roi_detected = True

    return {
        'original': rgb,
        'enhanced': enhanced_rgb,
        'edges': edges,
        'annotated': annotated,
        'roi_detected': roi_detected
    }

def predict_waste(pil_image):
    if model is None:
        return None, 0.0, {}
    
    img_resized = pil_image.convert('RGB').resize((224, 224))
    arr = np.array(img_resized, dtype=np.float32)
    # Check if model has lambda preprocess_input
    arr_batch = np.expand_dims(arr, axis=0)
    
    # Try direct predict or preprocessed
    try:
        preds = model.predict(arr_batch, verbose=0)[0]
    except Exception:
        arr_pre = preprocess_input(arr_batch)
        preds = model.predict(arr_pre, verbose=0)[0]
        
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    pred_class = CATEGORIES[pred_idx]
    
    prob_dict = {CATEGORIES[i]: float(preds[i]) for i in range(len(CATEGORIES))}
    return pred_class, confidence, prob_dict

# UI Header
st.title("🏥 MediSort AI — Medical Waste Classification & Segregation")
st.markdown(
    "**End-to-End Deep Learning & Computer Vision System** | Transfer Learning with **MobileNetV2** & **OpenCV Preprocessing**"
)

# Sidebar
st.sidebar.header("⚙️ Configuration & Controls")

input_mode = st.sidebar.radio(
    "Choose Input Source:",
    ["📤 Upload Image", "📸 Live Webcam Capture", "🧪 Sample Dataset Images"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 OpenCV Processing Controls")
clahe_clip = st.sidebar.slider("CLAHE Clip Limit (Contrast)", 1.0, 6.0, 3.0, 0.5)
canny_low = st.sidebar.slider("Canny Low Threshold", 10, 100, 50, 10)
canny_high = st.sidebar.slider("Canny High Threshold", 100, 250, 150, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ System Status")
if model is not None:
    st.sidebar.success(f"✅ MobileNetV2 Loaded\n({os.path.basename(loaded_path)})")
else:
    st.sidebar.warning("⚠️ Model weights loading or training...")

st.sidebar.markdown(
    """
    **Supported WHO Categories:**
    - 🟡 Infectious Waste
    - ⚪ Sharps Waste
    - 🟤 Pharmaceutical Waste
    - 🔴 Plastic Medical Waste
    - 🔵 Glass & General Waste
    """
)

# Input Handling
image_to_process = None

if input_mode == "📤 Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a medical waste image (JPEG, PNG, JPG):",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)

elif input_mode == "📸 Live Webcam Capture":
    camera_file = st.camera_input("Take a photo using your webcam")
    if camera_file is not None:
        image_to_process = Image.open(camera_file)

elif input_mode == "🧪 Sample Dataset Images":
    test_base = r"C:\Users\admin\Downloads\MedBin_dataset.v7i.multiclass\test"
    sample_options = {}
    if os.path.exists(test_base):
        for cat in CATEGORIES:
            cat_dir = os.path.join(test_base, cat)
            if os.path.exists(cat_dir):
                files = os.listdir(cat_dir)
                if files:
                    sample_options[f"Sample {cat}"] = os.path.join(cat_dir, files[0])
    
    if sample_options:
        selected_sample = st.selectbox("Choose a sample test image:", list(sample_options.keys()))
        if selected_sample:
            image_to_process = Image.open(sample_options[selected_sample])
    else:
        st.info("Dataset samples loading...")

# Main Processing Section
if image_to_process is not None:
    st.markdown("---")
    
    # 1. OpenCV Preprocessing
    cv_results = opencv_process_pipeline(
        image_to_process,
        clahe_clip=clahe_clip,
        canny_thresh1=canny_low,
        canny_thresh2=canny_high
    )
    
    st.subheader("🔬 OpenCV Computer Vision Feature Analysis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(cv_results['original'], caption="1. Original Input", use_container_width=True)
    with col2:
        st.image(cv_results['enhanced'], caption="2. OpenCV CLAHE Enhanced", use_container_width=True)
    with col3:
        st.image(cv_results['edges'], caption="3. OpenCV Canny Edges", use_container_width=True)
    with col4:
        st.image(cv_results['annotated'], caption="4. OpenCV ROI Bounding Box", use_container_width=True)
        
    st.markdown("---")
    
    # 2. Deep Learning Classification
    if model is not None:
        pred_class, confidence, prob_dict = predict_waste(image_to_process)
        guidelines = WHO_DISPOSAL_GUIDELINES.get(pred_class, {})
        
        st.subheader("🧠 Deep Learning Classification & Segregation Protocol")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown(
                f"""
                <div style="background-color:#f8f9fa; border-left: 6px solid {guidelines.get('badge_color', '#3498db')}; padding: 18px; border-radius: 6px; margin-bottom: 15px;">
                    <h3 style="margin:0; color:#2c3e50;">Predicted Category: <b>{pred_class.replace('_', ' ')}</b></h3>
                    <p style="margin:6px 0 0 0; font-size:18px;"><b>Confidence:</b> {confidence*100:.2f}%</p>
                    <p style="margin:4px 0 0 0; font-size:18px;"><b>Recommended Disposal:</b> {guidelines.get('bin_color')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.progress(min(1.0, max(0.0, confidence)))
            
            st.markdown(f"**Hazard Category:** {guidelines.get('symbol')}")
            st.markdown(f"**Typical Items:** {guidelines.get('examples')}")
            st.markdown(f"**Treatment Protocol:** {guidelines.get('treatment')}")
            st.warning(f"⚠️ **Safety Advisory:** {guidelines.get('safety')}")
            
        with res_col2:
            st.markdown("#### 📊 Prediction Probability Distribution")
            df_probs = pd.DataFrame({
                'Category': [c.replace('_', ' ') for c in prob_dict.keys()],
                'Probability (%)': [round(v * 100, 2) for v in prob_dict.values()]
            })
            
            chart = alt.Chart(df_probs).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5).encode(
                x=alt.X('Probability (%):Q', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('Category:N', sort='-x'),
                color=alt.Color('Category:N', legend=None)
            ).properties(height=260)
            
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Loading model weights... Please train or ensure medical_waste_model.h5 exists.")
else:
    st.info("👆 Please upload an image, use your webcam, or select a sample image above to see MediSort AI in action!")
