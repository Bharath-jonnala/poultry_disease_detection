import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Poultry Disease Detection",
    page_icon="🐔",
    layout="centered"
)

st.markdown(
    """
    <style>
    .main { background-color: #f7f9fc; }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# LOAD MODEL (FIXED PATH HANDLING)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "best.pt")
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    
    return YOLO(model_path)

model = load_model()

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🐔 Poultry Disease Detection System")
st.caption("Disease detection with reliability-aware inference")

st.write("Upload a poultry image to detect disease and assess prediction reliability.")

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------
if uploaded_file is not None and model is not None:

    # Save uploaded image
    suffix = "." + uploaded_file.name.split(".")[-1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    image_path = temp_file.name

    # Read image
    img = cv2.imread(image_path)

    # -------------------------------------------------
    # YOLO PREDICTION
    # -------------------------------------------------
    results = model(image_path, conf=0.25, verbose=False)

    if len(results[0].boxes) == 0:
        st.error("No disease detected.")
    else:
        pred_class = int(results[0].boxes.cls[0])
        pred_conf = float(results[0].boxes.conf[0])

        class_names = {0: "Fowl Pox", 1: "Healthy"}

        st.subheader("🧠 Prediction Result")
        st.write(f"**Disease:** {class_names[pred_class]}")
        st.write(f"**Confidence:** {pred_conf:.2f}")

        # -------------------------------------------------
        # SHOW ORIGINAL & DETECTED IMAGES
        # -------------------------------------------------
        annotated_img = results[0].plot()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(img, channels="BGR", width=350)

        with col2:
            st.subheader("📦 Detected Disease Region")
            st.image(annotated_img, channels="BGR", width=350)

        # -------------------------------------------------
        # RELIABILITY ANALYSIS
        # -------------------------------------------------
        pred_classes = []
        pred_scores = []
        augmented_images = []

        for i in range(20):
            aug = img.copy()

            # Blur
            if i % 2 == 0:
                aug = cv2.GaussianBlur(aug, (5, 5), 0)

            # Brightness variation
            alpha = 1.0 + np.random.uniform(-0.1, 0.1)
            aug = cv2.convertScaleAbs(aug, alpha=alpha)

            # Store first 6 images only (FIX)
            if i < 6:
                augmented_images.append(aug)

            temp_aug = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_aug.name, aug)

            r = model(temp_aug.name, conf=0.25, verbose=False)

            if len(r[0].boxes) > 0:
                pred_classes.append(int(r[0].boxes.cls[0]))
                pred_scores.append(float(r[0].boxes.conf[0]))

        if len(pred_scores) == 0:
            st.warning("Unable to compute reliability.")
        else:
            mean_conf = np.mean(pred_scores)
            std_conf = np.std(pred_scores)

            unique, counts = np.unique(pred_classes, return_counts=True)
            agreement = max(counts) / len(pred_classes)

            # -------------------------------------------------
            # SHOW AUGMENTED IMAGES
            # -------------------------------------------------
            st.subheader("🔍 Modified Images Used for Reliability Check")

            cols = st.columns(len(augmented_images))

            for idx, col in enumerate(cols):
                with col:
                    st.image(
                        augmented_images[idx],
                        channels="BGR",
                        width=120,
                        caption=f"Var {idx + 1}"
                    )

            # -------------------------------------------------
            # FINAL DECISION
            # -------------------------------------------------
            st.subheader("⚠️ Reliability Analysis")

            if std_conf > 0.08 or agreement < 0.85:
                st.warning("HIGH UNCERTAINTY")
                st.write(
                    "Prediction is sensitive to small variations. Manual verification recommended."
                )
            else:
                st.success("LOW UNCERTAINTY")
                st.write(
                    "Prediction is stable and considered reliable."
                )