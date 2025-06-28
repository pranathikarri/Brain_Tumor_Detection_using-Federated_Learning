import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("trained_global_model.h5")

# Title
st.title("🧠 Brain Tumor Detection from MRI (Grayscale Model)")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='🖼 Uploaded MRI', use_column_width=True)

    # Preprocess image: convert to grayscale, resize, normalize
    img = img.convert('L')  # 'L' mode = grayscale
    img = img.resize((64, 64))
    img_array = np.array(img)

    img_array = img_array.reshape(1, 64, 64, 1)  # (1, 64, 64, 1)
    img_array = img_array / 255.0  # Normalize to 0-1

    # Predict
    predictions = model.predict(img_array)
    tumor_prob = predictions[0][1]
    no_tumor_prob = predictions[0][0]

    # Show prediction result with confidence threshold
    if tumor_prob > 0.6:
        st.error(f"⚠️ Tumor Detected with {tumor_prob*100:.2f}% confidence.")
    else:
        st.success(f"✅ No Tumor Detected with {no_tumor_prob*100:.2f}% confidence.")

    # Show raw prediction scores
    st.write("### 🔍 Prediction Scores")
    st.write(f"**No Tumor:** {no_tumor_prob:.4f}")
    st.write(f"**Tumor:** {tumor_prob:.4f}")
