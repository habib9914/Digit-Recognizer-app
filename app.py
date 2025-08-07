import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load your trained model
model = load_model("digit_recognizer_24x24.h5")

# Page title
st.set_page_config(page_title="Habib Digit Recognizer", layout="centered")
st.title("🧠 Handwritten Digit Recognizer")
st.write("Upload a 24x24 grayscale image of a digit (0–8). Black digit on white background.")

# Upload image
uploaded_file = st.file_uploader("Choose a digit image (.png)", type=["png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = load_img(uploaded_file, color_mode='grayscale', target_size=(24, 24))
    st.image(img, caption="Uploaded Image", width=150)

    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 24, 24, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.success(f"🔢 Predicted Digit: **{predicted_digit}**")


