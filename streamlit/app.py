import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cifar10_mobilenet_final.h5")  # Ensure model.h5 is in the same directory
    return model

model = load_model()

# CIFAR-10 class labels
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.title("Image Classification with CIFAR-10")
st.write("Upload an image to classify it using a pre-trained model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load and preprocess image
    image = Image.open(uploaded_file).resize((32, 32))  # Resize to 32x32 (CIFAR-10)
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    st.subheader(f"Prediction: **{predicted_class}**")
