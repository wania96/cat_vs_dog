import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Define the custom layer
custom_objects = {'KerasLayer': hub.KerasLayer}
# Load the machine learning model
model = load_model(r'C:\Users\wania_96\Downloads\dogs-vs-cats (1)\model.h5', custom_objects=custom_objects)


# Streamlit app
st.title("Image Classification App")

# File upload widget
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Prediction logic
if uploaded_file is not None:
    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)
    
    # Resize and normalize the image
    input_image_resize = cv2.resize(image, (224, 224))
    input_image_scaled = input_image_resize / 255.0
    
    # Reshape the image
    image_reshaped = np.reshape(input_image_scaled, (1, 224, 224, 3))
    
    # Make prediction
    input_prediction = model.predict(image_reshaped)
    input_pred_label = np.argmax(input_prediction)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    # Display the prediction
    class_labels = ['Cat', 'Dog']
    st.write(f"Prediction: {class_labels[input_pred_label]}")
