import streamlit as st
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("potatoes.keras")

# Define class names
class_names = ['early blight', 'healthy', 'late blight']
IMAGE_SIZE = 256
# Function to preprocess image
def preprocess_image(image):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    return image

# Function to make prediction
def predict_leaf_type(image):
    # Preprocess the image
    image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(np.expand_dims(image, axis=0))

    # Get predicted class and confidence
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

# Streamlit app
st.title("Potato Leaf Disease Classifier")

uploaded_file = st.file_uploader("Upload an image of a potato leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = tf.image.decode_image(uploaded_file.read(), channels=3)
    image = np.array(image)  # Convert TensorFlow tensor to NumPy array
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    predicted_class, confidence = predict_leaf_type(image)

    # Display prediction
    st.write("Predicted Class:", predicted_class)
    st.write("Confidence:", confidence, "%")
