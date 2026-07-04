import streamlit as st
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from PIL import Image

# --------------------------------------------------
# Image upload function
# --------------------------------------------------

def upload_image(uploaded_file, target_size):
    image = Image.open(uploaded_file)
    if target_size == (28, 28):
        image = image.convert("L")  # Convert to grayscale for digit recognition
    else:
        image = image.convert("RGB")  # Convert to RGB for clothing recognition
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_batch = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array, image_batch

st.title("Recognize Clothes")
st.write("Upload an image of a piece of clothing and the model will identify what it is.")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="clothes")

if uploaded_file is not None:
    image_array, _ = upload_image(uploaded_file, target_size=(28, 28))
    display_image = Image.open(uploaded_file)
    st.image(display_image, caption='Uploaded Image', use_container_width=True)
    
    st.write("")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        predict_clicked = st.button("👗 PREDICT CLOTHING", key="predict_clothes", use_container_width=True)
    with col3:
        clear_clicked = st.button("🔄 CLEAR", key="clear_clothes", use_container_width=True)
    
    if clear_clicked:
        st.rerun()
    
    if predict_clicked:
        try:
            # Load the pre-trained model
            base_dir = Path(__file__).resolve().parent
            model = load_model(base_dir / "Model" / "04_A_mnist_cnn_m2_trained_model.keras")

            image_array = np.array(image_array).reshape(1, 28, 28, 1)
            # Make prediction
            prediction = model.predict(image_array)
            predicted_class = np.argmax(prediction)
            predicted_label = class_names[predicted_class]

            st.success(f"Predicted Clothing Item: **{predicted_label}**")

        except Exception as e:
            st.error(f"Error occurred while making prediction: {e}")