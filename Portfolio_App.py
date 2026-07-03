# --------------------------------------------------
# Project: AI Portfolio Web App
# --------------------------------------------------
from pathlib import Path

import joblib
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Website page setup
st.set_page_config(page_title="Saikat AI Portfolio", page_icon="🦾", layout="wide")

# Create menu
st.sidebar.title("Menu")
st.sidebar.write("Choose any project:")

# Option of menu
menu_options = [
    "Home",
    "Recognize Number",
    "Recognize Clothes",
    "Spam Email Detector",
    "Review Analysis"
]

choice = st.sidebar.radio("Project List:", menu_options)

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

# --------------------------------------------------
# Load the Model and Vectorizer (Review Analysis)
# --------------------------------------------------
@st.cache_resource  # er fale protiber pag refrash hole model bar bar load hoya app slow na hoya jai.
def load_review_analysis_models():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "Model" / "01_a_Clothing_sentiment_svm_model.joblib"
    vectorizer_path = base_dir / "Model" / "01_b_Clothing_vectorizer.joblib"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model_review_analysis, vectorizer_review_analysis = load_review_analysis_models()


# --------------------------------------------------
# Load the Model and Vectorizer (Spam Email Detector)
# --------------------------------------------------
@st.cache_resource  # er fale protiber pag refrash hole model bar bar load hoya app slow na hoya jai.
def load_spam_email_detector_models():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "Model" / "02_a_spam_classifier_model.joblib"
    vectorizer_path = base_dir / "Model" / "02_b_tfidf_vectorizer.joblib"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model_spam_email_detector, vectorizer_spam_email_detector = load_spam_email_detector_models()


# --------------------------------------------------
# Page content display
# --------------------------------------------------
if choice == "Home":
    st.title("Wellcome to the world of artificial intelligence (AI) created by Saikat Das.")
    st.write("This is a prtfolio of models I've built using Machine Learning over the past few days. Click on any of the projects from the left menu to test out the live magic of AI")

    st.write("---")


    # Sundor kore 4 ti box ba column toyri kora.
    col1, col2 = st.columns(2)

    with col1:
        st.info("**1. Recognize Number:**\nIf you give me a picture of any handwritten number from 0 to 9, my AI can recognize it.")
        st.success("**2. Recognize Clothes:**\nIf you give a picture of a T-shirt, shoe, or bag, the model will scan it and tell you what it is.")

    with col2:
        st.warning("**3. Review Analysis:**\nIf you give a review of any movie or product, the model will tell you whether it is positive or negative.")
        st.error("**4. Spam Email Detector:**\nThe model will be able to read an email and determine whether it is a real email or a spam.")

elif choice == "Recognize Number":
    st.title("Recognize Number")
    st.write("Upload an image of a handwritten digit (0-9) and the model will predict the number.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key= "mnist")

    if uploaded_file is not None:
        image_array, original_image = upload_image(uploaded_file, target_size=(28, 28))
        st.image(original_image, caption='Uploaded Image', use_column_width=True, width=150)
        
        st.write("")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            predict_clicked = st.button("🔮 PREDICT NUMBER", key="predict_mnist", use_container_width=True)
        with col3:
            clear_clicked = st.button("🔄 CLEAR", key="clear_mnist", use_container_width=True)
        
        if clear_clicked:
            st.rerun()
        
        if predict_clicked:
            try:
                # Load the pre-trained model
                base_dir = Path(__file__).resolve().parent
                model = load_model(base_dir / "Model" / "03_A_mnist_cnn_m1_trained_model.keras")

                # Make prediction
                prediction = model.predict(image_array)
                predicted_class = np.argmax(prediction)

                st.success(f"Predicted Number: **{predicted_class}**")

            except Exception as e:
                st.error(f"Error occurred while making prediction: {e}")

elif choice == "Recognize Clothes":
    st.title("Recognize Clothes")
    st.write("Upload an image of a piece of clothing and the model will identify what it is.")

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="clothes")

    if uploaded_file is not None:
        image_array, original_image = upload_image(uploaded_file, target_size=(28, 28))
        st.image(original_image, caption='Uploaded Image', use_column_width=True, width=150)
        
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

                # Make prediction
                prediction = model.predict(image_array)
                predicted_class = np.argmax(prediction)
                predicted_label = class_names[predicted_class]

                st.success(f"Predicted Clothing Item: **{predicted_label}**")

            except Exception as e:
                st.error(f"Error occurred while making prediction: {e}")

elif choice == "Review Analysis":
    st.title("Review Analysis")
    st.write("Upload a review of any movie or product and the model will determine if it is positive or negative.")
    user_review = st.text_area("Enter your review:", height=150)
    
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button("⭐ ANALYZE SENTIMENT", key="analyze_sentiment", use_container_width=True)

    if analyze_clicked:
        if user_review.strip() == "":
            st.warning("Please, Enter your review.")
        else:
            review_tfidf = vectorizer_review_analysis.transform([user_review])
            prediction = model_review_analysis.predict(review_tfidf)[0]

            st.write("---")
            st.subheader("Result")

            if prediction == 1:
                st.success("**POSITIVE SENTIMENT!** Customer like this Product.")
                st.balloons()
            else:
                st.error("**NEGATIVE SENTIMENT!** Customer don't like this Product.")

elif choice == "Spam Email Detector":
    st.title("Spam Email Detector")
    st.write("Upload an email and the model will determine if it is a real email or a spam.")
    email = st.text_area("Enter your email:", height=150)
    
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button("📧 ANALYZE EMAIL", key="analyze_email", use_container_width=True)
    
    if analyze_clicked:
        if email.strip() == "":
            st.warning("Please, enter your email.")
        else:
            review_tfidf = vectorizer_spam_email_detector.transform([email])
            prediction = model_spam_email_detector.predict(review_tfidf)[0]

            st.write("---")
            st.subheader("Result")

            if prediction == 1:
                st.error("**SPAM EMAIL!** This email is classified as spam.")
            else:
                st.success("**LEGITIMATE EMAIL!** This email is not spam.")
                st.balloons()
