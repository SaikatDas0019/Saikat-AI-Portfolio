import joblib
import streamlit as st
from pathlib import Path


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