import joblib
import streamlit as st
from pathlib import Path


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