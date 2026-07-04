# --------------------------------------------------
# Project: AI Portfolio Web App
# --------------------------------------------------
import streamlit as st


# Website page setup
st.set_page_config(page_title="Saikat AI Portfolio", page_icon="🦾", layout="wide")


# --------------------------------------------------
# Page content display
# --------------------------------------------------
st.title("Wellcome to the world of artificial intelligence (AI) created by Saikat Das.")
st.write("This is a prtfolio of models I've built using Machine Learning over the past few days. Click on any of the projects from the left menu to test out the live magic of AI")

st.write("---")


# Sundor kore 4 ti box ba column toyri kora.

Recognize_Number = st.button("**1. Recognize Number:**\nIf you give me a picture of any handwritten number from 0 to 9, my AI can recognize it.", use_container_width=True)
Recognize_Clothes = st.button("**2. Recognize Clothes:**\nIf you give a picture of a T-shirt, shoe, or bag, the model will scan it and tell you what it is.", use_container_width=True)
Review_Analysis = st.button("**3. Review Analysis:**\nIf you give a review of any movie or product, the model will tell you whether it is positive or negative.", use_container_width=True)
Spam_Email_Detector = st.button("**4. Spam Email Detector:**\nThe model will be able to read an email and determine whether it is a real email or a spam.", use_container_width=True)

if Recognize_Number:
    st.switch_page("pages\Recognize_Number.py")

elif Recognize_Clothes:
    st.switch_page("pages\Recognize_Clothes.py")

elif Review_Analysis:
    st.switch_page("pages\Review_Analysis.py")

elif Spam_Email_Detector:
    st.switch_page("pages\Spam_Email_Detector.py")
