import streamlit as st
import pickle
import pdfplumber

# Load model and vectorizer
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Resume Screening AI")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:

    text = ""

    # extract text from PDF
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    st.subheader("Extracted Resume Text")
    st.write(text[:500])   # show first 500 characters

    if st.button("Predict Category"):

        text_vec = vectorizer.transform([text])

        prediction = model.predict(text_vec)

        st.success("Predicted Category: " + prediction[0])
        