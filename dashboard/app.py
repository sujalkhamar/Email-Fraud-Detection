import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get API URL from environment variable, fallback to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Email Fraud Guard", layout="wide")

st.title("🛡️ Email Fraud Guard: Phishing & Spam Detection")
st.markdown("---")

# Sidebar for manual input
st.sidebar.header("Email Details")

def user_input_features():
    subject = st.sidebar.text_input("Subject", "Win a free iPhone now!")
    body = st.sidebar.text_area("Body", "Click here to claim your prize! Limited time offer.", height=200)
    return subject, body

subject_input, body_input = user_input_features()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Action: Analyze Email")
    
    st.markdown("**Subject:**")
    st.info(subject_input if subject_input else "(Empty)")
    st.markdown("**Body:**")
    st.info(body_input if body_input else "(Empty)")
    
    if st.button("Scan Email", type="primary"):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"subject": subject_input, "body": body_input}
            )
            
            if response.status_code == 200:
                result = response.json()
                prob = result["fraud_probability"]
                prediction = result["prediction"]
                
                if prediction == 1:
                    st.error(f"🚨 PHISHING / SPAM DETECTED! (Probability: {prob:.2%})")
                else:
                    st.success(f"✅ SAFE EMAIL (Probability: {prob:.2%})")
                
                # Visual Gauge
                fig, ax = plt.subplots(figsize=(6, 2))
                sns.barplot(x=[prob], palette="RdYlGn_r", ax=ax)
                ax.set_xlim(0, 1)
                ax.set_title("Fraud Risk Level")
                st.pyplot(fig)
            else:
                st.error("API Error: Make sure the FastAPI backend is running.")
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")

with col2:
    st.subheader("System Insights")
    st.info("The NLP model is trained on a synthetic dataset of common spam and ham emails, utilizing TF-IDF vectorization and machine learning classifiers.")
    
    st.markdown("""
    ### How it works:
    1. **Text Extraction**: The system combines the subject and body.
    2. **Vectorization**: The text is converted into numerical features using TF-IDF, evaluating word importance.
    3. **Classification**: A trained ML model determines the probability of the email being a phishing attempt or spam.
    """)

st.markdown("---")
st.caption("Email Fraud Detection System v2.0.0 | NLP Implementation")
