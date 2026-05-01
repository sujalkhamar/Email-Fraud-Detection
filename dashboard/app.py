import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Fraud Guard Dashboard", layout="wide")

st.title("🛡️ Fraud Guard: Real-Time Detection System")
st.markdown("---")

# Sidebar for manual input
st.sidebar.header("Transaction Features")

def user_input_features():
    features = []
    # 28 PCA components
    for i in range(1, 29):
        val = st.sidebar.slider(f"V{i}", -5.0, 5.0, 0.0)
        features.append(val)
    
    amount = st.sidebar.number_input("Amount ($)", min_value=0.0, value=100.0)
    time = st.sidebar.number_input("Time (seconds)", min_value=0.0, value=0.0)
    
    features.append(amount)
    features.append(time)
    return features

input_data = user_input_features()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Action: Analyze Transaction")
    if st.button("Predict Fraud"):
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"features": input_data}
            )
            
            if response.status_code == 200:
                result = response.json()
                prob = result["fraud_probability"]
                prediction = result["prediction"]
                
                if prediction == 1:
                    st.error(f"🚨 FRAUD DETECTED! (Probability: {prob:.2%})")
                else:
                    st.success(f"✅ LEGITIMATE TRANSACTION (Probability: {prob:.2%})")
                
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
    st.info("The model is trained on a synthetic dataset resembling real-world transaction patterns.")
    
    # Placeholder for some data insights
    data_points = np.random.randn(100, 2)
    df_chart = pd.DataFrame(data_points, columns=['Feature A', 'Feature B'])
    st.line_chart(df_chart)

st.markdown("---")
st.caption("Fraud Detection System v1.0.0 | Clean Architecture Implementation")
