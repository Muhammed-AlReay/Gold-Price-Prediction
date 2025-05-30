import numpy as np
import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "saved", "Gold_price_model.sav")
model = pickle.load(open(model_path, 'rb'))
# Streamlit app
st.set_page_config(
    page_title="Gold Price Prediction"
)
st.title("Gold Price Prediction 🚀")
st.sidebar.header("Data Inputs")


# Sidebar sliders for user input
spx = st.sidebar.slider("SPX (S&P 5000 Index)", 0.0, 5000.0, 1400.0, step=0.1)
uso = st.sidebar.slider("USO (United States Oil Fund)", 0.0, 200.0, 80.0, step=0.1)
slv = st.sidebar.slider("SLV (Silver Price)", 0.0, 50.0, 15.5, step=0.1)
eur_usd = st.sidebar.slider("EUR/USD (Euro to Dollar Exchange Rate)", 0.0, 2.0, 1.48, step=0.0001)

# Predict button
if st.sidebar.button("Predict Price"):
    # Prepare input data
    input_data = np.asarray([spx, uso, slv, eur_usd]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.success(f"Predicted Gold Price (GLD): {prediction[0]:.2f}")

    # Load feature importance from model
    feature_importance = pd.DataFrame({
        'feature': ['SPX', 'USO', 'SLV', 'EUR/USD'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Feature Importance Visualization
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='importance', y='feature', data=feature_importance, palette="viridis", ax=ax)
    ax.set_title('Feature Importance in Predicting Gold Prices (GLD)')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    st.pyplot(fig)
