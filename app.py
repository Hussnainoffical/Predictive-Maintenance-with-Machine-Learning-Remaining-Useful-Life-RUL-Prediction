# 5_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("rf_rul_model.pkl")

st.title("🚀 Predictive Maintenance - RUL Prediction")
st.write("Input sensor readings for one engine cycle to predict Remaining Useful Life (RUL)")

# List of sensors
sensors = [f"s{i}" for i in range(1, 22)]

# Create input fields for sensors
input_data = {}
st.sidebar.header("Sensor Inputs")
for sensor in sensors:
    input_data[sensor] = st.sidebar.number_input(sensor, value=0.0)

# Button to predict
if st.button("Predict RUL"):
    # Convert input to DataFrame
    df_input = pd.DataFrame([input_data])

    # Feature engineering: create rolling features (simple version: use input as rolling mean/std/diff=0)
    for sensor in sensors:
        df_input[f"{sensor}_roll_mean"] = df_input[sensor]  # just the input itself
        df_input[f"{sensor}_roll_std"] = 0  # no variability in single cycle
        df_input[f"{sensor}_diff"] = 0  # difference unknown for single input

    # Select columns used in model
    feature_cols = []
    for sensor in sensors:
        feature_cols += [f"{sensor}_roll_mean", f"{sensor}_roll_std", f"{sensor}_diff"]

    # Predict
    predicted_rul = model.predict(df_input[feature_cols])[0]
    st.success(f"Predicted RUL: {predicted_rul:.2f} cycles")
