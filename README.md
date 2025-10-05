
📖 Project Overview

This project implements a Predictive Maintenance System for turbofan engines using the NASA CMAPSS dataset.

The system predicts the Remaining Useful Life (RUL) of an engine based on multivariate sensor data. It combines end-to-end ML workflow with an interactive Streamlit dashboard for real-time prediction.

Workflow

✅ Data Preprocessing & Cleaning (load raw data → compute RUL)
✅ Exploratory Data Analysis (EDA) with sensor trends & degradation curves
✅ Feature Engineering (rolling statistics & sensor deltas)
✅ Model Training (Random Forest Regression)
✅ Model Evaluation (RMSE & MAE)
✅ Streamlit Dashboard for real-time RUL prediction

📂 Project Structure
RUL_PredictiveMaintenance/
│── data/
│   ├── train_FD001.txt           # Raw training dataset
│   ├── test_FD001.txt            # Raw test dataset
│   ├── RUL_FD001.txt             # True RUL labels for test set
│   ├── train_clean.csv           # Cleaned training data with RUL
│   ├── train_features.csv        # Feature engineered training data
│   ├── test_features.csv         # Feature engineered test data
│
│── 0_load_data.py                # Load raw data & compute RUL
│── 1_eda.py                      # Exploratory Data Analysis
│── 2_feature_engineering.py      # Feature engineering for training data
│── 2_feature_engineering_test.py # Feature engineering for test data
│── 3_model_train.py              # Train Random Forest on features
│── evaluate.py                   # Evaluate trained model on test set
│── 5_dashboard.py                # Streamlit app for RUL prediction
           



🎛️ Streamlit Dashboard

The dashboard provides a real-time interface for predictive maintenance:

📥 Upload or input live sensor readings

⚡ Instantly predict Remaining Useful Life (RUL)

📉 Monitor degradation trends interactively
