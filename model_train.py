# 3_model_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# -----------------------------
# Load features
# -----------------------------
train = pd.read_csv(r"data/train_features.csv")

# -----------------------------
# Select feature columns
# -----------------------------
# Take all rolling and diff features for sensors
feature_cols = [col for col in train.columns if "_roll_" in col or "_diff" in col]

X = train[feature_cols]
y = train["RUL"]

# -----------------------------
# Split train/validation
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train Random Forest Regressor
# -----------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# -----------------------------
# Predict on validation
# -----------------------------
y_pred = rf.predict(X_val)

# -----------------------------
# Evaluate
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_val, y_pred))  # Manual RMSE
mae = mean_absolute_error(y_val, y_pred)

print(f"✅ Validation RMSE: {rmse:.2f}")
print(f"✅ Validation MAE: {mae:.2f}")

# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(rf, "rf_rul_model.pkl")
print("✅ Random Forest model saved as rf_rul_model.pkl")
