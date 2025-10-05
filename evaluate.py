# evaluate.py
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Load test features (preprocessed)
test = pd.read_csv(r"data/test_features.csv")

# Load trained model
model = joblib.load("rf_rul_model.pkl")

# Only keep the columns used during training
feature_cols = [col for col in test.columns if "_roll_" in col or "_diff" in col]
X_test = test[feature_cols]
y_test = test["RUL"]

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Test RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Plot predicted vs actual
plt.figure(figsize=(12,4))
plt.plot(y_test.values, label="Actual RUL")
plt.plot(y_pred, label="Predicted RUL")
plt.xlabel("Test sample")
plt.ylabel("RUL")
plt.title("RUL Prediction vs Actual")
plt.legend()
plt.show()
