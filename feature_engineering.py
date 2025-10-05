# 2_feature_engineering.py
import pandas as pd

# Load cleaned train data
train = pd.read_csv(r"data/train_clean.csv")

sensors = [f"s{i}" for i in range(1, 22)]
window = 5

# Loop through each engine
for engine in train["unit"].unique():
    engine_data = train[train["unit"] == engine]

    for sensor in sensors:
        # Rolling mean
        train.loc[engine_data.index, f"{sensor}_roll_mean"] = engine_data[sensor].rolling(window, min_periods=1).mean()
        # Rolling std
        train.loc[engine_data.index, f"{sensor}_roll_std"] = engine_data[sensor].rolling(window,
                                                                                         min_periods=1).std().fillna(0)
        # Difference
        train.loc[engine_data.index, f"{sensor}_diff"] = engine_data[sensor].diff().fillna(0)

# Save features
train.to_csv(r"data/train_features.csv", index=False)
print("✅ train_features.csv saved successfully")
