# 2_feature_engineering_test.py
import pandas as pd

# Column names
col_names = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"s{i}" for i in range(1,22)]

# Load raw test TXT data
test = pd.read_csv(r"data/test_FD001.txt", sep=r"\s+", header=None, names=col_names)

# Load true RULs for test set (optional, for evaluation)
rul = pd.read_csv(r"data/RUL_FD001.txt", header=None, names=["RUL"])
# Assign RUL to each engine
engines = test["unit"].unique()
for i, u in enumerate(engines):
    max_cycle = test[test["unit"]==u]["cycle"].max()
    test.loc[test["unit"]==u, "RUL"] = rul.loc[i, "RUL"]

# Feature engineering (same as train)
sensors = [f"s{i}" for i in range(1,22)]
window = 5

for engine in test["unit"].unique():
    engine_data = test[test["unit"] == engine]
    for sensor in sensors:
        test.loc[engine_data.index, f"{sensor}_roll_mean"] = engine_data[sensor].rolling(window, min_periods=1).mean()
        test.loc[engine_data.index, f"{sensor}_roll_std"] = engine_data[sensor].rolling(window, min_periods=1).std().fillna(0)
        test.loc[engine_data.index, f"{sensor}_diff"] = engine_data[sensor].diff().fillna(0)

# Save test_features.csv
test.to_csv(r"data/test_features.csv", index=False)
print("✅ test_features.csv saved successfully")
