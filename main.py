# 0_load_data.py
import pandas as pd

# Column names
col_names = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"s{i}" for i in range(1,22)]

# Load raw TXT data
train = pd.read_csv(r"data/train_FD001.txt", sep=r"\s+", header=None, names=col_names)

# ---------- Compute RUL ----------
train["RUL"] = 0
units = train["unit"].unique()

for u in units:
    engine_rows = train[train["unit"] == u]
    max_cycle = engine_rows["cycle"].max()
    train.loc[train["unit"] == u, "RUL"] = max_cycle - engine_rows["cycle"]

# Save cleaned CSV
train.to_csv(r"data/train_clean.csv", index=False)
print("✅ train_clean.csv saved successfully")
