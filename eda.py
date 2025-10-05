# 1_eda.py
import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned train data
train = pd.read_csv(r"data/train_clean.csv")

# Choose engine to visualize
engine_id = 1
engine_data = train[train["unit"] == engine_id]

# Choose sensors to plot
sensors_to_plot = ["s1", "s2", "s3", "s4"]

plt.figure(figsize=(12, 8))
for sensor in sensors_to_plot:
    plt.plot(engine_data["cycle"], engine_data[sensor], label=sensor)
plt.xlabel("Cycle")
plt.ylabel("Sensor value")
plt.title(f"Sensor trends for Engine {engine_id}")
plt.legend()
plt.show()

# Plot RUL
plt.figure(figsize=(12, 4))
plt.plot(engine_data["cycle"], engine_data["RUL"], color="red", label="RUL")
plt.xlabel("Cycle")
plt.ylabel("RUL (Remaining Useful Life)")
plt.title(f"RUL over cycles for Engine {engine_id}")
plt.legend()
plt.show()
