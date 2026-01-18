import pandas as pd
import matplotlib.pyplot as plt
import os



#
file_path = "/Users/tina/Desktop/vitaldb_cases/case_1/aligned_120s_case1_lab1_t3060s_glu_154mgdl_50hz_filtered.csv"


if not os.path.exists(file_path):
    raise FileNotFoundError(f"csv file not found at : {file_path}")

# Read csv file
df = pd.read_csv(file_path, sep="\t")

#plot
plt.figure(figsize=(10,5))
plt.plot(df["timestamp_sec"], df["ppg_value"], label = "PPG Signal", color="blue")

plt.title("PPG Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("PPG Value")
plt.grid(True, linestyle='--',alpha = 0.7)
plt.legend()
plt.tight_layout()
plt.show()
