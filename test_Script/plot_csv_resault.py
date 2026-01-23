import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load your results
try:
    df = pd.read_csv('transmission_results_V2.csv')
    # Filter only successful sends for the graph
    df = df[df['Status'] == 'Success']
except FileNotFoundError:
    print("Error: transmission_results.csv not found.")
    exit()

# Create the figure with two subplots
plt.figure(figsize=(12, 5))

# --- Plot 1: Size vs Time (The "Pente") ---
plt.subplot(1, 2, 1)
plt.scatter(df['Size_KB'], df['Time_Seconds'], color='blue', alpha=0.6, label='Individual Images')

# Calculate the trend line (linear regression)
m, b = np.polyfit(df['Size_KB'], df['Time_Seconds'], 1)
plt.plot(df['Size_KB'], m*df['Size_KB'] + b, color='red', label=f'Trend (Slope: {m:.4f})')

plt.title('Impact of Image Size on Transmission Time V2')
plt.xlabel('Size (KB)')
plt.ylabel('Time (Seconds)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# --- Plot 2: Speed Distribution (Histogram) ---
plt.subplot(1, 2, 2)
# Speed = Size / Time
df['Speed_KB_s'] = df['Size_KB'] / df['Time_Seconds']

plt.hist(df['Speed_KB_s'], bins=10, color='skyblue', edgecolor='black')
plt.title('Network Stability (Transmission Speed)')
plt.xlabel('Speed (KB/s)')
plt.ylabel('Number of Images')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('pfe_results_analysisV2.png')
print("Analysis plots saved as 'pfe_results_analysis.png'")
