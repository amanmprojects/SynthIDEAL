# visualize_scores.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # For checking NaN

# --- Configuration ---
CSV_FILENAME = "bayesian_calibration_scores.csv" # The file generated by calibrate_detector.py
PLOT_FILENAME = "calibration_plot.png" # Optional: filename to save the plot
CURRENT_THRESHOLD = 0.1 # The threshold you are currently using (for visualization)

# --- Load Data ---
print(f"Loading scores from {CSV_FILENAME}...")
try:
    df = pd.read_csv(CSV_FILENAME)
    print(f"Loaded {len(df)} scores.")
except FileNotFoundError:
    print(f"Error: File not found: '{CSV_FILENAME}'")
    print("Please run calibrate_detector.py first to generate the scores.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# --- Data Cleaning (Optional but recommended) ---
# Remove rows where score calculation failed (e.g., text too short)
original_count = len(df)
df.dropna(subset=['score'], inplace=True)
removed_count = original_count - len(df)
if removed_count > 0:
    print(f"Removed {removed_count} rows with NaN scores.")

if df.empty:
    print("Error: No valid scores found in the CSV file after cleaning.")
    exit()

# Separate the scores for clarity in plotting/analysis
watermarked_scores = df[df['type'] == 'watermarked']['score']
non_watermarked_scores = df[df['type'] == 'non_watermarked']['score']

print(f"\nFound {len(watermarked_scores)} watermarked scores.")
print(f"Found {len(non_watermarked_scores)} non-watermarked scores.")

if watermarked_scores.empty or non_watermarked_scores.empty:
     print("Warning: Missing scores for one or both types. Visualization might be incomplete.")


# --- Create Visualization ---
print("\nGenerating plot...")
plt.figure(figsize=(10, 6)) # Adjust figure size if needed

# Use seaborn's histplot for easy comparison with density curves
sns.histplot(data=df, x='score', hue='type', kde=True, bins=30, palette='viridis')

# Add a vertical line for the current threshold
plt.axvline(CURRENT_THRESHOLD, color='red', linestyle='--', linewidth=1, label=f'Current Threshold ({CURRENT_THRESHOLD})')

# Add titles and labels
plt.title('Distribution of Watermark Detection Scores')
plt.xlabel('Weighted Mean Score')
plt.ylabel('Density / Count')
plt.legend(title='Text Type')
plt.grid(axis='y', alpha=0.5)

# Improve layout
plt.tight_layout()

# --- Save and Show Plot ---
try:
    plt.savefig(PLOT_FILENAME)
    print(f"Plot saved as {PLOT_FILENAME}")
except Exception as e:
    print(f"Error saving plot: {e}")

print("Displaying plot...")
plt.show()

# --- Analysis Hint ---
print("\n--- How to Interpret the Plot ---")
print("1. Look at the two distributions (watermarked vs. non-watermarked).")
print("2. Notice where they overlap. This overlap zone is where misclassifications happen.")
print("3. See where the red dashed line (current threshold) falls.")
print("   - Scores to the right of the line are classified as 'watermarked'.")
print("   - Non-watermarked scores to the right are FALSE POSITIVES.")
print("   - Watermarked scores to the left are FALSE NEGATIVES.")
print("4. Decide on a *new* threshold that best separates the curves based on")
print("   whether you want to minimize false positives or false negatives more.")
print("   (Often, you'll choose a value somewhere in the overlap region).")
print("-----------------------------------")