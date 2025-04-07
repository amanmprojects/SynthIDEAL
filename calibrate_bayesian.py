# calibrate_bayesian.py

import torch
import numpy as np
import pandas as pd
import pickle
import gc
import os
import time
# No JAX import needed here unless BayesianDetector itself needs it globally

print("--- Starting Bayesian Detector Calibration Script ---")

# --- Configuration ---
CACHE_DATA_CSV = "training_data_cache.csv" # Input data (token IDs)
TRAINED_DETECTOR_FILE = "bayesian_detector.pkl" # Input trained detector
OUTPUT_SCORES_CSV = "bayesian_calibration_scores.csv" # Output scores

# --- Import necessary components ---
print("Importing components...")
try:
    # Need tokenizer and device from inference.py
    from inference import tokenizer, device
    # Need BayesianDetector class for loading pickle
    from synthid_text import detector_bayesian

    # Check if detector file exists
    if not os.path.exists(TRAINED_DETECTOR_FILE):
        print(f"Fatal Error: Trained detector file not found: {TRAINED_DETECTOR_FILE}")
        print("Please run train_bayesian_detector.py successfully first.")
        exit()

    # Check if data cache exists
    if not os.path.exists(CACHE_DATA_CSV):
        print(f"Fatal Error: Data cache file not found: {CACHE_DATA_CSV}")
        print("Please ensure train_bayesian_detector.py generated the data cache.")
        exit()

except ImportError as e:
    print(f"Fatal Error during imports: {e}"); exit()
except Exception as e:
    print(f"An unexpected fatal error occurred during imports: {e}"); exit()


# --- Load Trained Detector ---
print(f"Loading trained Bayesian detector from {TRAINED_DETECTOR_FILE}...")
try:
    with open(TRAINED_DETECTOR_FILE, 'rb') as f:
        loaded_detector = pickle.load(f)
    if not isinstance(loaded_detector, detector_bayesian.BayesianDetector):
         print("Error: Loaded object is not a BayesianDetector instance!")
         raise RuntimeError("Loaded invalid detector file.")
    print("Trained Bayesian detector loaded successfully.")
    # Ensure the detector's internal processor is on the right device if necessary
    # (Though usually handled by the score method's input device)
    # loaded_detector.logits_processor.device = torch.device(device) # Optional check/enforcement

except Exception as e:
    print(f"Fatal Error loading trained Bayesian detector: {e}")
    exit()

# --- Load Cached Data ---
print(f"Loading cached generation data from {CACHE_DATA_CSV}...")
try:
    df_data = pd.read_csv(CACHE_DATA_CSV)
    if 'ids_str' not in df_data.columns or 'type' not in df_data.columns:
        print("Error: CSV is missing required 'ids_str' or 'type' columns.")
        exit()
    print(f"Loaded {len(df_data)} samples from cache.")
except Exception as e:
     print(f"Error loading or processing data from CSV: {e}")
     exit()

# --- Calculate Bayesian Scores ---
print(f"\nCalculating Bayesian scores for {len(df_data)} samples...")
results = []
calculation_start_time = time.time()

for index, row in df_data.iterrows():
    sample_type = row['type']
    ids_str = row['ids_str']
    current_prompt = row.get('prompt', f'Sample_{index}') # Use prompt if available, else index

    if not isinstance(ids_str, str) or not ids_str:
        print(f"Warning: Skipping row {index} due to invalid ids_str.")
        continue

    print(f"Processing sample {index+1}/{len(df_data)} (Type: {sample_type}, Prompt: '{current_prompt[:30]}...')")

    try:
        # Convert string back to 1D tensor
        token_ids = torch.tensor(list(map(int, ids_str.split(','))), dtype=torch.long).to(device)

        # Check length (use ngram_len from detector's processor)
        ngram_len = loaded_detector.logits_processor.ngram_len
        if token_ids.shape[0] < ngram_len:
             print(f"  Warning: Sample too short ({token_ids.shape[0]} tokens) for ngram_len={ngram_len}. Assigning NaN score.")
             score = np.nan
        else:
            # Add batch dimension
            token_ids_batch = token_ids.unsqueeze(0)

            # Get score from Bayesian detector (pass PyTorch tensor)
            with torch.no_grad(): # Ensure no gradients are computed
                 bayesian_scores = loaded_detector.score(token_ids_batch)
            score = float(bayesian_scores[0]) # Extract float score

        results.append({'score': score, 'type': sample_type})
        # Optional: print score as it's calculated
        # print(f"  Score: {score:.4f}")

        # Cleanup tensor
        del token_ids, token_ids_batch
        if 'bayesian_scores' in locals(): del bayesian_scores

    except Exception as e:
        print(f"  Error calculating score for sample {index+1}: {e}")
        results.append({'score': np.nan, 'type': sample_type}) # Append NaN on error

    # Periodic cleanup
    if (index + 1) % 20 == 0:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

calculation_end_time = time.time()
print(f"\nScore calculation finished. Time taken: {calculation_end_time - calculation_start_time:.2f} seconds.")

# --- Save Scores ---
print(f"\nSaving Bayesian scores to {OUTPUT_SCORES_CSV}...")
try:
    df_scores = pd.DataFrame(results)
    # Drop rows where score calculation failed before saving
    valid_scores_count = df_scores['score'].notna().sum()
    df_scores.dropna(subset=['score'], inplace=True)
    df_scores.to_csv(OUTPUT_SCORES_CSV, index=False)
    print(f"Successfully saved {valid_scores_count} valid Bayesian scores.")
except Exception as e:
    print(f"Error saving scores to CSV: {e}")

# --- Next Steps ---
print("\n--- Calibration Complete ---")
print(f"Next Steps:")
print(f"1. Analyze the scores saved in '{OUTPUT_SCORES_CSV}'.")
print(f"2. Use/Adapt 'visualize_scores.py' (update input filename to '{OUTPUT_SCORES_CSV}')")
print(f"   to plot the distributions of the *Bayesian* scores.")
print(f"3. Choose a new threshold based on the plot (scores are often log-odds, 0.0 is neutral).")
print(f"4. Update `BAYESIAN_THRESHOLD` in your `api.py` file.")
print("-----------------------------")

# --- Final Cleanup ---
print("\nCleaning up...")
try:
    del loaded_detector, df_data, df_scores, results
except NameError: pass
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("Calibration script finished.")