# calibrate_detector.py

import torch
import numpy as np
import transformers
import pandas as pd # Using pandas for easier data handling and CSV saving
import gc
import time
import random

print("--- Starting Calibration Script ---")

# --- Configuration ---
NUM_SAMPLES = 10 # Increase for better statistics (e.g., 200-500 if compute allows)
OUTPUT_CSV = "calibration_scores.csv"
# List of diverse prompts to avoid generating the same thing repeatedly
PROMPTS = [
    "Explain the difference between nuclear fission and fusion.",
    "Write a short story about a robot discovering music.",
    "Describe the water cycle.",
    "What are the pros and cons of renewable energy sources?",
    "Summarize the plot of Hamlet.",
    "Explain the concept of black holes.",
    "Write a poem about a rainy day in a city.",
    "What is the significance of the Rosetta Stone?",
    "Describe the process of making bread from scratch.",
    "Explain the basics of DNA and genetics.",
    # Add more diverse prompts here
]


# --- Import necessary components ---
print("Importing components...")
try:
    from inference import model, tokenizer, device, watermarking_config as inference_watermarking_config
    from detector_weighted_mean import calculate_weighted_mean_score

    # Import the correct LogitsProcessor class
    try:
        from transformers.generation import SynthIDLogitsProcessor
        print("Using SynthIDLogitsProcessor from transformers.")
    except ImportError:
        from synthid_text import logits_processing
        SynthIDLogitsProcessor = logits_processing.SynthIDLogitsProcessor
        print("Warning: Using SynthIDLogitsProcessor from synthid_text library.")

except ImportError as e:
    print(f"Fatal Error during imports: {e}")
    print("Ensure 'inference.py', 'detector_weighted_mean.py' exist and define required components.")
    exit()
except Exception as e:
    print(f"An unexpected fatal error occurred during imports: {e}")
    exit()

# --- Initialize Detection Processor ---
print("Initializing detection processor...")
detection_processor = None
detection_config = None

try:
    imported_config_obj = inference_watermarking_config
    if hasattr(imported_config_obj, 'to_dict'):
        temp_config = imported_config_obj.to_dict()
    elif isinstance(imported_config_obj, dict):
        temp_config = imported_config_obj.copy()
    else: # Fallback: try reading attributes
        temp_config = {
            'keys': imported_config_obj.keys, 'ngram_len': imported_config_obj.ngram_len,
            'sampling_table_size': getattr(imported_config_obj, 'sampling_table_size', 65536),
            'sampling_table_seed': getattr(imported_config_obj, 'sampling_table_seed', 0),
            'context_history_size': getattr(imported_config_obj, 'context_history_size', 1024),
            'skip_first_ngram_calls': getattr(imported_config_obj, 'skip_first_ngram_calls', False),
         }

    detection_config = {
        'keys': temp_config['keys'], 'ngram_len': temp_config['ngram_len'],
        'sampling_table_size': temp_config.get('sampling_table_size', 65536),
        'sampling_table_seed': temp_config.get('sampling_table_seed', 0),
        'context_history_size': temp_config.get('context_history_size', 1024),
        'skip_first_ngram_calls': temp_config.get('skip_first_ngram_calls', False),
        'temperature': 1.0, 'top_k': 40, 'device': str(device)
    }

    init_kwargs = detection_config.copy()
    try:
        detection_processor = SynthIDLogitsProcessor(**init_kwargs)
    except TypeError as e:
        if 'device' in str(e) and 'device' in init_kwargs:
            print("Retrying processor init without 'device' argument...")
            del init_kwargs['device']
            detection_processor = SynthIDLogitsProcessor(**init_kwargs)
        else: raise
    print("SynthIDLogitsProcessor initialized successfully.")

except Exception as e:
    print(f"Fatal Error initializing detection processor: {e}")
    exit()

# --- Helper Function for Non-Watermarked Generation ---
def compute_generated_ids_no_wm(prompt: str):
    """Generates text WITHOUT the watermark."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Respond to the user in 5-10 lines."},
        {"role": "user", "content": prompt}
    ]
    inputs_templated = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    # IMPORTANT: Omit 'watermarking_config' for non-watermarked generation
    generation_config_no_wm = {
        "max_new_tokens": 1000, "do_sample": True, "temperature": 0.6,
        "top_p": 0.9, "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        # NO "watermarking_config": ... line here!
    }
    with torch.no_grad():
        outputs = model.generate(inputs_templated, **generation_config_no_wm)
    generated_ids = outputs[0, inputs_templated.shape[-1]:]
    return generated_ids

# --- Helper Function for Detection Logic ---
def run_detection(generated_ids: torch.Tensor) -> float:
    """Runs the detection steps and returns the score."""
    if generated_ids.shape[0] < detection_config['ngram_len']:
        print(f"  Skipping detection: Text too short ({generated_ids.shape[0]} tokens)")
        return np.nan # Return NaN for scores that can't be computed

    generated_ids_batched = generated_ids.unsqueeze(0)
    try:
        # Compute g-values
        g_values = detection_processor.compute_g_values(input_ids=generated_ids_batched)

        # Compute Mask
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None: return np.nan # Cannot compute mask

        eos_token_mask = detection_processor.compute_eos_token_mask(
            input_ids=generated_ids_batched, eos_token_id=eos_token_id
        )[:, detection_config['ngram_len'] - 1 :]

        context_repetition_mask = detection_processor.compute_context_repetition_mask(
            input_ids=generated_ids_batched,
        )
        combined_mask = context_repetition_mask * eos_token_mask

        if not torch.any(combined_mask):
            print("  Skipping detection: No valid tokens after masking.")
            return np.nan

        # Convert to NumPy
        g_values_np = g_values.cpu().numpy()
        mask_np = combined_mask.cpu().numpy()

        # Calculate Score
        scores = calculate_weighted_mean_score(g_values_np, mask_np)
        return float(scores[0])

    except Exception as e:
        print(f"  Error during detection: {e}")
        return np.nan # Return NaN on error


# --- Main Generation and Detection Loop ---
print(f"\nStarting generation and detection for {NUM_SAMPLES} samples each...")
results = [] # Store results as dictionaries

for i in range(NUM_SAMPLES):
    start_iter_time = time.time()
    # Cycle through prompts
    current_prompt = PROMPTS[i % len(PROMPTS)]
    print(f"\n--- Sample {i+1}/{NUM_SAMPLES} ---")
    print(f"Prompt: '{current_prompt[:50]}...'")

    # 1. Generate and Detect Watermarked
    print("Generating Watermarked...")
    try:
        # Need to import the original function from inference for watermarked text
        from inference import compute_generated_ids as compute_generated_ids_wm
        wm_ids = compute_generated_ids_wm(current_prompt)
        print(f"  Generated {wm_ids.shape[0]} tokens.")
        wm_score = run_detection(wm_ids)
        if not np.isnan(wm_score):
            results.append({"score": wm_score, "type": "watermarked"})
            print(f"  Watermarked Score: {wm_score:.4f}")
        del wm_ids # Clean up tensor
    except Exception as e:
        print(f"  Error generating/detecting watermarked: {e}")

    # 2. Generate and Detect Non-Watermarked
    print("Generating Non-Watermarked...")
    try:
        no_wm_ids = compute_generated_ids_no_wm(current_prompt)
        print(f"  Generated {no_wm_ids.shape[0]} tokens.")
        no_wm_score = run_detection(no_wm_ids)
        if not np.isnan(no_wm_score):
             results.append({"score": no_wm_score, "type": "non_watermarked"})
             print(f"  Non-Watermarked Score: {no_wm_score:.4f}")
        del no_wm_ids # Clean up tensor
    except Exception as e:
        print(f"  Error generating/detecting non-watermarked: {e}")

    # Cleanup memory periodically
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    end_iter_time = time.time()
    print(f"Iteration {i+1} time: {end_iter_time - start_iter_time:.2f} seconds")


# --- Save Results ---
print(f"\nSaving results to {OUTPUT_CSV}...")
try:
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully saved {len(df)} scores.")
except Exception as e:
    print(f"Error saving results to CSV: {e}")

# --- Analysis Hint ---
print("\n--- Calibration Complete ---")
print(f"Analysis Tip: Load '{OUTPUT_CSV}' into Python (pandas/matplotlib/seaborn) or a spreadsheet.")
print("Plot histograms of the 'score' column for each 'type' ('watermarked', 'non_watermarked').")
print("Look at the score distributions to determine a suitable detection threshold.")
print("-----------------------------")