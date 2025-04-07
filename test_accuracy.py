# test_accuracy.py

import torch
import numpy as np
import transformers
import gc
import time
import pickle
import random
import os
import jax.numpy as jnp # Needed if converting inside detector call (but we pass torch tensor)

print("--- Starting Detector Accuracy Test Script ---")

# --- Configuration ---
NUM_TEST_SAMPLES = 10 # Number of prompts to test
BAYESIAN_DETECTOR_FILE = "bayesian_detector.pkl"
# !!! IMPORTANT: Use the SAME threshold you calibrated and set in api.py !!!
BAYESIAN_THRESHOLD = 0.5 # Example threshold - MAKE SURE THIS MATCHES api.py
MAX_GEN_LENGTH = 500 # Max tokens to generate for test samples

# List of diverse prompts for testing (can be same as training or different)
TEST_PROMPTS = [
    "What are the main functions of the United Nations?",
    "Write a brief summary of the Industrial Revolution.",
    "Explain the concept of artificial intelligence.",
    "Describe the rules of baseball.",
    "Write a short poem about the moon.",
    "What are the health benefits of eating fruits and vegetables?",
    "Summarize the story of 'Romeo and Juliet'.",
    "Explain how GPS (Global Positioning System) works.",
    "Describe a beautiful landscape you have seen.",
    "Write a short dialogue between a cat and a dog.",
    "What caused the dinosaurs to go extinct?",
    "Explain the importance of recycling.",
    "Describe the process of cloud formation.",
    "Write a short fantasy story.",
    "What are the primary colors and secondary colors?",
    # Add more if needed, script will pick randomly
]

# --- Import necessary components ---
print("Importing components...")
try:
    # Import model components from inference.py
    # Assuming inference.py is structured to allow importing these
    from inference import model, tokenizer, device, watermarking_config
    # Import the Bayesian detector class and processor class
    from synthid_text import detector_bayesian
    try:
        from transformers.generation import SynthIDLogitsProcessor
    except ImportError:
        from synthid_text import logits_processing
        SynthIDLogitsProcessor = logits_processing.SynthIDLogitsProcessor # Alias

    # Check if detector file exists
    if not os.path.exists(BAYESIAN_DETECTOR_FILE):
        print(f"Fatal Error: Trained detector file not found: {BAYESIAN_DETECTOR_FILE}")
        exit()

except ImportError as e:
    print(f"Fatal Error during imports: {e}"); exit()
except Exception as e:
    print(f"An unexpected fatal error occurred during imports: {e}"); exit()

# --- Load Trained Detector ---
print(f"Loading trained Bayesian detector from {BAYESIAN_DETECTOR_FILE}...")
loaded_detector = None
try:
    with open(BAYESIAN_DETECTOR_FILE, 'rb') as f:
        loaded_detector = pickle.load(f)
    if not isinstance(loaded_detector, detector_bayesian.BayesianDetector):
         raise RuntimeError("Loaded object is not a BayesianDetector instance!")
    print("Trained Bayesian detector loaded successfully.")
    # Ensure internal components use the correct device
    # loaded_detector.logits_processor.device = torch.device(device) # Usually handled if detector loaded correctly
    # loaded_detector.detector_module... # May need device placement if it uses JAX/Flax directly - check library details if issues arise
except Exception as e:
    print(f"Fatal Error loading trained Bayesian detector: {e}")
    exit()

# --- Helper Function for Non-Watermarked Generation ---
def compute_generated_ids_no_wm(prompt: str):
    """Generates text WITHOUT the watermark."""
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": prompt}]
    inputs_templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    generation_config_no_wm = {
        "max_new_tokens": MAX_GEN_LENGTH, "do_sample": True, "temperature": 0.6, "top_p": 0.9,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    with torch.no_grad(): outputs = model.generate(inputs_templated, **generation_config_no_wm)
    generated_ids = outputs[0, inputs_templated.shape[-1]:]; return generated_ids

# --- Helper Function for Detection ---
def detect(token_ids: torch.Tensor) -> tuple[bool, float]:
    """Runs detection using the loaded Bayesian detector and returns (detected_bool, score)."""
    if loaded_detector is None:
        print("Error: Detector not loaded.")
        return False, -999.0 # Indicate error

    # Need access to ngram_len, get from detector's processor
    ngram_len = loaded_detector.logits_processor.ngram_len
    if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1 or token_ids.shape[0] < ngram_len:
        print(f"  Skipping detection: Invalid input tensor or too short ({token_ids.shape[0]} tokens, need >={ngram_len}).")
        return False, -999.0 # Return sentinel score for invalid input

    token_ids_batch = token_ids.unsqueeze(0).to(device) # Ensure on correct device & add batch dim

    try:
        # Pass PyTorch tensor directly based on last fix
        with torch.no_grad():
            bayesian_scores = loaded_detector.score(token_ids_batch)
        score = float(bayesian_scores[0])
        detected = score > BAYESIAN_THRESHOLD
        return detected, score
    except Exception as e:
        print(f"  Error during detection call: {e}")
        return False, -999.0 # Return sentinel score on error

# --- Main Testing Loop ---
print(f"\nStarting accuracy test for {NUM_TEST_SAMPLES} random prompts...")
correct_predictions = 0
total_predictions = 0
test_start_time = time.time()

# Ensure we have enough unique prompts if NUM_TEST_SAMPLES > len(TEST_PROMPTS)
if NUM_TEST_SAMPLES > len(TEST_PROMPTS):
    print(f"Warning: Only {len(TEST_PROMPTS)} unique prompts available, testing will reuse prompts.")
    selected_prompts = random.choices(TEST_PROMPTS, k=NUM_TEST_SAMPLES)
else:
    selected_prompts = random.sample(TEST_PROMPTS, NUM_TEST_SAMPLES)


for i, current_prompt in enumerate(selected_prompts):
    print(f"\n--- Test Sample {i+1}/{NUM_TEST_SAMPLES} ---")
    print(f"Prompt: '{current_prompt[:60]}...'")

    wm_ids = None
    nwm_ids = None
    wm_detected = False
    wm_score = np.nan
    nwm_detected = True # Default to incorrect for calculation if detection fails
    nwm_score = np.nan

    # 1. Generate and Detect Watermarked
    print("Generating Watermarked...")
    gen_wm_start = time.time()
    try:
        # Import the WM generation function from inference.py
        from inference import compute_generated_ids as compute_generated_ids_wm
        wm_ids = compute_generated_ids_wm(current_prompt)
        gen_wm_end = time.time()
        print(f"  Generated {wm_ids.shape[0]} tokens (Time: {gen_wm_end - gen_wm_start:.2f}s)")
        wm_detected, wm_score = detect(wm_ids)
        total_predictions += 1
        if wm_detected: # Ground truth is WM, detection is True
            correct_predictions += 1
            print(f"  Detection Result (WM): CORRECT (Detected, Score: {wm_score:.4f})")
        else:
            print(f"  Detection Result (WM): INCORRECT (Not Detected, Score: {wm_score:.4f})")
    except Exception as e:
        print(f"  Error during WM generation/detection: {e}")


    # 2. Generate and Detect Non-Watermarked
    print("Generating Non-Watermarked...")
    gen_nwm_start = time.time()
    try:
        nwm_ids = compute_generated_ids_no_wm(current_prompt)
        gen_nwm_end = time.time()
        print(f"  Generated {nwm_ids.shape[0]} tokens (Time: {gen_nwm_end - gen_nwm_start:.2f}s)")
        nwm_detected, nwm_score = detect(nwm_ids)
        total_predictions += 1
        if not nwm_detected: # Ground truth is non-WM, detection is False
            correct_predictions += 1
            print(f"  Detection Result (Non-WM): CORRECT (Not Detected, Score: {nwm_score:.4f})")
        else:
             print(f"  Detection Result (Non-WM): INCORRECT (Detected, Score: {nwm_score:.4f})")
    except Exception as e:
        print(f"  Error during Non-WM generation/detection: {e}")

    # Cleanup
    del wm_ids, nwm_ids
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# --- Calculate and Report Accuracy ---
test_end_time = time.time()
print(f"\n--- Test Finished ---")
print(f"Total time: {test_end_time - test_start_time:.2f} seconds")
if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Correct Predictions: {correct_predictions} / {total_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("No predictions were made.")

# --- Final Cleanup ---
print("\nCleaning up...")
try:
    del model, tokenizer, loaded_detector
except NameError: pass
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("Accuracy test script finished.")