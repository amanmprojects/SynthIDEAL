# main_detection_script.py

import torch
import numpy as np
import gc # Garbage collector
import time # To time operations

# Try importing official SynthID processor if available, else fallback
try:
    from transformers.generation import SynthIDLogitsProcessor
    print("Using SynthIDLogitsProcessor from transformers.")
except ImportError:
    try:
        from synthid_text import logits_processing
        SynthIDLogitsProcessor = logits_processing.SynthIDLogitsProcessor # Alias for consistency
        print("Warning: Using SynthIDLogitsProcessor from synthid_text library.")
    except ImportError:
        print("Error: Neither transformers.generation.SynthIDLogitsProcessor nor synthid_text found.")
        print("Please install/update transformers and/or synthid-text.")
        exit()


# --- Import your custom functions and initialized objects ---
# Assumes inference.py and detector_weighted_mean.py are in the same directory
try:
    # Import the specific function we need to call
    from inference import compute_generated_ids
    # Import initialized objects needed for detection logic
    from inference import tokenizer, device, watermarking_config as inference_watermarking_config
    # Import the detector function
    from detector_weighted_mean import calculate_weighted_mean_score
    print("Successfully imported components from inference.py and detector_weighted_mean.py")
except ImportError as e:
    print(f"Error importing functions or objects: {e}")
    print("Please ensure 'inference.py' and 'detector_weighted_mean.py' exist,")
    print("and inference.py defines 'tokenizer', 'device', and 'watermarking_config' globally.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    exit()


# ==============================================================
# 1. Setup: Initialize Detection Processor
#    (Model, tokenizer, device etc. are handled by inference.py)
# ==============================================================
print("\nInitializing detection components...")

# Use the watermarking config imported from inference.py
# Convert the SynthIDTextWatermarkingConfig object to a dictionary
# if the processor expects a dict (depends on which version is imported)
if hasattr(inference_watermarking_config, 'to_dict'): # Check if it's the transformers config object
     # Start from the transformers config dict, but adjust it
     temp_config = inference_watermarking_config.to_dict()
     print("Using base config dictionary converted from SynthIDTextWatermarkingConfig.")
elif isinstance(inference_watermarking_config, dict): # If inference.py used a dict
    temp_config = inference_watermarking_config.copy() # Work on a copy
    print("Using config dictionary directly from inference.py.")
else:
    # Attempt to manually create dict if type is unrecognized
    print("Warning: Imported 'watermarking_config' from inference.py is not a recognized type.")
    try:
        temp_config = {
            'keys': inference_watermarking_config.keys,
            'ngram_len': inference_watermarking_config.ngram_len,
            # Get optional fields safely
            'sampling_table_size': getattr(inference_watermarking_config, 'sampling_table_size', 65536),
            'sampling_table_seed': getattr(inference_watermarking_config, 'sampling_table_seed', 0),
            'context_history_size': getattr(inference_watermarking_config, 'context_history_size', 1024),
            'skip_first_ngram_calls': getattr(inference_watermarking_config, 'skip_first_ngram_calls', False),
            'debug_mode': getattr(inference_watermarking_config, 'debug_mode', False), # Get debug_mode if present
         }
        print("Manually created config dictionary from attributes.")
    except AttributeError as e:
         print(f"Error: Could not create config dict from imported object. Missing attribute: {e}")
         exit()

# --- ** ADJUST CONFIG FOR synthid_text PROCESSOR ** ---
CONFIG = {
    # Required fields from the imported config
    'keys': temp_config['keys'],
    'ngram_len': temp_config['ngram_len'],
    'sampling_table_size': temp_config.get('sampling_table_size', 65536), # Use .get for safety
    'sampling_table_seed': temp_config.get('sampling_table_seed', 0),
    'context_history_size': temp_config.get('context_history_size', 1024),
    'skip_first_ngram_calls': temp_config.get('skip_first_ngram_calls', False), # Accepted by synthid-text processor

    # ** Add required fields for synthid-text processor **
    'temperature': 1.0, # Default suitable for detection logic/g-value calc
    'top_k': 40,        # Default suitable for detection logic/g-value calc

    # Add device
    'device': str(device)
}

# 'debug_mode' is intentionally excluded as it's not accepted by synthid_text processor
print(f"Final CONFIG for synthid-text processor: {CONFIG.keys()}")
# --- ** END CONFIG ADJUSTMENT ** ---


# Initialize the Logits Processor needed for g-value/mask calculation
try:
    # Use the synthid-text processor alias defined earlier
    logits_processor = SynthIDLogitsProcessor(**CONFIG)
    print("SynthIDLogitsProcessor initialized for detection.")
# ... (rest of the try-except block remains the same) ...
except TypeError as e:
     print(f"Error initializing SynthIDLogitsProcessor: {e}")
     print("Check if the CONFIG dictionary keys match the processor's expected arguments.")
     print(f"Config keys used: {CONFIG.keys()}")
     # Example: Transformers version might not take 'device' in __init__
     if 'device' in str(e) and 'device' in CONFIG:
         try:
             print("Retrying without 'device' argument...")
             del CONFIG['device']
             logits_processor = SynthIDLogitsProcessor(**CONFIG)
             print("Retry: Initialized SynthIDLogitsProcessor without 'device' argument.")
         except Exception as inner_e:
              print(f"Retry failed: {inner_e}")
              exit()
     else:
         exit() # Exit if it's not a device issue we can fix here
except Exception as e:
     print(f"An unexpected error occurred initializing SynthIDLogitsProcessor: {e}")
     exit()

# ==============================================================
# (Rest of the script remains the same)
# ==============================================================

# ==============================================================
# 2. Generate Text and Get IDs using imported function
# ==============================================================

prompt = "Describe the process of photosynthesis."
print(f"\nCalling compute_generated_ids from inference.py for prompt: '{prompt}'")
start_gen_time = time.time()

# Call the function from inference.py
# It uses the model, tokenizer etc. loaded within that file's scope
try:
    # This function in your inference.py always applies the watermark
    generated_ids = compute_generated_ids(prompt) # Returns a 1D tensor
except Exception as e:
    print(f"Error during call to compute_generated_ids: {e}")
    print("Check the function implementation in inference.py and its dependencies (model loaded? etc.).")
    exit()

end_gen_time = time.time()
# Ensure IDs are on the correct device for subsequent processing
generated_ids = generated_ids.to(device)

print(f"Received {generated_ids.shape[0]} generated tokens. (Generation time: {end_gen_time - start_gen_time:.2f} seconds)")
try:
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\nDecoded Generated Text (expected to be watermarked):")
    print("-" * 30)
    print(decoded_text)
    print("-" * 30)
except Exception as e:
    print(f"Error decoding generated IDs: {e}")

# ==============================================================
# 3. Compute g-values and Mask for Detection
# ==============================================================
print("\nComputing g-values and mask for detection...")
start_detect_time = time.time()

# Add batch dimension: [seq_length] -> [1, seq_length]
generated_ids_batched = generated_ids.unsqueeze(0)

# --- Compute g-values ---
try:
    # Use the processor initialized in *this* script
    g_values = logits_processor.compute_g_values(input_ids=generated_ids_batched)
    # print(f"Computed g_values with shape: {g_values.shape}")
except Exception as e:
    print(f"Error computing g-values: {e}")
    exit()

# --- Compute Mask ---
# Ensure eos_token_id is valid
eos_token_id = tokenizer.eos_token_id
if eos_token_id is None:
    print("Error: tokenizer.eos_token_id is None. Cannot compute EOS mask.")
    # Handle error: maybe use a default ID or skip EOS masking
    exit()

try:
    # Mask tokens after the first EOS token
    eos_token_mask = logits_processor.compute_eos_token_mask(
        input_ids=generated_ids_batched,
        eos_token_id=eos_token_id,
    )
    # Align mask length with g_values, considering ngram_len
    # Ensure ngram_len is available from CONFIG
    ngram_len = CONFIG.get('ngram_len')
    if ngram_len is None:
        print("Error: 'ngram_len' not found in CONFIG. Cannot align EOS mask.")
        exit()
    eos_token_mask = eos_token_mask[:, ngram_len - 1 :]

    # Mask tokens part of repeated n-grams
    context_repetition_mask = logits_processor.compute_context_repetition_mask(
        input_ids=generated_ids_batched,
    )

    # Combine masks (both must be true for a token to be valid)
    combined_mask = context_repetition_mask * eos_token_mask
    # print(f"Computed combined_mask with shape: {combined_mask.shape}")
except Exception as e:
     print(f"Error computing mask: {e}")
     exit()

# ==============================================================
# 4. Calculate Weighted Mean Score using imported function
# ==============================================================
print("\nCalculating weighted mean score...")

# Convert tensors to NumPy arrays
g_values_np = g_values.cpu().numpy()
mask_np = combined_mask.cpu().numpy()

# Call the function from detector_weighted_mean.py
try:
    scores = calculate_weighted_mean_score(g_values_np, mask_np)
    final_score = scores[0] # Get score for the first (only) item in batch
    end_detect_time = time.time()
    print(f"(Detection logic time: {end_detect_time - start_detect_time:.2f} seconds)")
    print(f"\n>>> Weighted Mean Score: {final_score:.4f} <<<")
    print("(Higher scores suggest a higher likelihood of being watermarked with the specified config)")

    # Example threshold interpretation (adjust based on calibration)
    detection_threshold = 0.1
    if final_score > detection_threshold:
         print(f"Score > {detection_threshold}, indicating the text is likely watermarked (as expected).")
    else:
         print(f"Score <= {detection_threshold}, indicating the text might not be watermarked (unexpected for this function).")

except Exception as e:
    print(f"Error calculating weighted mean score: {e}")
    exit()

# ==============================================================
# 5. Cleanup (Optional but Recommended)
# ==============================================================
print("\nCleaning up resources...")
# We don't delete model/tokenizer here as they live in inference.py's scope now
del logits_processor, g_values, combined_mask, g_values_np, mask_np
del generated_ids, generated_ids_batched
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Cleanup finished. Script complete.")