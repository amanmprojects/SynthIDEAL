# train_bayesian_detector.py (Corrected with Caching, JAX fixes, and removed redundant padding)

import torch
import numpy as np
import transformers
import gc
import time
import pickle
import random
import pandas as pd # For CSV handling
import os         # For file existence check
import jax        # Import JAX

# --- JAX Device Check ---
# Run this early to confirm JAX setup before proceeding
print("\n--- Checking JAX Devices ---")
try:
    print(f"JAX version: {jax.__version__}")
    available_devices = jax.devices()
    print(f"JAX available devices: {available_devices}")
    gpu_devices = [d for d in available_devices if 'gpu' in d.device_kind.lower()]
    if gpu_devices:
        print(">>> JAX correctly detected GPU(s)! <<<")
    else:
        print(">>> WARNING: JAX did NOT detect any GPU devices! Ensure JAX is installed with CUDA support. <<<")
        # Optionally exit if GPU is strictly required
        # exit()
except ImportError:
    print("Fatal Error: JAX library not found. It's required for Bayesian training.")
    exit()
except Exception as e:
    print(f"Fatal Error checking JAX devices: {e}")
    exit()
print("--- End JAX Device Check ---\n")

print("--- Starting Bayesian Detector Training Script (with Caching) ---")

# --- Configuration ---
NUM_SAMPLES = 200 # Target number of samples for EACH type (WM and non-WM) - Increase if possible!
OUTPUT_DETECTOR_FILE = "bayesian_detector.pkl"
GENERATED_DATA_CSV = "training_data_cache.csv" # Cache file
# Training parameters
POS_TRUNCATION_LENGTH = 150
NEG_TRUNCATION_LENGTH = 150
MAX_PADDED_LENGTH = 500 # Max length for GENERATION and internal padding
N_EPOCHS = 20
LEARNING_RATE = 3e-3

# List of diverse prompts
PROMPTS = [
    "Explain the difference between nuclear fission and fusion.", "Write a short story about a robot discovering music.",
    "Describe the water cycle.", "What are the pros and cons of renewable energy sources?",
    "Summarize the plot of Hamlet.", "Explain the concept of black holes.",
    "Write a poem about a rainy day in a city.", "What is the significance of the Rosetta Stone?",
    "Describe the process of making bread from scratch.", "Explain the basics of DNA and genetics.",
    "How does photosynthesis work?", "Compare and contrast Python and Java.",
    "Write a travel blog post about visiting Tokyo.", "Explain the theory of relativity simply.",
    "Describe how the Internet works in simple terms.", "Write a dialogue between a human and an alien.",
    "List the causes and effects of global warming.", "Explain blockchain technology simply.",
    "Write a guide on how to start a vegetable garden.", "Describe the structure of the human brain.",
    "What is the importance of the moon to Earth?", "Write a mystery story set in a library.",
    "Explain machine learning in easy words.", "Describe the history of the Great Wall of China.",
    "List some benefits of learning a second language.", "Compare the cultures of India and Japan.",
    "Explain how electric cars work.", "Describe the life cycle of a butterfly.",
    "Write a story about a lost treasure map.", "Explain the basics of quantum computing.",
    "Describe a futuristic city powered entirely by renewable energy.", "Create a character sketch of a time-traveling detective.",
    "Explain the rules and strategy of chess.", "Describe how coffee is made from bean to cup.",
    "Write a motivational speech for students.", "Explain the history and origin of the Olympic Games.",
    "List advantages and disadvantages of social media.", "Describe a day in the life of a honeybee.",
    "Explain why sleep is important for humans.", "Write a guide on how to stay productive while working from home.",
    "Describe the working of a hydroelectric power plant.", "Write a story where animals can talk to humans.",
    "Explain what virtual reality is and how it works.", "List tips for saving money as a student.",
    "Describe the structure of the solar system.", "Explain the concept of supply and demand in economics.",
    "Describe how to write a good resume.", "Write a comedy script about aliens visiting Earth.",
    "Explain the benefits of meditation.", "Describe the history of ancient Egypt.",
    "Explain the process of recycling plastic waste.", "Describe what makes a good leader.",
    "Explain how the human digestive system works.", "Write a review of your favorite movie.",
    "Describe how volcanoes are formed.", "Write a letter to your future self.",
    "Describe a utopian world without any crime.", "Write a survival guide for living on Mars.",
    "Explain the basics of 3D printing technology.", "Describe the process of election in a democratic country.",
    "Write an essay about kindness and empathy.", "Explain the role of Artificial Intelligence in healthcare.",
    "Write a horror story set in a deserted village.", "Describe the parts and functions of a cell.",
    "Write an imaginary interview with a famous scientist.", "Explain the causes of the French Revolution.",
    "Describe the working of a smart home system."
]


# --- Import necessary components ---
print("Importing components...")
try:
    # Import model components from inference.py
    from inference import model, tokenizer, device, watermarking_config as inference_watermarking_config
    # Import the Bayesian detector trainer
    from synthid_text import detector_bayesian
    # Import the LogitsProcessor
    try:
        from transformers.generation import SynthIDLogitsProcessor
        print("Using SynthIDLogitsProcessor from transformers.")
    except ImportError:
        from synthid_text import logits_processing
        SynthIDLogitsProcessor = logits_processing.SynthIDLogitsProcessor
        print("Warning: Using SynthIDLogitsProcessor from synthid_text library.")

except ImportError as e:
    print(f"Fatal Error during imports: {e}"); exit()
except Exception as e:
    print(f"An unexpected fatal error occurred during imports: {e}"); exit()

# --- Initialize Logits Processor ---
print("Initializing logits processor...")
detection_processor = None
detection_config = None
try:
    imported_config_obj = inference_watermarking_config
    if hasattr(imported_config_obj, 'to_dict'): temp_config = imported_config_obj.to_dict()
    elif isinstance(imported_config_obj, dict): temp_config = imported_config_obj.copy()
    else: temp_config = {
            'keys': imported_config_obj.keys, 'ngram_len': imported_config_obj.ngram_len,
            'sampling_table_size': getattr(imported_config_obj, 'sampling_table_size', 65536),
            'sampling_table_seed': getattr(imported_config_obj, 'sampling_table_seed', 0),
            'context_history_size': getattr(imported_config_obj, 'context_history_size', 1024),
            'skip_first_ngram_calls': getattr(imported_config_obj, 'skip_first_ngram_calls', False),}

    ngram_len_to_use = 5 # Make sure this matches the config used in inference.py
    print(f"Using ngram_len = {ngram_len_to_use} for detection processor.")

    detection_config = {'keys': temp_config['keys'], 'ngram_len': ngram_len_to_use,
        'sampling_table_size': temp_config.get('sampling_table_size', 65536),
        'sampling_table_seed': temp_config.get('sampling_table_seed', 0),
        'context_history_size': temp_config.get('context_history_size', 1024),
        'skip_first_ngram_calls': temp_config.get('skip_first_ngram_calls', False),
        'temperature': 1.0, 'top_k': 40, 'device': str(device)}
    init_kwargs = detection_config.copy()
    try: detection_processor = SynthIDLogitsProcessor(**init_kwargs)
    except TypeError as e:
        if 'device' in str(e) and 'device' in init_kwargs:
            del init_kwargs['device']; detection_processor = SynthIDLogitsProcessor(**init_kwargs)
        else: raise
    print("Logits processor initialized successfully.")
except Exception as e: print(f"Fatal Error initializing detection processor: {e}"); exit()


# --- Helper Function for Non-Watermarked Generation ---
def compute_generated_ids_no_wm(prompt: str):
    """Generates text WITHOUT the watermark."""
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": prompt}]
    inputs_templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    generation_config_no_wm = {"max_new_tokens": MAX_PADDED_LENGTH, "do_sample": True, "temperature": 0.6, "top_p": 0.9, "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id}
    with torch.no_grad(): outputs = model.generate(inputs_templated, **generation_config_no_wm)
    generated_ids = outputs[0, inputs_templated.shape[-1]:]; return generated_ids

# --- Load Existing Data / Determine Needed Samples ---
print(f"\nChecking cache file: {GENERATED_DATA_CSV}")
if os.path.exists(GENERATED_DATA_CSV):
    try:
        df_existing = pd.read_csv(GENERATED_DATA_CSV)
        if 'type' not in df_existing.columns or 'ids_str' not in df_existing.columns:
             print("Warning: CSV malformed. Starting fresh.")
             df_existing = pd.DataFrame(columns=['prompt', 'ids_str', 'type'])
    except pd.errors.EmptyDataError:
         print("Cache file is empty. Starting fresh.")
         df_existing = pd.DataFrame(columns=['prompt', 'ids_str', 'type'])
    except Exception as e:
        print(f"Error reading cache file '{GENERATED_DATA_CSV}': {e}. Starting fresh.")
        df_existing = pd.DataFrame(columns=['prompt', 'ids_str', 'type'])
else:
    print("Cache file not found. Starting fresh.")
    df_existing = pd.DataFrame(columns=['prompt', 'ids_str', 'type'])

existing_wm_count = len(df_existing[df_existing['type'] == 'watermarked']) if 'type' in df_existing.columns else 0
existing_nwm_count = len(df_existing[df_existing['type'] == 'non_watermarked']) if 'type' in df_existing.columns else 0

needed_wm = max(0, NUM_SAMPLES - existing_wm_count)
needed_nwm = max(0, NUM_SAMPLES - existing_nwm_count)

print(f"Found {existing_wm_count} existing watermarked samples.")
print(f"Found {existing_nwm_count} existing non-watermarked samples.")
print(f"Need to generate {needed_wm} more watermarked samples.")
print(f"Need to generate {needed_nwm} more non-watermarked samples.")

# --- Generate Missing Training Data ---
new_data_rows = []
overall_generation_start_time = time.time()

if needed_wm > 0 or needed_nwm > 0:
    print(f"\nGenerating {needed_wm + needed_nwm} total new samples...")
    max_needed = max(needed_wm, needed_nwm)
    current_wm_generated = 0
    current_nwm_generated = 0

    # Use a set to track prompts used in this run to ensure more variety if needed
    prompts_this_run = set()
    prompt_indices = list(range(len(PROMPTS)))
    random.shuffle(prompt_indices) # Shuffle prompts for better variety
    prompt_idx_counter = 0

    while current_wm_generated < needed_wm or current_nwm_generated < needed_nwm:
        # Ensure we don't loop infinitely if prompts run out or generation fails repeatedly
        if prompt_idx_counter >= len(PROMPTS) and (current_wm_generated < needed_wm or current_nwm_generated < needed_nwm):
             print("Warning: Ran out of unique prompts but still need samples. Reusing prompts.")
             # Reset counter or break depending on desired behavior
             prompt_idx_counter = 0 # Allow reuse
             # Alternatively, break if reuse is not desired and generation failed for some

        prompt_list_index = prompt_indices[prompt_idx_counter % len(PROMPTS)]
        current_prompt = PROMPTS[prompt_list_index]
        prompt_idx_counter += 1

        print(f"\n--- Generating for Prompt Index {prompt_list_index} ---")
        print(f"Prompt: '{current_prompt[:50]}...'")

        # Generate Watermarked if needed
        if current_wm_generated < needed_wm:
            print("Generating Watermarked...")
            gen_wm_start = time.time()
            try:
                from inference import compute_generated_ids as compute_generated_ids_wm
                wm_ids = compute_generated_ids_wm(current_prompt)
                gen_wm_end = time.time(); gen_wm_duration = gen_wm_end - gen_wm_start
                ids_list = wm_ids.cpu().tolist()
                # Basic check for valid generation (not just EOS/PAD)
                if len(ids_list) > ngram_len_to_use:
                    ids_str = ",".join(map(str, ids_list))
                    new_data_rows.append({'prompt': current_prompt, 'ids_str': ids_str, 'type': 'watermarked'})
                    current_wm_generated += 1
                    print(f"  WM sample {existing_wm_count + current_wm_generated}/{NUM_SAMPLES} generated ({len(ids_list)} tokens) - Time: {gen_wm_duration:.2f}s")
                else:
                    print(f"  Warning: Generated WM sample too short ({len(ids_list)} tokens). Discarding.")
                del wm_ids, ids_list
                if 'ids_str' in locals(): del ids_str
            except Exception as e:
                gen_wm_end = time.time(); gen_wm_duration = gen_wm_end - gen_wm_start
                print(f"  Error generating watermarked sample: {e} (Time: {gen_wm_duration:.2f}s)")

        # Generate Non-Watermarked if needed
        if current_nwm_generated < needed_nwm:
            print("Generating Non-Watermarked...")
            gen_nwm_start = time.time()
            try:
                no_wm_ids = compute_generated_ids_no_wm(current_prompt)
                gen_nwm_end = time.time(); gen_nwm_duration = gen_nwm_end - gen_nwm_start
                ids_list = no_wm_ids.cpu().tolist()
                # Basic check for valid generation
                if len(ids_list) > ngram_len_to_use:
                    ids_str = ",".join(map(str, ids_list))
                    new_data_rows.append({'prompt': current_prompt, 'ids_str': ids_str, 'type': 'non_watermarked'})
                    current_nwm_generated += 1
                    print(f"  Non-WM sample {existing_nwm_count + current_nwm_generated}/{NUM_SAMPLES} generated ({len(ids_list)} tokens) - Time: {gen_nwm_duration:.2f}s")
                else:
                     print(f"  Warning: Generated Non-WM sample too short ({len(ids_list)} tokens). Discarding.")
                del no_wm_ids, ids_list
                if 'ids_str' in locals(): del ids_str
            except Exception as e:
                gen_nwm_end = time.time(); gen_nwm_duration = gen_nwm_end - gen_nwm_start
                print(f"  Error generating non-watermarked sample: {e} (Time: {gen_nwm_duration:.2f}s)")

        # Periodic cleanup
        if prompt_idx_counter % 5 == 0:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    overall_generation_end_time = time.time()
    print(f"\nNew data generation finished. Total time: {overall_generation_end_time - overall_generation_start_time:.2f} seconds.")

    # --- Append New Data and Save ---
    if new_data_rows:
        print(f"Appending {len(new_data_rows)} new rows to cache...")
        df_new = pd.DataFrame(new_data_rows)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        try:
            # Overwrite the file with the combined data
            df_combined.to_csv(GENERATED_DATA_CSV, index=False)
            print(f"Successfully updated cache file: {GENERATED_DATA_CSV}")
            df_existing = df_combined # Update reference
        except Exception as e:
            print(f"Error saving updated cache file: {e}")
else:
    print("Sufficient samples found in cache. Skipping generation.")

# --- Load Data for Training ---
print(f"\nLoading final data for training from {GENERATED_DATA_CSV}...")
try:
    df_final = pd.read_csv(GENERATED_DATA_CSV)
    df_wm = df_final[df_final['type'] == 'watermarked'].head(NUM_SAMPLES)
    df_nwm = df_final[df_final['type'] == 'non_watermarked'].head(NUM_SAMPLES)

    if len(df_wm) < NUM_SAMPLES or len(df_nwm) < NUM_SAMPLES:
        print(f"Error: Not enough samples in final dataset ({len(df_wm)} WM, {len(df_nwm)} non-WM). Required {NUM_SAMPLES}. Exiting.")
        exit()

    print(f"Converting {len(df_wm)} WM and {len(df_nwm)} non-WM samples to tensors...")
    # Create lists of 1D tensors directly from the CSV strings
    tokenized_wm_outputs = [torch.tensor(list(map(int, s.split(','))), dtype=torch.long) for s in df_wm['ids_str'] if s]
    tokenized_uwm_outputs = [torch.tensor(list(map(int, s.split(','))), dtype=torch.long) for s in df_nwm['ids_str'] if s]

    # Check again after conversion
    if len(tokenized_wm_outputs) < NUM_SAMPLES or len(tokenized_uwm_outputs) < NUM_SAMPLES:
         print(f"Error: Not enough valid samples after converting IDs. Exiting.")
         exit()

    print("Data loaded and converted.")

except FileNotFoundError:
     print(f"Error: Cache file {GENERATED_DATA_CSV} not found. Exiting.")
     exit()
except Exception as e:
     print(f"Error loading or processing data from CSV for training: {e}")
     exit()

# --- Train the Bayesian Detector ---
print("\nStarting Bayesian Detector training...")
training_start_time = time.time()

try:
    # --- Force JAX platform attempt ---
    print("Attempting to force JAX platform to GPU...")
    try:
        jax.config.update('jax_platform_name', 'gpu')
        print(f"JAX default backend explicitly set to: {jax.default_backend()}")
    except Exception as e:
        print(f"Could not force JAX platform: {e}")

    # --- DEBUG block ---
    device_obj_to_pass = torch.device(device)
    print(f"\nDEBUG: About to call train_best_detector with:")
    print(f"  torch_device = {device_obj_to_pass}")
    print(f"  len(tokenized_wm_outputs) = {len(tokenized_wm_outputs)}") # Should be > 1 now
    print(f"  len(tokenized_uwm_outputs) = {len(tokenized_uwm_outputs)}") # Should be > 1 now
    if tokenized_wm_outputs: print(f"  Shape of first WM tensor = {tokenized_wm_outputs[0].shape}") # Should be 1D
    if tokenized_uwm_outputs: print(f"  Shape of first Non-WM tensor = {tokenized_uwm_outputs[0].shape}\n") # Should be 1D
    # --- End DEBUG block ---

    l2_weights=np.zeros((1,))
    bayesian_detector, test_loss = (
        detector_bayesian.BayesianDetector.train_best_detector(
            # --- Pass the LISTS of 1D tensors ---
            tokenized_wm_outputs=tokenized_wm_outputs,
            tokenized_uwm_outputs=tokenized_uwm_outputs,
            # --- End change ---
            logits_processor=detection_processor,
            tokenizer=tokenizer,
            torch_device=torch.device(device), # Corrected device object
            max_padded_length=MAX_PADDED_LENGTH,
            pos_truncation_length=POS_TRUNCATION_LENGTH,
            neg_truncation_length=NEG_TRUNCATION_LENGTH,
            verbose=True,
            learning_rate=LEARNING_RATE,
            n_epochs=N_EPOCHS,
            l2_weights=l2_weights,
        )
    )
    # ... (rest of the script remains the same) ...
    training_end_time = time.time()
    print(f"\nTraining finished. Test Loss: {test_loss:.4f}")
    print(f"Training time: {training_end_time - training_start_time:.2f} seconds.")

    # --- Save the Trained Detector ---
    print(f"\nSaving trained detector to {OUTPUT_DETECTOR_FILE}...")
    with open(OUTPUT_DETECTOR_FILE, 'wb') as f:
        pickle.dump(bayesian_detector, f)
    print("Detector saved successfully.")

except ImportError:
    print("\nError: Could not import 'detector_bayesian'. Is synthid-text library installed correctly?")
except Exception as e:
    print(f"\nAn error occurred during training or saving: {e}")
    import traceback
    traceback.print_exc()
# --- Final Cleanup ---
print("\nCleaning up...")
try:
    # Don't delete model/tokenizer if they are meant to be persistent in inference.py
    del detection_processor, tokenized_wm_outputs, tokenized_uwm_outputs, bayesian_detector
except NameError: pass
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("Training script finished.")