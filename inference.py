# inference.py (Updated for INT4 Quantization)

import torch
# Import BitsAndBytesConfig for fine-tuning 4-bit quantization
from transformers import AutoTokenizer, AutoModelForCausalLM, SynthIDTextWatermarkingConfig, BitsAndBytesConfig
import random


from dotenv import load_dotenv
from huggingface_hub import login
import os

# Load environment variables and login to Huggingface
load_dotenv()
# Optional: wrap login in try-except if token might be missing
try:
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
except Exception as e:
    print(f"Hugging Face login failed (is HUGGINGFACE_TOKEN set?): {e}")
    # Decide if you want to proceed without login, depending on model access rights
    # exit() # Or just print warning and continue

# 1. Specify Model ID and Device
model_id = "meta-llama/Llama-3.2-1B-Instruct" # Using Llama 3.1 1B Instruct
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# SynthID Text configuration (Unchanged)
watermarking_config = SynthIDTextWatermarkingConfig(
    keys=[654, 400, 836, 123, 340, 443, 597, 160, 57, 589, 796, 896, 521, 145, 654, 400, 836, 123, 36, 443, 597, 13, 325, 971, 734, 886, 361, 145],
    ngram_len=5,
    sampling_table_size=65563, # Size of the sampling table for the watermarking process
    sampling_table_seed=0,
    context_history_size=1024,
    skip_first_ngram_calls=False,
    debug_mode=False,
)

# Determine compute dtype (used during inference despite 4-bit weights)
# We still want computations done in a higher precision format for accuracy
compute_dtype = torch.float32 # Default for CPU
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        print("Using bfloat16 for computation.")
    else:
        compute_dtype = torch.float16
        print("Using float16 for computation.")

# 2. Load Tokenizer (Unchanged)
print(f"Loading tokenizer: {model_id}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Trying with trust_remote_code=True")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("Tokenizer loaded successfully with trust_remote_code=True.")
    except Exception as e_inner:
        print(f"Fatal error loading tokenizer even with trust_remote_code=True: {e_inner}")
        exit()


# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

# 3. Load Model with INT4 Quantization
print(f"Loading model: {model_id} with INT4 quantization...")

# Configure quantization parameters using BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",             # NF4 quantization type recommended
    bnb_4bit_compute_dtype=compute_dtype,  # Use determined compute dtype
    bnb_4bit_use_double_quant=True,        # Optional: Nested quantization
)

# Load the quantized model
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config, # Pass the config object
        device_map=device, # Keep user's setting, 'auto' also works well
        # torch_dtype argument is generally not used directly with quantization_config
        # trust_remote_code=True # Uncomment if model architecture requires it
    )
    print("INT4 Quantized Model loaded successfully.")
except ImportError:
    print("Error: bitsandbytes library not found or import failed.")
    print("Please ensure it's installed correctly (`pip install bitsandbytes`)")
    exit()
except Exception as e:
     print(f"Fatal error loading quantized model: {e}")
     print("Check CUDA setup, bitsandbytes installation, and model access rights.")
     exit()

# Model is now loaded in 4-bit, but computations will use compute_dtype

# 4. Prepare Input using the Chat Template (Unchanged)
def generate_responses(user_prompt: str):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Respond to the user in 5-10 lines."},
        {"role": "user", "content": user_prompt}
    ]
    inputs_templated = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    print(f"\nInput shape: {inputs_templated.shape}")

    # 5. Generate Text (Unchanged Logic - model handles quantized inference)
    print("Generating responses (using INT4 model)...")

    selection_for_watermark = random.choice([0, 1, 2, 3])
    result={
        "choices": [],
        "watermarked_choice": selection_for_watermark
    }
    print(f"--------------------------\nCorrect choice (watermarked): {selection_for_watermark}\n---------------------------")
    for i in range(4):
        if i == selection_for_watermark:
            generation_config = {
                "max_new_tokens": 1000, "do_sample": True, "temperature": 0.6,
                "top_p": 0.9, "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "watermarking_config": watermarking_config # Apply watermark
            }
            print(f"Generating choice {i} (Watermarked)")
        else:
            generation_config = {
                "max_new_tokens": 1000, "do_sample": True, "temperature": 0.6,
                "top_p": 0.9, "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                # NO watermarking_config here
            }
            print(f"Generating choice {i} (Not Watermarked)")

        with torch.no_grad():
            outputs = model.generate(inputs_templated, **generation_config)

        # print("Generation complete for choice {i}.") # Less verbose

        # 6. Decode the Output (Unchanged)
        generated_ids = outputs[0, inputs_templated.shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # print(f"\n--- Model Response {i} ---") # Less verbose
        # print(response)
        # print("--------------------")
        result["choices"].append(response)

    return result


def compute_generated_ids(prompt: str):
    """Generates text WITH watermark using the INT4 model."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Respond to the user in 5-10 lines."},
        {"role": "user", "content": prompt}
    ]
    inputs_templated = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    generation_config = {
        "max_new_tokens": 1000, "do_sample": True, "temperature": 0.6,
        "top_p": 0.9, "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "watermarking_config": watermarking_config # Apply watermark
    }
    print("Generating single watermarked response (using INT4 model)...")
    with torch.no_grad():
        outputs = model.generate(inputs_templated, **generation_config)

    print("Generation complete.")
    generated_ids = outputs[0, inputs_templated.shape[-1]:]
    return generated_ids


# Example of direct execution (optional)
if __name__ == "__main__":
    print("\n--- Running direct execution example ---")
    test_prompt = "What is the capital of France?"
    # Example using generate_responses
    # responses_data = generate_responses(test_prompt)
    # print("\n--- generate_responses output ---")
    # print(responses_data)

    # Example using compute_generated_ids
    watermarked_ids = compute_generated_ids(test_prompt)
    watermarked_text = tokenizer.decode(watermarked_ids, skip_special_tokens=True)
    print("\n--- compute_generated_ids output ---")
    print(f"Generated {len(watermarked_ids)} tokens.")
    print("Decoded Text:")
    print(watermarked_text)
    print("------------------------------------")