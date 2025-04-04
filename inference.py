import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, SynthIDTextWatermarkingConfig
import random


from dotenv import load_dotenv
from huggingface_hub import login
import os

# Load environment variables and login to Huggingface
load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# 1. Specify Model ID and Device
model_id = "meta-llama/Llama-3.2-1B-Instruct" # Using Llama 3.1 1B Instruct
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# SynthID Text configuration
watermarking_config = SynthIDTextWatermarkingConfig(
    keys=[654, 400, 836, 123, 340, 443, 597, 160, 57, 589, 796, 896, 521, 145, 654, 400, 836, 123, 36, 443, 597, 13, 325, 971, 734, 886, 361, 145],
    ngram_len=20,
)

# Determine torch dtype based on device and capabilities
# Use bfloat16 for Ampere GPUs or newer, float16 otherwise for GPU, or default for CPU
torch_dtype = torch.float32 # Default for CPU
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        print("Using bfloat16")
    else:
        torch_dtype = torch.float16
        print("Using float16")

# 2. Load Tokenizer
# Note: trust_remote_code=True might be needed for some models/tokenizers
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Trying with trust_remote_code=True")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("Tokenizer loaded successfully with trust_remote_code=True.")


# Set padding token if not already set (common for Llama models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

# 3. Load Model
print(f"Loading model: {model_id} with dtype: {torch_dtype}")
# Add device_map='auto' for better distribution if using multiple GPUs or offloading
# Add load_in_8bit=True or load_in_4bit=True (requires bitsandbytes) for quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    device_map=device # Use 'auto' if you want Transformers to handle distribution/offload
    # load_in_8bit=True, # Uncomment for 8-bit quantization
    # load_in_4bit=True, # Uncomment for 4-bit quantization
    # trust_remote_code=True # May be required for some model architectures
)
print("Model loaded successfully.")

# 4. Prepare Input using the Chat Template
# **Crucial for Instruct models**: Use the specific chat template!
def generate_responses(user_prompt: str):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Always try to response in a useful, but brief way"},
        {"role": "user", "content": user_prompt}
    ]

    # Apply the chat template. add_generation_prompt=True adds the prompt structure
    # for the model to know it should generate the next turn (assistant's response).
    inputs_templated = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt" # Return PyTorch tensors
    ).to(device) # Move tensors to the correct device

    print(f"\nInput shape: {inputs_templated.shape}")
    # Optional: Decode the templated input to see how it looks
    # print("--- Templated Input ---")
    # print(tokenizer.decode(inputs_templated[0]))
    # print("-----------------------\n")

    # 5. Generate Text
    print("Generating response...")
    # Set generation parameters
    generation_config = {
        "max_new_tokens": 1000,  # Limit the length of the response
        "do_sample": True,      # Use sampling for more creative/varied output
        "temperature": 0.6,     # Controls randomness (lower = more deterministic)
        "top_p": 0.9,           # Nucleus sampling: considers tokens cumulative probability > top_p
        "eos_token_id": tokenizer.eos_token_id, # Stop generation at EOS token
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, # Prevent warnings
        "watermarking_config":watermarking_config
    }


    selection_for_watermark = random.choice([0, 1, 2, 3])
    result={
        "choices": [],
        "watermarked_choice": selection_for_watermark
    }
    print(f"--------------------------\nCorrect choice: {selection_for_watermark}\n---------------------------")
    for i in range(4):
        if i == selection_for_watermark:
            generation_config["watermarking_config"] = watermarking_config
        else:
            generation_config.pop("watermarking_config", False)
        # Generate output token IDs
        # Use torch.no_grad() to save memory during inference
        with torch.no_grad():
            outputs = model.generate(
                inputs_templated,
                **generation_config
            )

        print("Generation complete.")

        # 6. Decode the Output
        # The output includes the input tokens, so slice it to get only the generated part
        generated_ids = outputs[0, inputs_templated.shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print("\n--- Model Response ---")
        print(response)
        print("--------------------")
        result["choices"].append(response)
    
    return result




# generate_responses("Where is India?")