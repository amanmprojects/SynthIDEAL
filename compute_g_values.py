from transformers import SynthIDTextWatermarkingConfig
from inference import compute_generated_ids
from synthid_text import logits_processing
import transformers # Needed for tokenizer later

# SynthID Text configuration
CONFIG = {
    "keys":[654, 400, 836, 123, 340, 443, 597, 160, 57, 589, 796, 896, 521, 145, 654, 400, 836, 123, 36, 443, 597, 13, 325, 971, 734, 886, 361, 145],
    "ngram_len":5,
}






# Initialize the processor
# Note: The notebook also passed top_k and temperature, but they are primarily
# used during generation, not essential for compute_g_values itself.
logits_processor = logits_processing.SynthIDLogitsProcessor(**CONFIG)


generated_ids = compute_generated_ids(prompt="Hey, where is India?")


# Assuming 'generated_ids' is a PyTorch tensor of shape [batch_size, sequence_length]
# containing the token IDs of the text to analyze.
g_values = logits_processor.compute_g_values(input_ids=generated_ids)

# g_values will be a PyTorch tensor
# Shape: [batch_size, sequence_length - (ngram_len - 1), depth]
# Depth is usually 1 for the mean/weighted mean detectors.
print("Shape of g_values:", g_values.shape)




