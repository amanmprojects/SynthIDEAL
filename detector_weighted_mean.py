# detector_weighted_mean.py

# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Provides a function to calculate the weighted mean score for SynthID text detection.

This script is based on the usage shown in the SynthID Text notebook example.
It utilizes the detector_mean module from the synthid-text library.
"""

import numpy as np
from synthid_text import detector_mean
from jaxtyping import Float, Bool, Array

def calculate_weighted_mean_score(
    g_values: Float[Array, "batch seq_len depth"],
    mask: Bool[Array, "batch seq_len"]
) -> Float[Array, "batch"]:
    """Calculates the weighted mean score for watermark detection.

    This function wraps the `weighted_mean_score` function from the
    `synthid_text.detector_mean` module. It takes the computed g-values
    and a validity mask as input.

    Args:
        g_values: A NumPy array containing the g-values computed during text
          generation/analysis. Expected shape [batch_size, sequence_length, depth].
          Note: For mean/weighted_mean, the effective depth used seems to be 1.
        mask: A boolean NumPy array indicating which tokens are valid for scoring.
          Invalid tokens (e.g., padding, EOS, context repetitions) should be
          marked False. Expected shape [batch_size, sequence_length].

    Returns:
        A NumPy array containing the weighted mean score for each item in the batch.
        Shape [batch_size]. Higher scores suggest a higher likelihood of the
        text being watermarked with the corresponding configuration.
    """
    if not isinstance(g_values, np.ndarray):
        raise TypeError(f"Expected g_values to be a numpy array, but got {type(g_values)}")
    if not isinstance(mask, np.ndarray):
         raise TypeError(f"Expected mask to be a numpy array, but got {type(mask)}")
    if g_values.ndim != 3:
         raise ValueError(f"Expected g_values to have 3 dimensions [batch, seq, depth], but got {g_values.ndim}")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask to have 2 dimensions [batch, seq], but got {mask.ndim}")
    if g_values.shape[0] != mask.shape[0] or g_values.shape[1] != mask.shape[1]:
         raise ValueError(
             "Batch size and sequence length must match between g_values "
            f"({g_values.shape}) and mask ({mask.shape})"
        )
    if mask.dtype != np.bool_:
        # Convert to boolean if not already, as required by the logic
        mask = mask.astype(np.bool_)


    weighted_mean_scores = detector_mean.weighted_mean_score(g_values, mask)
    return weighted_mean_scores

if __name__ == "__main__":
    # Example Usage based on the notebook's context:
    # Assume we have obtained g_values and mask after generating text
    # (e.g., using logits_processor.compute_g_values and masking logic)

    # Example Placeholder Data (replace with actual data)
    batch_size = 4
    sequence_length = 50  # Example sequence length after removing prompt etc.
    depth = 1 # Depth for g_values (seems to be implicitly 1 for mean funcs)

    # Dummy g_values (replace with actual computed g-values)
    # Watermarked text tends to have higher positive g-values
    dummy_g_values_wm = np.random.rand(batch_size, sequence_length, depth) * 2 - 0.5 # Mostly positive
    dummy_g_values_uwm = np.random.rand(batch_size, sequence_length, depth) * 2 - 1.0 # Centered around 0

    # Dummy mask (replace with actual mask)
    # True for valid tokens, False for padding/EOS/context repetitions etc.
    dummy_mask = np.random.rand(batch_size, sequence_length) > 0.1 # Most tokens are valid
    dummy_mask = dummy_mask.astype(np.bool_)

    print("Calculating weighted mean scores for dummy 'watermarked' data...")
    wm_scores = calculate_weighted_mean_score(dummy_g_values_wm, dummy_mask)
    print("Weighted Mean Scores (Watermarked):", wm_scores)
    print("-" * 20)

    print("Calculating weighted mean scores for dummy 'unwatermarked' data...")
    uwm_scores = calculate_weighted_mean_score(dummy_g_values_uwm, dummy_mask)
    print("Weighted Mean Scores (Unwatermarked):", uwm_scores)
    print("-" * 20)

    # Expected output: wm_scores should generally be higher than uwm_scores