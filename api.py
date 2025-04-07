# api.py (Modified for Bayesian Detector)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import transformers
import gc
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from huggingface_hub import login
import pickle # Needed to load the detector
import jax.numpy as jnp

# --- Import necessary components ---
try:
    from inference import tokenizer, device, watermarking_config as inference_watermarking_config
    # Keep generate_responses if the /ask endpoint is still needed
    from inference import generate_responses

    # Import the LogitsProcessor (still needed for g-values/mask)
    try:
        from transformers.generation import SynthIDLogitsProcessor
        print("Using SynthIDLogitsProcessor from transformers.")
    except ImportError:
        from synthid_text import logits_processing
        SynthIDLogitsProcessor = logits_processing.SynthIDLogitsProcessor
        print("Warning: Using SynthIDLogitsProcessor from synthid_text library.")

    # Import Bayesian Detector class (needed for loading)
    from synthid_text import detector_bayesian # Import the class itself

except ImportError as e:
    print(f"Fatal Error during imports: {e}")
    print("Ensure 'inference.py', 'detector_weighted_mean.py' exist and define required components.")
    print("Also ensure 'transformers' and 'synthid-text' libraries are installed.")
    exit()
except Exception as e:
    print(f"An unexpected fatal error occurred during imports: {e}")
    exit()

# --- Global Variables ---
logits_processor = None # Still needed for g-value/mask calculation
loaded_bayesian_detector = None # To store the loaded trained detector
detection_config = None
TRAINED_DETECTOR_FILE = "bayesian_detector.pkl"
# Threshold for Bayesian detector (log-odds). 0.0 is neutral. Needs calibration!
BAYESIAN_THRESHOLD = 0.5

# --- Helper Function to Prepare Config (Unchanged) ---
def prepare_detection_config():
    global detection_config
    imported_config_obj = inference_watermarking_config
    print("Preparing detection config...")
    if hasattr(imported_config_obj, 'to_dict'):
        temp_config = imported_config_obj.to_dict()
    elif isinstance(imported_config_obj, dict):
        temp_config = imported_config_obj.copy()
    else:
        print("Warning: Imported 'watermarking_config' type unrecognized. Attempting attribute access.")
        try:
            temp_config = {
                'keys': imported_config_obj.keys, 'ngram_len': imported_config_obj.ngram_len,
                'sampling_table_size': getattr(imported_config_obj, 'sampling_table_size', 65536),
                'sampling_table_seed': getattr(imported_config_obj, 'sampling_table_seed', 0),
                'context_history_size': getattr(imported_config_obj, 'context_history_size', 1024),
                'skip_first_ngram_calls': getattr(imported_config_obj, 'skip_first_ngram_calls', False),
             }
        except AttributeError as e: raise RuntimeError(...) from e

    detection_config = {
        'keys': temp_config['keys'], 'ngram_len': temp_config['ngram_len'],
        'sampling_table_size': temp_config.get('sampling_table_size', 65536),
        'sampling_table_seed': temp_config.get('sampling_table_seed', 0),
        'context_history_size': temp_config.get('context_history_size', 1024),
        'skip_first_ngram_calls': temp_config.get('skip_first_ngram_calls', False),
        'temperature': 1.0, 'top_k': 40, 'device': str(device)
    }
    print(f"Detection CONFIG prepared: {detection_config.keys()}")


# --- FastAPI Lifespan Event ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("FastAPI starting up...")
    global logits_processor, loaded_bayesian_detector
    prepare_detection_config()

    # 1. Initialize the Logits Processor (still needed)
    try:
        print("Initializing SynthIDLogitsProcessor...")
        temp_init_config = detection_config.copy()
        # Handle potential device argument issue
        try:
            logits_processor = SynthIDLogitsProcessor(**temp_init_config)
        except TypeError as e:
            if 'device' in str(e) and 'device' in temp_init_config:
                 del temp_init_config['device']
                 logits_processor = SynthIDLogitsProcessor(**temp_init_config)
            else: raise
        print("SynthIDLogitsProcessor initialized successfully.")
    except Exception as e:
        print(f"Fatal Error initializing SynthIDLogitsProcessor: {e}")
        raise RuntimeError("Failed to initialize logits processor.") from e

    # 2. Load the Trained Bayesian Detector
    try:
        print(f"Loading trained Bayesian detector from {TRAINED_DETECTOR_FILE}...")
        with open(TRAINED_DETECTOR_FILE, 'rb') as f:
            # Make sure the detector_bayesian module is available here
            loaded_bayesian_detector = pickle.load(f)
        if not isinstance(loaded_bayesian_detector, detector_bayesian.BayesianDetector):
             print("Error: Loaded object is not a BayesianDetector instance!")
             raise RuntimeError("Loaded invalid detector file.")
        print("Trained Bayesian detector loaded successfully.")
    except FileNotFoundError:
        print(f"Fatal Error: Trained detector file not found: {TRAINED_DETECTOR_FILE}")
        print("Please run train_bayesian_detector.py first.")
        # Decide behavior: raise error or run without detection? Raising is safer.
        raise RuntimeError(f"Missing detector file: {TRAINED_DETECTOR_FILE}")
    except Exception as e:
        print(f"Fatal Error loading trained Bayesian detector: {e}")
        raise RuntimeError("Failed to load trained detector.") from e

    yield # API runs here

    # Code to run on shutdown
    print("FastAPI shutting down...")
    del logits_processor, loaded_bayesian_detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleanup finished.")


# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Pydantic Models (Unchanged) ---
class AskRequest(BaseModel): text: str
class DetectRequest(BaseModel): text: str
class DetectResponse(BaseModel):
    watermark_detected: bool
    score: float
    threshold: float

# --- API Endpoints ---

@app.post("/ask")
async def get_llm_response(request: AskRequest):
    # (Endpoint unchanged, assuming it's still needed)
    try:
        result = generate_responses(request.text)
        return result
    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error generating response.")

@app.post("/detect_watermark", response_model=DetectResponse)
async def detect_watermark(request: DetectRequest):
    """
    Detects if the provided text contains the configured watermark
    using the loaded Bayesian detector.
    """
    global loaded_bayesian_detector

    if loaded_bayesian_detector is None:
        raise HTTPException(status_code=503, detail="Detection service not initialized.")

    text_to_check = request.text
    if not text_to_check:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    print(f"\nReceived request to detect watermark (Bayesian): '{text_to_check[:100]}...'")

    try:
        # 1. Tokenize the input text
        inputs = tokenizer(text_to_check, return_tensors="pt", truncation=True, max_length=1024)
        generated_ids = inputs['input_ids'].to(device) # PyTorch tensor on device

        # Check length
        ngram_len = loaded_bayesian_detector.logits_processor.ngram_len
        if generated_ids.shape[1] < ngram_len:
             print(f"Warning: Text too short ({generated_ids.shape[1]} tokens) for ngram_len={ngram_len}.")
             return DetectResponse(watermark_detected=False, score=-999.0, threshold=BAYESIAN_THRESHOLD)

        # --- CHANGE HERE: Pass PyTorch tensor directly ---
        # Remove the conversion to JAX array:
        # generated_ids_jax = jnp.array(generated_ids.cpu().numpy()) # REMOVE THIS LINE

        # 3. Calculate Score using BAYESIAN detector's score method
        # Pass the PyTorch tensor directly. The score method internally computes g_values/mask.
        bayesian_scores = loaded_bayesian_detector.score(generated_ids) # Pass PyTorch tensor
        # --- End Change ---

        final_score = float(bayesian_scores[0])

        # 4. Apply Threshold
        is_detected = final_score > BAYESIAN_THRESHOLD

        print(f"Bayesian detection complete. Score: {final_score:.4f}, Detected: {is_detected}")

        return DetectResponse(
            watermark_detected=is_detected,
            score=final_score,
            threshold=BAYESIAN_THRESHOLD
        )

    except Exception as e:
        print(f"Error during Bayesian watermark detection: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during detection process: {e}")


if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)