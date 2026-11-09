import os
import sys
from datetime import datetime

# =====================================================================
# 1. PATH & DIRECTORY CONFIGURATION
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATASET_PATH = os.path.join(BASE_DIR, "dataset_test.jsonl")
BASE_BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmarks_llamacpp")

# Create main benchmarks directory if it doesn't exist
os.makedirs(BASE_BENCHMARK_DIR, exist_ok=True)

# =====================================================================
# 2. MODEL REGISTRY CONFIGURATION (GGUF Format for llama.cpp)
# Adjust current model here before running an evaluation pass
# =====================================================================

# AVAILABLE_MODELS = {
#     "qwen2.5-7b": {
#         "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
#         "filename": "qwen2.5-7b-instruct-q4_k_m.gguf"
#     },
#     "qwen3.5-9b": {
#         "repo_id": "bartowski/Qwen_Qwen3.5-9B-GGUF",
#         "filename": "Qwen3.5-9B-Q4_K_M.gguf"
#     },
#     "llama3.1-8b": {
#         "repo_id": "bartowski/Llama-3.1-8B-Instruct-GGUF",
#         "filename": "Llama-3.1-8B-Instruct-Q4_K_M.gguf"
#     },
#     "mistral-7b": {
#         "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
#         "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
#     },
# }

# =====================================================================
# 2. MODEL REGISTRY CONFIGURATION (GGUF Format for llama.cpp)
# =====================================================================

AVAILABLE_MODELS = {
    "qwen2.5-7b": {
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    },
    "qwen3.5-9b": {
        "repo_id": "bartowski/Qwen_Qwen3.5-9B-GGUF",
        "filename": "Qwen_Qwen3.5-9B-Q4_K_M.gguf"
    },
    "llama3.1-8b": {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    },
    "mistral-7b": {
        "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    },
}

# Set the active model key for the run
ACTIVE_MODEL_KEY = "mistral-7b"  # Change to: "qwen2.5-7b", "qwen3.5-9b", "llama3.1-8b", or "mistral-7b"
ACTIVE_MODEL_CONFIG = AVAILABLE_MODELS.get(ACTIVE_MODEL_KEY, AVAILABLE_MODELS["qwen3.5-9b"])

# =====================================================================
# 3. llama.cpp INFERENCE HYPERPARAMETERS
# =====================================================================
TEMPERATURE = 0.0          # Deterministic sampling for benchmark reproducibility, 0.0
TOP_P = 0.25
MAX_TOKENS = 16            # Minimal token budget since grammar strictly restricts outputs

N_GPU_LAYERS = -1          # Offload all layers completely to your RTX 5070 Ti VRAM
N_CTX = 16384              # Context length window matching your requirements

# =====================================================================
# 4. PARSER REGEX PATTERNS & NATIVE GBNF GRAMMAR
# =====================================================================
# GBNF Grammar rule physically constraining generation to binary choices
CLASSIFICATION_GRAMMAR = r'''
root ::= "flaky" | "non_flaky"
'''

NEGATIVE_PATTERNS = [
    r"\bnon-flaky\b",
    r"\bnon_flaky\b",
    r"\bnot flaky\b",
    r"\bdeterministic\b",
    r"\bclassification:\s*non-flaky\b",
]

POSITIVE_PATTERNS = [
    r"\bflaky\b",
    r"\border-dependent\b",
    r"\binfrastructure\b",
    r"\bclassification:\s*flaky\b",
]

# =====================================================================
# 5. UNIQUE RUN DIRECTORY GENERATOR
# =====================================================================
def get_run_output_dir(run_tag: str = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{run_tag}" if run_tag else ""
    folder_name = f"run_{ACTIVE_MODEL_KEY}{tag}_{timestamp}"
    run_dir = os.path.join(BASE_BENCHMARK_DIR, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

if __name__ == "__main__":
    print(f"[*] Configuration (llama.cpp) Loaded Successfully.")
    print(f"[*] Active Model Key: {ACTIVE_MODEL_KEY}")
    print(f"[*] Repo ID: {ACTIVE_MODEL_CONFIG['repo_id']}")
    print(f"[*] Filename: {ACTIVE_MODEL_CONFIG['filename']}")
    print(f"[*] Test Dataset Path: {TEST_DATASET_PATH}")
    print(f"[*] Benchmark Run Folder Example: {get_run_output_dir('test')}")