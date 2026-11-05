import os
import sys
from datetime import datetime

# =====================================================================
# 1. PATH & DIRECTORY CONFIGURATION
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATASET_PATH = os.path.join(BASE_DIR, "dataset_test.jsonl")
BASE_BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmarks")

# Create main benchmarks directory if it doesn't exist
os.makedirs(BASE_BENCHMARK_DIR, exist_ok=True)

# =====================================================================
# 2. MODEL REGISTRY CONFIGURATION
# Adjust current model here before running an evaluation pass
# =====================================================================
AVAILABLE_MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}

# Set the active model key for the run
ACTIVE_MODEL_KEY = "qwen2.5-7b"  # Change to: "qwen3.5-9b", "llama3.1-8b", or "mistral-7b"
MODEL_NAME = AVAILABLE_MODELS.get(ACTIVE_MODEL_KEY, ACTIVE_MODEL_KEY)

# =====================================================================
# 3. vLLM INFERENCE HYPERPARAMETERS
# =====================================================================
TEMPERATURE = 0.0          # Deterministic sampling for benchmark reproducibility
TOP_P = 1.0
MAX_TOKENS = 256            # Sufficient budget for classification & reasoning snippet
GPU_MEMORY_UTILIZATION = 0.85 # Fits within 16GB VRAM on RTX 5070 Ti
MAX_MODEL_LEN = 4096       # Context length limit

# =====================================================================
# 4. PARSER REGEX PATTERNS
# =====================================================================
# Negative indicators checked FIRST to prevent false "flaky" matches on "non-flaky"
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
# Prevents overwriting benchmarks by appending model key and timestamp
# =====================================================================
def get_run_output_dir(run_tag: str = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{run_tag}" if run_tag else ""
    folder_name = f"run_{ACTIVE_MODEL_KEY}{tag}_{timestamp}"
    run_dir = os.path.join(BASE_BENCHMARK_DIR, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

if __name__ == "__main__":
    print(f"[*] Configuration Loaded Successfully.")
    print(f"[*] Active Model: {MODEL_NAME}")
    print(f"[*] Test Dataset Path: {TEST_DATASET_PATH}")
    print(f"[*] Benchmark Run Folder Example: {get_run_output_dir('test')}")