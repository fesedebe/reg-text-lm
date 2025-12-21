#serve.py
#Serves merged model via vLLM with OpenAI-compatible API.

import os
import subprocess
import sys
from pathlib import Path

from config import (
    model_config,
    inference_config,
    data_config,
    MERGED_MODEL_DIR,
)

# local config
API_KEY = os.environ.get("VLLM_API_KEY", "")
if not API_KEY:
    raise ValueError("Set VLLM_API_KEY environment variable")

def serve():
    model_path = MERGED_MODEL_DIR
    
    print(f"Model: {model_config.model_name} ({model_config.model_key})")
    print(f"Path: {model_path}")
    print(f"Host: {inference_config.host}")
    print(f"Port: {inference_config.port}")
    print(f"API Key: {API_KEY[:8]}...")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Merged model not found: {model_path}."
        )
    
    # vLLM server command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", "legal-simplifier",
        "--host", inference_config.host,
        "--port", str(inference_config.port),
        "--dtype", "bfloat16",
        "--max-model-len", str(inference_config.max_new_tokens + 1024),
        "--gpu-memory-utilization", "0.90",
        "--trust-remote-code",
        "--api-key", API_KEY,
    ]
    
    print(f"\nStarting vLLM server...")
    print(f"API will be available at: http://{inference_config.host}:{inference_config.port}/v1")
    print(f"\nSystem prompt for reference:\n{data_config.system_prompt}\n")
    
    # run server
    subprocess.run(cmd)

if __name__ == "__main__":
    serve()