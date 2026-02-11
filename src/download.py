"""
Model Downloader for Mistral NPU Chat
Downloads OpenVINO-optimized models from HuggingFace
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# Load environment variables
load_dotenv()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

# Available models (NPU-optimized)
MODELS = {
    "mistral-7b": "OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov",
    "deepseek-1.5b": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov",
    "deepseek-7b": "OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-cw-ov",
    "qwen3-8b": "OpenVINO/Qwen3-8B-int4-cw-ov",
    "phi3-mini": "OpenVINO/Phi-3-mini-4k-instruct-int4-cw-ov",
}


def download_model(model_key: str, local_name: str = None):
    """Download a model from HuggingFace"""

    if model_key not in MODELS:
        print(f"[!] Unknown model: {model_key}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False

    model_id = MODELS[model_key]
    local_name = local_name or model_key.replace("-", "_") + "_npu_cw"
    local_dir = MODEL_DIR / local_name

    # Get HuggingFace token from environment (optional for public models)
    hf_token = os.getenv("HF_TOKEN")

    print(f"Downloading {model_id}...")
    print(f"Destination: {local_dir}")

    if hf_token:
        print("Using HuggingFace token from .env")
    else:
        print("No HF_TOKEN set (not required for public models)")

    print()

    try:
        # Create models directory if not exists
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            token=hf_token,
            local_dir_use_symlinks=False
        )
        print(f"\n[OK] Download complete: {local_dir}")
        return True
    except Exception as e:
        print(f"\n[!] Download failed: {e}")
        return False


def list_models():
    """List available models"""
    print("\nAvailable models for download:")
    print("-" * 50)
    for key, repo in MODELS.items():
        print(f"  {key:15} -> {repo}")
    print("-" * 50)
    print("\nUsage: python src/download.py <model-name>")
    print("Example: python src/download.py mistral-7b")


def main():
    if len(sys.argv) < 2:
        list_models()
        return

    model_key = sys.argv[1].lower()

    if model_key in ['--list', '-l', 'list']:
        list_models()
        return

    local_name = sys.argv[2] if len(sys.argv) > 2 else None
    download_model(model_key, local_name)


if __name__ == "__main__":
    main()
