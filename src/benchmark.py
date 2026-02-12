"""
Benchmark Script for Mistral-for-NPU
Measures tokens/s, first token latency, and memory usage across models and devices.
Results can be printed as Markdown tables matching the README format.
"""

import openvino_genai as ov_genai
import openvino as ov
import time
import os
import sys
import json
import platform
import argparse
from pathlib import Path
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "benchmarks"

# Standard benchmark prompts — short, medium, long to test different scenarios
PROMPTS = {
    "short": "What is the capital of France?",
    "medium": "Explain quantum computing in simple terms that a high school student could understand.",
    "long": (
        "Write a detailed comparison of Python, JavaScript, and Rust programming languages. "
        "Cover their type systems, performance characteristics, memory management approaches, "
        "ecosystem and package management, and ideal use cases for each language."
    ),
}

DEFAULT_PROMPT_KEY = "medium"

# Models and their expected local folder names (matching download.py conventions)
MODELS = {
    "mistral-7b":    {"folder": "mistral_7b_npu_cw",    "params": "7B"},
    "deepseek-1.5b": {"folder": "deepseek_1.5b_npu_cw", "params": "1.5B"},
    "deepseek-7b":   {"folder": "deepseek_7b_npu_cw",   "params": "7B"},
    "qwen3-8b":      {"folder": "qwen3_8b_npu_cw",      "params": "8B"},
    "phi3-mini":     {"folder": "phi3_mini_npu_cw",      "params": "3.8B"},
}

# Also check the old naming convention from the .env default
FOLDER_ALIASES = {
    "mistral_npu_cw": "mistral-7b",
}

DEVICES = ["NPU", "CPU", "GPU"]

MAX_PROMPT_LEN = 2048
MAX_NEW_TOKENS = 128  # Fixed output length for fair comparison
WARMUP_ROUNDS = 1
BENCH_ROUNDS = 3


class TokenCounter:
    """Streaming callback that counts tokens and captures first-token latency."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.token_count = 0
        self.first_token_time = None
        self.start_time = None

    def start(self):
        self.reset()
        self.start_time = time.perf_counter()

    def __call__(self, token: str) -> bool:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.token_count += 1
        return False  # don't stop generation

    @property
    def first_token_latency(self):
        if self.first_token_time and self.start_time:
            return self.first_token_time - self.start_time
        return None

    @property
    def elapsed(self):
        if self.start_time:
            return time.perf_counter() - self.start_time
        return None


def get_memory_mb():
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return None


def find_model_path(model_key):
    """Find model folder on disk. Tries known folder names."""
    info = MODELS.get(model_key)
    if info:
        path = MODEL_DIR / info["folder"]
        if path.exists():
            return path

    # Try aliases and common naming patterns
    for folder_name in [
        model_key.replace("-", "_") + "_npu_cw",
        model_key.replace("-", "_"),
        model_key,
    ]:
        path = MODEL_DIR / folder_name
        if path.exists():
            return path

    # Check project root for backward compat
    for folder_name in [
        model_key.replace("-", "_") + "_npu_cw",
        model_key.replace("-", "_"),
    ]:
        path = PROJECT_ROOT / folder_name
        if path.exists():
            return path

    return None


def detect_available_models():
    """Scan models/ directory and return list of (model_key, path) tuples."""
    available = []
    if not MODEL_DIR.exists():
        return available

    for model_key, info in MODELS.items():
        path = find_model_path(model_key)
        if path:
            available.append((model_key, path))

    # Check for the default .env model name that doesn't match any key
    env_model = os.getenv("MODEL_NAME", "mistral_npu_cw")
    env_path = MODEL_DIR / env_model
    if env_path.exists():
        alias_key = FOLDER_ALIASES.get(env_model)
        if alias_key and not any(k == alias_key for k, _ in available):
            available.append((alias_key, env_path))
        elif not alias_key and not any(str(p) == str(env_path) for _, p in available):
            available.append((env_model, env_path))

    return available


def detect_available_devices():
    """Return list of devices available on this system."""
    core = ov.Core()
    system_devices = core.available_devices
    available = []
    for d in DEVICES:
        if d in system_devices:
            available.append(d)
    return available


def run_single_benchmark(model_path, device, prompt, max_new_tokens, counter,
                         warmup_rounds=WARMUP_ROUNDS, bench_rounds=BENCH_ROUNDS):
    """Run a single inference and return metrics dict."""
    mem_before = get_memory_mb()

    # MAX_PROMPT_LEN is only supported by the NPU plugin
    if device == "NPU":
        pipe = ov_genai.LLMPipeline(str(model_path), device, MAX_PROMPT_LEN=MAX_PROMPT_LEN)
    else:
        pipe = ov_genai.LLMPipeline(str(model_path), device)

    mem_after_load = get_memory_mb()

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = False  # greedy for reproducibility

    # Warmup
    for _ in range(warmup_rounds):
        pipe.generate(prompt, config)

    # Benchmark rounds
    latencies = []
    token_counts = []
    total_times = []

    for _ in range(bench_rounds):
        counter.start()
        pipe.generate(prompt, config, streamer=counter)
        total_time = time.perf_counter() - counter.start_time

        latencies.append(counter.first_token_latency)
        token_counts.append(counter.token_count)
        total_times.append(total_time)

    mem_peak = get_memory_mb()

    avg_tokens = sum(token_counts) / len(token_counts)
    avg_time = sum(total_times) / len(total_times)
    avg_latency = sum(l for l in latencies if l is not None) / max(sum(1 for l in latencies if l is not None), 1)

    tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0
    model_mem = (mem_after_load - mem_before) if (mem_before and mem_after_load) else None
    peak_mem = mem_peak

    del pipe

    return {
        "tokens_per_sec": round(tokens_per_sec, 2),
        "first_token_latency_ms": round(avg_latency * 1000, 1) if avg_latency else None,
        "avg_output_tokens": round(avg_tokens, 1),
        "avg_total_time_s": round(avg_time, 2),
        "model_memory_mb": round(model_mem, 1) if model_mem else None,
        "peak_memory_mb": round(peak_mem, 1) if peak_mem else None,
    }


def get_system_info():
    """Collect system information for the benchmark report."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "openvino": ov.__version__ if hasattr(ov, "__version__") else "unknown",
    }

    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        info["total_ram_gb"] = round(mem.total / (1024 ** 3), 1)

    try:
        core = ov.Core()
        info["available_devices"] = core.available_devices
    except Exception:
        pass

    return info


def print_table_1(results):
    """Print Table 1: NPU Inference Performance (Markdown)."""
    print("\n### NPU Inference Performance\n")
    print("| Model | Parameters | Quantization | Tokens/s (NPU) | First Token Latency | Memory Usage |")
    print("|-------|-----------|-------------|----------------|---------------------|-------------|")

    for model_key, metrics in results.items():
        params = MODELS.get(model_key, {}).get("params", "?")
        tps = f"{metrics['tokens_per_sec']:.2f}" if metrics else "SKIP"
        ftl = f"{metrics['first_token_latency_ms']:.0f} ms" if metrics and metrics.get("first_token_latency_ms") else "N/A"
        mem = f"{metrics['peak_memory_mb']:.0f} MB" if metrics and metrics.get("peak_memory_mb") else "N/A"
        print(f"| {model_key} | {params} | INT4 | {tps} | {ftl} | {mem} |")


def print_table_2(results):
    """Print Table 2: NPU vs CPU vs GPU (Markdown)."""
    print("\n### NPU vs CPU vs GPU (Mistral-7B INT4)\n")
    print("| Device | Tokens/s | First Token Latency | Peak Memory |")
    print("|--------|---------|---------------------|-------------|")

    for device, metrics in results.items():
        if metrics is None:
            print(f"| {device} | N/A (not available) | N/A | N/A |")
            continue
        tps = f"{metrics['tokens_per_sec']:.2f}"
        ftl = f"{metrics['first_token_latency_ms']:.0f} ms" if metrics.get("first_token_latency_ms") else "N/A"
        mem = f"{metrics['peak_memory_mb']:.0f} MB" if metrics.get("peak_memory_mb") else "N/A"
        print(f"| {device} | {tps} | {ftl} | {mem} |")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM inference on Intel NPU / CPU / GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/benchmark.py                     # Run all available models on NPU
  python src/benchmark.py --models mistral-7b # Benchmark a specific model
  python src/benchmark.py --devices NPU CPU   # Compare NPU vs CPU
  python src/benchmark.py --prompt long       # Use a longer prompt
  python src/benchmark.py --rounds 5          # More rounds for stable results
  python src/benchmark.py --max-tokens 256    # Generate more tokens per run
  python src/benchmark.py --save              # Save results to benchmarks/ as JSON
        """,
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Model keys to benchmark (e.g., mistral-7b phi3-mini). Default: all available.",
    )
    parser.add_argument(
        "--devices", nargs="*", default=None,
        help="Devices to test (e.g., NPU CPU GPU). Default: NPU only for Table 1.",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT_KEY, choices=PROMPTS.keys(),
        help=f"Prompt length to use (default: {DEFAULT_PROMPT_KEY}).",
    )
    parser.add_argument(
        "--custom-prompt", default=None,
        help="Use a custom prompt string instead of built-in prompts.",
    )
    parser.add_argument(
        "--rounds", type=int, default=BENCH_ROUNDS,
        help=f"Number of benchmark rounds per test (default: {BENCH_ROUNDS}).",
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_ROUNDS,
        help=f"Number of warmup rounds before benchmarking (default: {WARMUP_ROUNDS}).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_NEW_TOKENS,
        help=f"Max new tokens to generate per run (default: {MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--compare-devices", action="store_true",
        help="Run Table 2: compare NPU vs CPU vs GPU on a single model (default: mistral-7b).",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results as JSON to benchmarks/ directory.",
    )

    args = parser.parse_args()

    prompt = args.custom_prompt if args.custom_prompt else PROMPTS[args.prompt]

    if not HAS_PSUTIL:
        print("[WARN] psutil not installed — memory measurements will be unavailable.")
        print("       Install with: pip install psutil\n")

    # System info
    sys_info = get_system_info()
    print("=" * 60)
    print("  Mistral-for-NPU Benchmark")
    print("=" * 60)
    print(f"  Platform    : {sys_info['platform']}")
    print(f"  Processor   : {sys_info['processor']}")
    print(f"  Python      : {sys_info['python']}")
    print(f"  OpenVINO    : {sys_info['openvino']}")
    if "total_ram_gb" in sys_info:
        print(f"  RAM         : {sys_info['total_ram_gb']} GB")
    if "available_devices" in sys_info:
        print(f"  OV Devices  : {sys_info['available_devices']}")
    print(f"  Prompt      : {args.prompt} ({len(prompt)} chars)")
    print(f"  Max Tokens  : {args.max_tokens}")
    print(f"  Rounds      : {args.rounds} (warmup: {args.warmup})")
    print(f"  Timestamp   : {datetime.now().isoformat()}")
    print("=" * 60)

    available_models = detect_available_models()
    available_devices = detect_available_devices()

    if not available_models:
        print("\n[!] No models found in models/ directory.")
        print("    Download a model first: python src/download.py mistral-7b")
        sys.exit(1)

    print(f"\nModels found: {[k for k, _ in available_models]}")
    print(f"Devices available: {available_devices}\n")

    counter = TokenCounter()

    # ── Table 1: All models on NPU ──
    if not args.compare_devices:
        device = "NPU"
        if args.devices:
            device = args.devices[0]

        if device not in available_devices:
            print(f"[!] Device '{device}' not available. Available: {available_devices}")
            sys.exit(1)

        target_models = available_models
        if args.models:
            target_models = [(k, p) for k, p in available_models if k in args.models]
            if not target_models:
                print(f"[!] None of the specified models are available: {args.models}")
                print(f"    Available: {[k for k, _ in available_models]}")
                sys.exit(1)

        table1_results = {}
        for model_key, model_path in target_models:
            print(f"[*] Benchmarking {model_key} on {device}...")
            print(f"    Path: {model_path}")
            try:
                metrics = run_single_benchmark(
                    model_path, device, prompt, args.max_tokens, counter,
                    warmup_rounds=args.warmup, bench_rounds=args.rounds,
                )
                table1_results[model_key] = metrics
                print(f"    -> {metrics['tokens_per_sec']} tokens/s, "
                      f"first token: {metrics['first_token_latency_ms']} ms, "
                      f"peak mem: {metrics['peak_memory_mb']} MB")
            except Exception as e:
                print(f"    [!] Failed: {e}")
                table1_results[model_key] = None

        print("\n" + "=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print_table_1(table1_results)

        if args.save:
            save_results("table1", table1_results, sys_info, args)

    # ── Table 2: Single model across devices ──
    if args.compare_devices:
        # Find the model to compare
        compare_model = "mistral-7b"
        if args.models:
            compare_model = args.models[0]

        model_path = find_model_path(compare_model)
        if not model_path:
            # Fallback: use the first available model
            if available_models:
                compare_model, model_path = available_models[0]
                print(f"[WARN] Requested model not found, using {compare_model}")
            else:
                print("[!] No models available for device comparison.")
                sys.exit(1)

        devices_to_test = args.devices if args.devices else [d for d in DEVICES if d in available_devices]

        table2_results = {}
        for device in devices_to_test:
            if device not in available_devices:
                print(f"[*] Skipping {device} (not available)")
                table2_results[device] = None
                continue

            print(f"[*] Benchmarking {compare_model} on {device}...")
            try:
                metrics = run_single_benchmark(
                    model_path, device, prompt, args.max_tokens, counter,
                    warmup_rounds=args.warmup, bench_rounds=args.rounds,
                )
                table2_results[device] = metrics
                print(f"    -> {metrics['tokens_per_sec']} tokens/s, "
                      f"first token: {metrics['first_token_latency_ms']} ms")
            except Exception as e:
                print(f"    [!] Failed on {device}: {e}")
                table2_results[device] = None

        print("\n" + "=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print_table_2(table2_results)

        if args.save:
            save_results("table2", table2_results, sys_info, args)

    print()


def save_results(table_name, results, sys_info, args):
    """Save benchmark results to a JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"benchmark_{table_name}_{timestamp}.json"

    data = {
        "table": table_name,
        "timestamp": datetime.now().isoformat(),
        "system_info": sys_info,
        "config": {
            "prompt_key": args.prompt,
            "max_tokens": args.max_tokens,
            "rounds": args.rounds,
            "warmup": args.warmup,
        },
        "results": {},
    }

    # Convert results — handle None values for JSON serialization
    for key, metrics in results.items():
        data["results"][key] = metrics

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to {filename}")


if __name__ == "__main__":
    main()
