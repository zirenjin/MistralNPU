"""
Benchmark Script for NPU Chat
Measures: tokens/s, first token latency, memory usage, model load time
Usage: python src/benchmark.py [--device NPU] [--model mistral_npu_cw] [--runs 3]
"""

import openvino_genai as ov_genai
import os
import sys
import time
import argparse
import json
import platform
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

# --- Test prompts (short / medium / long) to cover different scenarios ---
TEST_PROMPTS = {
    "short": "What is 2 + 2?",
    "medium": "Explain the difference between TCP and UDP in networking. Keep it concise.",
    "long": (
        "Write a detailed comparison of Python and Rust programming languages. "
        "Cover performance, memory safety, ecosystem, learning curve, and use cases. "
        "Provide specific examples for each point."
    ),
}


def get_memory_mb():
    """Get current process memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback for Linux: read from /proc
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # KB to MB
        except FileNotFoundError:
            pass
    return -1


def get_system_info():
    """Collect system info for benchmark context"""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "Unknown",
        "python": platform.python_version(),
    }

    # Try to get CPU model name on Linux
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu"] = line.split(":")[1].strip()
                    break
    except FileNotFoundError:
        pass

    # Try to get total RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["ram_gb"] = round(int(line.split()[1]) / 1024**2, 1)
                        break
        except FileNotFoundError:
            pass

    # OpenVINO version
    try:
        import openvino
        info["openvino"] = openvino.__version__
    except Exception:
        pass

    return info


def find_model_path(model_name):
    """Find model path"""
    for base in [MODEL_DIR, PROJECT_ROOT]:
        path = base / model_name
        if path.exists():
            return str(path)
    return None


def count_tokens_approx(text):
    """Approximate token count (rough: ~4 chars per token for English)"""
    return max(1, len(text) // 4)


def run_single_benchmark(pipe, prompt, max_new_tokens, config):
    """Run a single generation and measure performance"""
    tokens = []
    token_times = []
    first_token_time = None

    start_time = time.perf_counter()

    def token_callback(token_text):
        nonlocal first_token_time
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now
        tokens.append(token_text)
        token_times.append(now)
        return False  # don't stop

    # Use streamer to capture individual tokens
    pipe.generate(prompt, config, streamer=token_callback)

    end_time = time.perf_counter()

    total_time = end_time - start_time
    num_tokens = len(tokens)
    output_text = "".join(tokens)

    # First token latency (time from start to first token)
    ttft = (first_token_time - start_time) if first_token_time else total_time

    # Tokens per second (exclude first token time for decode speed)
    if num_tokens > 1 and first_token_time:
        decode_time = end_time - first_token_time
        decode_tps = (num_tokens - 1) / decode_time if decode_time > 0 else 0
    else:
        decode_tps = num_tokens / total_time if total_time > 0 else 0

    overall_tps = num_tokens / total_time if total_time > 0 else 0

    return {
        "num_tokens": num_tokens,
        "total_time_s": round(total_time, 2),
        "ttft_s": round(ttft, 3),
        "decode_tps": round(decode_tps, 2),
        "overall_tps": round(overall_tps, 2),
        "output_preview": output_text[:100] + "..." if len(output_text) > 100 else output_text,
    }


def run_benchmark(device, model_name, num_runs, max_new_tokens, max_prompt_len):
    """Run full benchmark suite"""
    model_path = find_model_path(model_name)
    if not model_path:
        print(f"[!] Model '{model_name}' not found in {MODEL_DIR} or {PROJECT_ROOT}")
        print("    Run 'python src/download.py --list' to see available models.")
        sys.exit(1)

    print("=" * 60)
    print("  NPU Chat Benchmark")
    print("=" * 60)

    # System info
    sys_info = get_system_info()
    print("\n[System Info]")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    print(f"\n[Benchmark Config]")
    print(f"  Model:          {model_name}")
    print(f"  Model path:     {model_path}")
    print(f"  Device:         {device}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Max prompt len: {max_prompt_len}")
    print(f"  Runs per prompt: {num_runs}")

    # Measure model load time
    mem_before_load = get_memory_mb()
    print(f"\n[Loading model on {device}...]")
    load_start = time.perf_counter()

    try:
        pipe = ov_genai.LLMPipeline(model_path, device, MAX_PROMPT_LEN=max_prompt_len)
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        sys.exit(1)

    load_time = time.perf_counter() - load_start
    mem_after_load = get_memory_mb()

    print(f"  Load time:      {load_time:.2f}s")
    if mem_before_load > 0 and mem_after_load > 0:
        print(f"  Memory before:  {mem_before_load:.0f} MB")
        print(f"  Memory after:   {mem_after_load:.0f} MB")
        print(f"  Model memory:   {mem_after_load - mem_before_load:.0f} MB")

    # Generation config
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = False  # Greedy for reproducibility
    config.temperature = 1.0

    # Run benchmarks
    all_results = {}

    for prompt_name, prompt_text in TEST_PROMPTS.items():
        print(f"\n{'─' * 60}")
        print(f"  Prompt: [{prompt_name}] \"{prompt_text[:60]}{'...' if len(prompt_text) > 60 else ''}\"")
        print(f"{'─' * 60}")

        runs = []
        for i in range(num_runs):
            pipe.start_chat()
            mem_before = get_memory_mb()

            result = run_single_benchmark(pipe, prompt_text, max_new_tokens, config)

            mem_after = get_memory_mb()
            result["mem_during_mb"] = round(mem_after, 0) if mem_after > 0 else -1

            pipe.finish_chat()
            runs.append(result)

            print(f"\n  Run {i+1}/{num_runs}:")
            print(f"    Tokens generated: {result['num_tokens']}")
            print(f"    First token:      {result['ttft_s']:.3f}s")
            print(f"    Decode speed:     {result['decode_tps']:.2f} tokens/s")
            print(f"    Overall speed:    {result['overall_tps']:.2f} tokens/s")
            print(f"    Total time:       {result['total_time_s']:.2f}s")
            if result["mem_during_mb"] > 0:
                print(f"    Memory usage:     {result['mem_during_mb']:.0f} MB")

        # Calculate averages
        avg = {
            "num_tokens": round(sum(r["num_tokens"] for r in runs) / len(runs), 1),
            "ttft_s": round(sum(r["ttft_s"] for r in runs) / len(runs), 3),
            "decode_tps": round(sum(r["decode_tps"] for r in runs) / len(runs), 2),
            "overall_tps": round(sum(r["overall_tps"] for r in runs) / len(runs), 2),
            "total_time_s": round(sum(r["total_time_s"] for r in runs) / len(runs), 2),
        }

        all_results[prompt_name] = {"runs": runs, "average": avg}

        print(f"\n  >> Average ({num_runs} runs):")
        print(f"     First token:   {avg['ttft_s']:.3f}s")
        print(f"     Decode speed:  {avg['decode_tps']:.2f} tokens/s")
        print(f"     Overall speed: {avg['overall_tps']:.2f} tokens/s")

    # Summary table
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  Model:  {model_name}")
    print(f"  Device: {device}")
    print(f"  Load:   {load_time:.2f}s")
    if mem_after_load > 0 and mem_before_load > 0:
        print(f"  Memory: {mem_after_load - mem_before_load:.0f} MB (model only)")

    print(f"\n  {'Prompt':<10} {'TTFT':>8} {'Decode t/s':>12} {'Overall t/s':>13} {'Tokens':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*12} {'─'*13} {'─'*8}")
    for name, data in all_results.items():
        avg = data["average"]
        print(f"  {name:<10} {avg['ttft_s']:>7.3f}s {avg['decode_tps']:>11.2f} {avg['overall_tps']:>12.2f} {avg['num_tokens']:>8.0f}")

    # Save results to JSON
    output = {
        "system_info": sys_info,
        "config": {
            "model": model_name,
            "device": device,
            "max_new_tokens": max_new_tokens,
            "max_prompt_len": max_prompt_len,
            "num_runs": num_runs,
        },
        "load_time_s": round(load_time, 2),
        "memory_mb": round(mem_after_load - mem_before_load, 0) if mem_after_load > 0 and mem_before_load > 0 else None,
        "results": all_results,
    }

    results_dir = PROJECT_ROOT / "benchmarks"
    results_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"bench_{device.lower()}_{model_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {result_file}")
    print(f"{'=' * 60}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference on NPU/CPU/GPU")
    parser.add_argument("--device", default=os.getenv("DEVICE", "NPU"),
                        help="Device: NPU, CPU, or GPU (default: from .env or NPU)")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "mistral_npu_cw"),
                        help="Model folder name (default: from .env or mistral_npu_cw)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per prompt (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max new tokens per generation (default: 256)")
    parser.add_argument("--max-prompt-len", type=int,
                        default=int(os.getenv("MAX_PROMPT_LEN", "4096")),
                        help="Max prompt length (default: from .env or 4096)")
    parser.add_argument("--compare", action="store_true",
                        help="Run on all devices (NPU, CPU, GPU) for comparison")

    args = parser.parse_args()

    if args.compare:
        devices = ["NPU", "CPU", "GPU"]
        print(f"Running comparison benchmark across: {', '.join(devices)}")
        print("This will take a while...\n")
        for device in devices:
            try:
                run_benchmark(device, args.model, args.runs, args.max_tokens, args.max_prompt_len)
            except Exception as e:
                print(f"\n[!] {device} benchmark failed: {e}")
            print("\n")
    else:
        run_benchmark(args.device, args.model, args.runs, args.max_tokens, args.max_prompt_len)


if __name__ == "__main__":
    main()
