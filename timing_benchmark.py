import time
import os
import contextlib
import statistics
from llama_cpp import Llama
import argparse


def load_model(model_path, n_ctx=2048, n_threads=8, n_gpu_layers=1):
    """Load the LLaMA model with error output suppressed."""
    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )


def run_once(llm, prompt, max_tokens=64):
    """Run a single benchmark iteration and return TTFT and TPS."""
    messages = [{"role": "user", "content": prompt}]

    start = time.perf_counter()

    stream = llm.create_chat_completion(
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
    )

    first_token_time = None
    token_count = 0

    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter()

        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            token_count += 1

    end = time.perf_counter()

    ttft = first_token_time - start if first_token_time else None
    tps = token_count / (end - first_token_time) if first_token_time else 0

    return ttft, tps


def benchmark(llm, prompt, runs=10, warmup=2, max_tokens=64):
    """Run the benchmark with specified parameters and return statistics."""
    ttfts = []
    tpss = []

    print("\n--- warmup ---")
    for _ in range(warmup):
        run_once(llm, prompt, max_tokens)

    print("\n--- benchmark ---")

    for i in range(runs):
        ttft, tps = run_once(llm, prompt, max_tokens)

        ttfts.append(ttft)
        tpss.append(tps)

        print(f"run {i + 1}/{runs} | TTFT={ttft:.3f} s | TPS={tps:.2f} tok/s")

    return {
        "ttft_mean": statistics.mean(ttfts),
        "ttft_std": statistics.pstdev(ttfts),
        "tps_mean": statistics.mean(tpss),
        "tps_std": statistics.pstdev(tpss),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Local LLM benchmark")

    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model")
    parser.add_argument("--ctx", type=int, default=2048, help="Context size")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads")
    parser.add_argument("--gpu", type=int, default=1, help="GPU layers")
    parser.add_argument("--runs", type=int, default=10, help="Benchmark runs")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs")
    parser.add_argument("--tokens", type=int, default=64, help="Max tokens")
    parser.add_argument(
        "--prompt", type=str, default="Explain KV cache in one sentence."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    llm = load_model(
        model_path=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads,
        n_gpu_layers=args.gpu,
    )

    stats = benchmark(
        llm,
        prompt=args.prompt,
        runs=args.runs,
        warmup=args.warmup,
        max_tokens=args.tokens,
    )

    print("\n--- summary ---")
    print(f"TTFT mean: {stats['ttft_mean']:.3f} s")
    print(f"TTFT std:  {stats['ttft_std']:.3f} s")
    print(f"TPS mean:  {stats['tps_mean']:.2f} tok/s")
    print(f"TPS std:   {stats['tps_std']:.2f} tok/s")
