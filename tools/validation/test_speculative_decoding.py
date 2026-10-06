#!/usr/bin/env python3
"""Compare ordinary and exact greedy speculative tokens for local checkpoints."""

import argparse
import os
from pathlib import Path
import sys
import time


def collect(session, input_ids, max_length):
    tokens = []
    start = time.perf_counter()
    session.generate(
        input_ids, tokens.append, max_length=max_length,
        temperature=1.0, top_p=1.0, top_k=1
    )
    return tokens, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--model_type", choices=("qwen_bf16", "qwen_awq", "qwen3_bf16", "qwen3_awq"),
                        default="qwen3_awq")
    parser.add_argument("--context_capacity", type=int, default=4096)
    parser.add_argument("--spec_length", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Explain how a KV cache speeds up decoding.")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, os.environ.get("BUILD_DIR", str(repo_root / "build")))
    from frontend.checkpoint import load_model
    from model_bridge import Model
    from transformers import AutoTokenizer

    target_config, target_weights, _ = load_model(args.target, args.model_type)
    target = Model(target_config, target_weights, args.model_type, device="cuda")
    del target_weights
    draft_config, draft_weights, _ = load_model(args.draft, args.model_type)
    draft = Model(draft_config, draft_weights, args.model_type, device="cuda")
    del draft_weights
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    messages = [{"role": "user", "content": args.prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    max_length = len(input_ids) + args.max_new_tokens
    if max_length > args.context_capacity:
        parser.error("Prompt and generated-token budget exceed --context_capacity")

    ordinary = target.new_session(capacity=args.context_capacity)
    speculative = target.new_speculative_session(
        draft, capacity=args.context_capacity, spec_length=args.spec_length
    )
    expected, ordinary_seconds = collect(ordinary, input_ids, max_length)
    actual, speculative_seconds = collect(speculative, input_ids, max_length)
    if actual != expected:
        first = next((index for index, (left, right) in enumerate(zip(expected, actual))
                      if left != right), min(len(expected), len(actual)))
        raise RuntimeError(
            f"Greedy token mismatch at index {first}: ordinary={expected}, speculative={actual}"
        )
    if not expected:
        raise RuntimeError("Both sessions returned no tokens; this does not establish useful generation parity")
    print(f"Greedy tokens match: {len(expected)} tokens")
    print(f"Ordinary elapsed: {ordinary_seconds:.3f}s")
    print(f"Speculative elapsed: {speculative_seconds:.3f}s")
    print("Single un-warmed timing samples are diagnostic, not a benchmark.")
    print(tokenizer.decode(actual, skip_special_tokens=True))


if __name__ == "__main__":
    main()
