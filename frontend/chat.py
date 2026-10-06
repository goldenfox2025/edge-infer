#!/usr/bin/env python3
import sys
import os
import threading
import queue
import time
import argparse
from pathlib import Path
if __package__:
    from .checkpoint import MODEL_TYPES, load_model
else:
    from checkpoint import MODEL_TYPES, load_model

TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

TOKENIZERS_AVAILABLE = False
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    pass

def load_tokenizer(model_path: str, model_type: str):
    """Load the tokenizer for the selected model type."""
    model_path = Path(model_path)

    if model_type.startswith("qwen"):
        if not TRANSFORMERS_AVAILABLE:
            print("Error: transformers library required for Qwen models")
            exit(1)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print(f"Qwen tokenizer loaded from: {model_path}")
    else:  # llama
        if not TOKENIZERS_AVAILABLE:
            print("Error: tokenizers library required for Llama models")
            exit(1)
        tokenizer_path = model_path / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"Llama tokenizer loaded from: {tokenizer_path}")

    return tokenizer

# -------------------------------

# -------------------------------
def create_callback(q: queue.Queue):
    """Enqueue generated token IDs without decoding or collecting statistics."""
    def token_callback(token_id):
        q.put(token_id)
    return token_callback

def decode_stream_incremental(tokenizer, token_ids, rendered_text: str):
    decode_kwargs = {"skip_special_tokens": False}
    if not TOKENIZERS_AVAILABLE or not isinstance(tokenizer, Tokenizer):
        decode_kwargs["clean_up_tokenization_spaces"] = False

    decoded_text = tokenizer.decode(token_ids, **decode_kwargs)

    prefix_len = 0
    max_prefix = min(len(rendered_text), len(decoded_text))
    while prefix_len < max_prefix and rendered_text[prefix_len] == decoded_text[prefix_len]:
        prefix_len += 1

    pending_suffix = decoded_text[prefix_len:]
    replacement_pos = pending_suffix.find("\ufffd")
    if replacement_pos != -1:
        safe_suffix = pending_suffix[:replacement_pos]
        stable_text = decoded_text[:prefix_len + len(safe_suffix)]
        return safe_suffix, stable_text

    return pending_suffix, decoded_text

# -------------------------------

# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLaMA/Qwen/Qwen3 model chat")
    parser.add_argument('--model_path', type=str, default="./models/Qwen3-1.7B-AWQ", help="Model path")
    parser.add_argument('--model_type', type=str, default="qwen3_awq",
                       choices=MODEL_TYPES,
                       help="Model type")
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'], help="Execution device (cuda or cpu)")
    parser.add_argument('--system_prompt', type=str, default="You are a helpful AI assistant.", help="System prompt")
    parser.add_argument('--max_new_tokens', '--max_length', dest='max_new_tokens', type=int,
                       default=424, help="Maximum generated token count")
    parser.add_argument('--context_capacity', type=int, default=None,
                       help="Session token capacity (default: min(4096, model limit))")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature")
    parser.add_argument('--top_p', type=float, default=1, help="top-p sampling threshold")
    parser.add_argument('--top_k', type=int, default=20, help="top-k sampling threshold")
    args = parser.parse_args()

    config, weights, model_type = load_model(args.model_path, args.model_type)

    tokenizer = load_tokenizer(args.model_path, model_type)

    total_params = sum(t.numel() for t in weights.values())
    total_bytes = sum(t.element_size() * t.numel() for t in weights.values())
    print("\nModel size: {} parameters, {:.2f} MB".format(total_params, total_bytes / (1024 * 1024)))

    print("\nModel Configuration:")
    print(f"Model Type: {model_type}")
    if model_type.startswith("qwen"):
        precision = "BF16" if "bf16" in model_type or "awq" in model_type else "FP32"
        if "awq" in model_type:
            precision += " (AWQ Quantized)"
        print(f"Precision: {precision}")

    print(f"Hidden Size: {config['hidden_size']}")
    print(f"Num Layers: {config['num_hidden_layers']}")
    print(f"Num Attention Heads: {config['num_attention_heads']}")
    print(f"Num KV Heads: {config['num_key_value_heads']}")
    head_dim = config.get("head_dim", config['hidden_size'] // config['num_attention_heads'])
    print(f"Head Dimension: {head_dim}")

    print(f"Requested Device: {args.device}")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, os.environ.get("BUILD_DIR", str(repo_root / "build")))
    from model_bridge import Model

    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be positive")
    model = Model(config, weights, model_type, device=args.device)
    session = model.new_session(capacity=args.context_capacity)
    del weights
    print(f"Model initialized on {model.device}; context capacity: {session.capacity} tokens.\n")
    messages = [{"role": "system", "content": args.system_prompt}]

    print("Chat started. Type 'quit' or 'exit' to quit.\n")

    while True:
        user_message = input("User: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break
        if not user_message:
            continue

        messages.append({"role": "user", "content": user_message})
        if model_type.startswith("qwen"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer([text], return_tensors="pt")["input_ids"][0].tolist()
        else:
            conversation = "".join(
                f"<|{message['role']}|>\n{message['content']}</s>\n"
                for message in messages
            ) + "<|assistant|>\n"
            input_ids = tokenizer.encode(conversation).ids

        if len(input_ids) >= session.capacity:
            messages.pop()
            print("Conversation exceeds the session capacity. Start a new chat or use a larger capacity.")
            continue
        max_length = min(session.capacity, len(input_ids) + args.max_new_tokens)

        q = queue.Queue()
        callback = create_callback(q)

        generation_errors = []

        def run_generation():
            try:
                session.generate(
                    input_ids, callback, max_length=max_length,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
                )
            except Exception as error:
                generation_errors.append(error)
            finally:
                q.put(None)

        thread = threading.Thread(target=run_generation)
        thread.start()

        print("Assistant: ", end="", flush=True)

        # Decode tokens and collect timing statistics in the main thread.
        accumulated_tokens = []
        rendered_text = ""
        start_time = None
        total_tokens = 0

        while True:
            token_id = q.get()
            if token_id is None:
                break
            current_time = time.monotonic()
            if start_time is None:
                start_time = current_time

            accumulated_tokens.append(token_id)
            total_tokens += 1

            diff, rendered_text = decode_stream_incremental(tokenizer, accumulated_tokens, rendered_text)

            if diff:
                print(diff, end="", flush=True)

        thread.join()
        if generation_errors:
            messages.pop()
            print(f"\nGeneration failed: {generation_errors[0]}", file=sys.stderr)
            continue
        assistant_text = tokenizer.decode(accumulated_tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": assistant_text})

        final_text, rendered_text = decode_stream_incremental(tokenizer, accumulated_tokens, rendered_text)
        if final_text:
            print(final_text, end="", flush=True)

        total_time = current_time - start_time if start_time is not None else 0.0
        avg_speed = total_tokens / total_time if total_time > 0 else 0.0

        print("\n")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Total Tokens: {total_tokens}")
        print(f"Average Speed: {avg_speed:.2f} tokens/sec\n")

if __name__ == "__main__":
    main()
