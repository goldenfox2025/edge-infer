#!/usr/bin/env python3
import sys
import json
import os
import threading
import queue
import time
import argparse
from pathlib import Path
from safetensors import safe_open
import torch

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

# -------------------------------

# -------------------------------

def load_llama_model(model_path: str):
    """Load the Llama model weights and configuration."""
    model_path = Path(model_path)
    weights = {}

    with safe_open(model_path / "model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # if bfloat16, then convert to float32
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            weights[key] = tensor
            print(f"Loaded tensor {key} with shape {weights[key].shape}")

    with open(model_path / "config.json", 'r') as f:
        config = json.load(f)

        if "model.embed_tokens.weight" not in weights:
            config["tie_word_embeddings"] = True
            weights["model.embed_tokens.weight"] = weights["lm_head.weight"]
        print("Config loaded:", config)

    return config, weights, "llama"

def load_qwen_model(model_path: str, keep_bf16=True, is_awq=False):
    """Load Qwen weights and configuration, preserving BF16 or AWQ when requested."""
    model_path = Path(model_path)
    weights = {}

    with safe_open(model_path / "model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)

            if is_awq:

                if any(suffix in key for suffix in [".qweight", ".scales", ".qzeros"]):
                    # Keep quantized weights in their original representation
                    weights[key] = tensor
                    print(f"Loaded AWQ quantized tensor {key} with shape {tensor.shape} and dtype {tensor.dtype}")
                else:

                    if tensor.dtype == torch.bfloat16 and keep_bf16:
                        weights[key] = tensor
                        print(f"Loaded AWQ non-quantized bf16 tensor {key} with shape {tensor.shape}")
                    else:
                        weights[key] = tensor.to(torch.float32)
                        print(f"Loaded AWQ non-quantized fp32 tensor {key} with shape {weights[key].shape}")

            elif tensor.dtype == torch.bfloat16 and keep_bf16:
                weights[key] = tensor
                print(f"Loaded bf16 tensor {key} with shape {tensor.shape}")
            else:
                weights[key] = tensor.to(torch.float32)
                print(f"Loaded tensor {key} with shape {weights[key].shape}")

    with open(model_path / "config.json", 'r') as f:
        config = json.load(f)
        expected_keys = [
            "vocab_size", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads",
            "intermediate_size", "max_position_embeddings",
            "rms_norm_eps", "rope_theta"
        ]
        for key in expected_keys:
            if key not in config:
                print(f"Warning: {key} not found in config")
        print("Config loaded:", config)

        if is_awq:

            if "quantization_config" in config:
                quant_config = config["quantization_config"]
                if "group_size" in quant_config:
                    config["group_size"] = quant_config["group_size"]
                    print(f"Using group_size from config: {config['group_size']}")
                else:
                    config["group_size"] = 128 # Default value
                    print(f"Using default group_size: {config['group_size']}")
            else:
                config["group_size"] = 128 # Default value
                print(f"Using default group_size: {config['group_size']}")

    if is_awq:
        model_type = "qwen_awq"
    else:
        model_type = "qwen_bf16" if keep_bf16 else "qwen"

    return config, weights, model_type

def load_qwen3_model(model_path: str, keep_bf16=True, is_awq=False):
    """Load Qwen3 weights and configuration, preserving BF16 or AWQ when requested."""
    model_path = Path(model_path)
    weights = {}
    verbose_weights = os.environ.get("EDGE_INFER_VERBOSE_WEIGHTS") == "1"

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        print(f"Found the model index file: {index_path}")
        with open(index_path, 'r') as f:
            index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})

        weights_by_file = {}
        for key, file_name in weight_map.items():
            if file_name not in weights_by_file:
                weights_by_file[file_name] = []
            weights_by_file[file_name].append(key)

        for file_name, keys in weights_by_file.items():
            file_path = model_path / file_name
            if verbose_weights:
                print(f"Load weights from a file: {file_path}")

            with safe_open(file_path, framework="pt") as f:
                for key in keys:
                    tensor = f.get_tensor(key)

                    if is_awq:

                        if any(suffix in key for suffix in [".qweight", ".scales", ".qzeros"]):
                            # Keep quantized weights in their original representation
                            weights[key] = tensor
                            if verbose_weights:
                                print(f"Load AWQ quantized tensor {key}, Shape {tensor.shape}, Data type {tensor.dtype}")
                        else:

                            if tensor.dtype == torch.bfloat16 and keep_bf16:
                                weights[key] = tensor
                                if verbose_weights:
                                    print(f"Load AWQ non-quantized bf16 Tensor {key}, Shape {tensor.shape}")
                            else:
                                weights[key] = tensor.to(torch.float32)
                                if verbose_weights:
                                    print(f"Load AWQ non-quantized fp32 Tensor {key}, Shape {weights[key].shape}")

                    elif tensor.dtype == torch.bfloat16 and keep_bf16:
                        weights[key] = tensor
                        if verbose_weights:
                            print(f"Load bf16 Tensor {key}, Shape {tensor.shape}")
                    else:
                        weights[key] = tensor.to(torch.float32)
                        if verbose_weights:
                            print(f"Load tensor {key}, Shape {weights[key].shape}")
    else:

        safetensors_path = model_path / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Model weight file not found: Neither an index file {index_path}, and no single weight file {safetensors_path}")

        if verbose_weights:
            print(f"Index file not found, Trying to load from a single file: {safetensors_path}")
        with safe_open(safetensors_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if is_awq:

                    if any(suffix in key for suffix in [".qweight", ".scales", ".qzeros"]):
                        # Keep quantized weights in their original representation
                        weights[key] = tensor
                        if verbose_weights:
                            print(f"Load AWQ quantized tensor {key}, Shape {tensor.shape}, Data type {tensor.dtype}")
                    else:

                        if tensor.dtype == torch.bfloat16 and keep_bf16:
                            weights[key] = tensor
                            if verbose_weights:
                                print(f"Load AWQ non-quantized bf16 Tensor {key}, Shape {tensor.shape}")
                        else:
                            weights[key] = tensor.to(torch.float32)
                            if verbose_weights:
                                print(f"Load AWQ non-quantized fp32 Tensor {key}, Shape {weights[key].shape}")

                elif tensor.dtype == torch.bfloat16 and keep_bf16:
                    weights[key] = tensor
                    if verbose_weights:
                        print(f"Load bf16 Tensor {key}, Shape {tensor.shape}")
                else:
                    weights[key] = tensor.to(torch.float32)
                    if verbose_weights:
                        print(f"Load tensor {key}, Shape {weights[key].shape}")

    config_path = model_path / "config.json"
    if verbose_weights:
        print(f"Reading the configuration file: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_str = f.read()
            if verbose_weights:
                print(f"First 100 characters: {config_str[:100]}...")
            config = json.loads(config_str)
            if verbose_weights:
                print("Configuration loaded successfully")
    except Exception as e:
        print(f"Failed to load the configuration file: {e}")
        raise

    if verbose_weights:
        print("Original configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    cpp_config = {}

    for key, value in config.items():
        cpp_config[key] = value

    key_mapping = {
        "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads"
    }

    for orig_key, new_key in key_mapping.items():
        if orig_key in config:
            cpp_config[new_key] = config[orig_key]
            if verbose_weights:
                print(f"Add mapped key: {orig_key} -> {new_key}: {config[orig_key]}")

    if is_awq:
        cpp_config["quant_type"] = 1

        if "quantization_config" in config:
            quant_config = config["quantization_config"]
            if "group_size" in quant_config:
                cpp_config["group_size"] = quant_config["group_size"]
                if verbose_weights:
                    print(f"Using the configured group_size: {cpp_config['group_size']}")
            else:
                cpp_config["group_size"] = 128 # Default value
                if verbose_weights:
                    print("Using the default group_size: 128")
        else:
            cpp_config["group_size"] = 128 # Default value
            if verbose_weights:
                print("Using the default group_size: 128")
    else:
        cpp_config["quant_type"] = 0

    if verbose_weights:
        print("Final configuration:")
        for key, value in cpp_config.items():
            print(f"  {key}: {value}")

    if is_awq:
        model_type = "qwen3_awq"
    else:
        model_type = "qwen3_bf16" if keep_bf16 else "qwen3"

    return cpp_config, weights, model_type

def load_tokenizer(model_path: str, model_type: str):
    """Load the tokenizer for the selected model type."""
    model_path = Path(model_path)

    if model_type.startswith("qwen"):
        if not TRANSFORMERS_AVAILABLE:
            print("Error: transformers library required for Qwen models")
            exit(1)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
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
                       choices=['llama', 'qwen', 'qwen_bf16', 'qwen_awq', 'qwen3_bf16', 'qwen3_awq'],
                       help="Model type")
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'], help="Execution device (cuda or cpu)")
    parser.add_argument('--system_prompt', type=str, default="You are a helpful AI assistant.", help="System prompt")
    parser.add_argument('--max_length', type=int, default=424, help="Maximum generated-text length")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature")
    parser.add_argument('--top_p', type=float, default=1, help="top-p sampling threshold")
    parser.add_argument('--top_k', type=int, default=20, help="top-k sampling threshold")
    args = parser.parse_args()

    if args.model_type == "llama":
        config, weights, model_type = load_llama_model(args.model_path)
    elif args.model_type == "qwen":
        config, weights, model_type = load_qwen_model(args.model_path, keep_bf16=False, is_awq=False)
    elif args.model_type == "qwen_bf16":
        config, weights, model_type = load_qwen_model(args.model_path, keep_bf16=True, is_awq=False)
    elif args.model_type == "qwen_awq":
        config, weights, model_type = load_qwen_model(args.model_path, keep_bf16=True, is_awq=True)
    elif args.model_type == "qwen3_bf16":
        config, weights, model_type = load_qwen3_model(args.model_path, keep_bf16=True, is_awq=False)
    elif args.model_type == "qwen3_awq":
        config, weights, model_type = load_qwen3_model(args.model_path, keep_bf16=True, is_awq=True)
    else:
        print(f"Unsupported model type: {args.model_type}")
        exit(1)

    tokenizer = load_tokenizer(args.model_path, model_type)

    total_params = sum(t.numel() for t in weights.values())
    total_bytes = sum(t.element_size() * t.numel() for t in weights.values())
    print("\nModel size: {} parameters, {:.2f} MB".format(total_params, total_bytes / (1024 * 1024)))

    print("\nModel Configuration:")
    print(f"Model Type: {model_type}")
    if model_type.startswith("qwen"):
        precision = "BF16" if "bf16" in model_type else "FP32"
        if "awq" in model_type:
            precision += " (AWQ Quantized)"
        print(f"Precision: {precision}")

    if model_type.startswith("qwen3"):
        print(f"Hidden Size: {config['hidden_size']}")
        print(f"Num Layers: {config['n_layers']}")
        print(f"Num Attention Heads: {config['n_heads']}")
        print(f"Num KV Heads: {config['n_kv_heads']}")
        print(f"Head Dimension: {config['hidden_size'] // config['n_heads']}")
    else:
        print(f"Hidden Size: {config['hidden_size']}")
        print(f"Num Attention Heads: {config['num_attention_heads']}")
        print(f"Num Key Value Heads: {config['num_key_value_heads']}")
        print(f"Head Dimension: {config['hidden_size'] // config['num_attention_heads']}")

    print(f"Requested Device: {args.device}")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, os.environ.get("BUILD_DIR", str(repo_root / "build")))
    from model_bridge import init_model, generate_text_stream, set_default_device, get_default_device

    sys.path.insert(0, str(repo_root / "bindings" / "python"))
    import device_config

    print(f"\nSet the default device to: {args.device}")
    if not set_default_device(args.device):
        print(f"Failed to set device {args.device}; using the default device", file=sys.stderr)

    current_device = get_default_device()
    print(f"Current device: {current_device}")

    if not init_model(config, weights, model_type):
        print("Model initialization failed.", file=sys.stderr)
        exit(1)
    print("\nModel initialized successfully.\n")

    first_chat = True

    print("Chat started. Type 'quit' or 'exit' to quit.\n")

    while True:
        user_message = input("User: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break
        if not user_message:
            continue

        if model_type.startswith("qwen"):
            if first_chat:
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": user_message}
                ]
                first_chat = False
            else:
                messages = [{"role": "user", "content": user_message}]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt")
            # print("Type of text returned by apply_chat_template:", type(text))
            # print("Content:", text)

            input_ids = model_inputs["input_ids"][0].tolist()
        else:  # llama
            if first_chat:
                conversation = f"<|system|>\n{args.system_prompt}</s><|user|>\n\n{user_message}</s>\n<|assistant|>\n"
                first_chat = False
            else:
                conversation = f"{user_message}</s>\n:<|assistant|>\n"

            if isinstance(tokenizer, Tokenizer):
                encoded = tokenizer.encode(conversation)
                input_ids = encoded.ids
            else:
                model_inputs = tokenizer([conversation], return_tensors="pt")
                input_ids = model_inputs["input_ids"][0].tolist()

        q = queue.Queue()
        callback = create_callback(q)

        def run_generation():
            from model_bridge import generate_text_stream
            generate_text_stream(
                input_ids,
                callback,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k
            )
            q.put(None)

        thread = threading.Thread(target=run_generation)
        thread.start()

        print("Assistant: ", end="", flush=True)

        # Decode tokens and collect timing statistics in the main thread.
        accumulated_tokens = []
        rendered_text = ""
        start_time = None
        last_token_time = None
        total_tokens = 0

        while True:
            token_id = q.get()
            if token_id is None:
                break
            current_time = time.monotonic()
            if start_time is None:
                start_time = current_time

            if last_token_time is not None:
                token_time = current_time - last_token_time

                # print(f"Token time: {token_time:.4f}s", end="\r", flush=True)
            last_token_time = current_time

            accumulated_tokens.append(token_id)
            total_tokens += 1

            diff, rendered_text = decode_stream_incremental(tokenizer, accumulated_tokens, rendered_text)

            if diff:
                print(diff, end="", flush=True)

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
