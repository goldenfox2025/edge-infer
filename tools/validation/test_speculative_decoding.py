#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare standard and speculative decoding with target and draft Qwen3 models."""

import os
import sys
import time
import queue
import threading
from typing import List, Callable
import torch
from pathlib import Path
from safetensors import safe_open

from transformers import AutoTokenizer

def load_qwen3_model(model_path, keep_bf16=True, is_awq=True):
    """Load Qwen3 weights and configuration, preserving BF16 or AWQ when requested."""
    import json

    model_path = Path(model_path)
    weights = {}

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
            print(f"Load weights from a file: {file_path}")

            with safe_open(file_path, framework="pt") as f:
                for key in keys:
                    tensor = f.get_tensor(key)

                    if is_awq:

                        if any(suffix in key for suffix in [".qweight", ".scales", ".qzeros"]):
                            # Keep quantized weights in their original representation
                            weights[key] = tensor
                            print(f"Load AWQ quantized tensor {key}, Shape {tensor.shape}, Data type {tensor.dtype}")
                        else:

                            if tensor.dtype == torch.bfloat16 and keep_bf16:
                                weights[key] = tensor
                                print(f"Load AWQ non-quantized bf16 Tensor {key}, Shape {tensor.shape}")
                            else:
                                weights[key] = tensor.to(torch.float32)
                                print(f"Load AWQ non-quantized fp32 Tensor {key}, Shape {weights[key].shape}")

                    elif tensor.dtype == torch.bfloat16 and keep_bf16:
                        weights[key] = tensor
                        print(f"Load bf16 Tensor {key}, Shape {tensor.shape}")
                    else:
                        weights[key] = tensor.to(torch.float32)
                        print(f"Load tensor {key}, Shape {weights[key].shape}")
    else:

        safetensors_path = model_path / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Model weight file not found: Neither an index file {index_path}, and no single weight file {safetensors_path}")

        print(f"Index file not found, Trying to load from a single file: {safetensors_path}")
        with safe_open(safetensors_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if is_awq:

                    if any(suffix in key for suffix in [".qweight", ".scales", ".qzeros"]):
                        # Keep quantized weights in their original representation
                        weights[key] = tensor
                        print(f"Load AWQ quantized tensor {key}, Shape {tensor.shape}, Data type {tensor.dtype}")
                    else:

                        if tensor.dtype == torch.bfloat16 and keep_bf16:
                            weights[key] = tensor
                            print(f"Load AWQ non-quantized bf16 Tensor {key}, Shape {tensor.shape}")
                        else:
                            weights[key] = tensor.to(torch.float32)
                            print(f"Load AWQ non-quantized fp32 Tensor {key}, Shape {weights[key].shape}")

                elif tensor.dtype == torch.bfloat16 and keep_bf16:
                    weights[key] = tensor
                    print(f"Load bf16 Tensor {key}, Shape {tensor.shape}")
                else:
                    weights[key] = tensor.to(torch.float32)
                    print(f"Load tensor {key}, Shape {weights[key].shape}")

    config_path = model_path / "config.json"
    print(f"Reading the configuration file: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_str = f.read()
            print(f"First 100 characters: {config_str[:100]}...")
            config = json.loads(config_str)
            print("Configuration loaded successfully")
    except Exception as e:
        print(f"Failed to load the configuration file: {e}")
        raise

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
            print(f"Add mapped key: {orig_key} -> {new_key}: {config[orig_key]}")

    if is_awq:
        cpp_config["quant_type"] = 1

        if "quantization_config" in config:
            quant_config = config["quantization_config"]
            if "group_size" in quant_config:
                cpp_config["group_size"] = quant_config["group_size"]
                print(f"Using the configured group_size: {cpp_config['group_size']}")
            else:
                cpp_config["group_size"] = 128 # Default value
                print(f"Using the default group_size: 128")
        else:
            cpp_config["group_size"] = 128 # Default value
            print(f"Using the default group_size: 128")
    else:
        cpp_config["quant_type"] = 0

    print("Final configuration:")
    for key, value in cpp_config.items():
        print(f"  {key}: {value}")

    return cpp_config, weights

def create_callback(q):
    """Create a callback that enqueues generated token IDs."""
    def callback(token_id):
        q.put(token_id)
    return callback

def main():

    logits_dirs = ["./logits_data", "./logits_data/target", "./logits_data/draft", "./logits_data/visualizations"]
    for dir_path in logits_dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Created logits Data directory")

    target_model_path = "./models/Qwen3-1.7B-AWQ"
    draft_model_path = "./models/Qwen3-0.6B-AWQ"
    print(f"Target model path: {target_model_path}")
    print(f"Draft model path: {draft_model_path}")

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, os.environ.get("BUILD_DIR", str(repo_root / "build")))
    from model_bridge import (
        init_model, generate_text_stream, set_default_device,
        init_speculative_decoder, generate_text_stream_speculative
    )

    print("Set the device to: cuda")
    set_default_device("cuda")

    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            memory_info = result.stdout.strip().split(', ')
            total_mem, used_mem, free_mem = map(int, memory_info)
            print(f"[GPU Memory] Total: {total_mem} MB, Used: {used_mem} MB, Free: {free_mem} MB")
            if free_mem < 3000:
                print("Warning: GPU There may not be enough free memory, Consider releasing other CUDA program")
    except:
        print("Cannot retrieve GPU Memory information")

    print("[Debug] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    print("\u3010 Debug \u3011tokenizer Loading complete")

    print(f"[Debug] Loading the target model: {target_model_path}")
    target_config, target_weights = load_qwen3_model(target_model_path)
    print("[Debug] Target model weights loaded, Initializing...")

    model_type = "qwen3_awq"
    print(f"[Debug] Initializing the target model, Type: {model_type}")
    if not init_model(target_config, target_weights, model_type):
        print("Target model initialization failed")
        return
    print("\u3010 Debug \u3011 Target model initialized successfully")

    print(f"[Debug] Loading the draft model: {draft_model_path}")
    draft_config, draft_weights = load_qwen3_model(draft_model_path)
    print("[Debug] Draft model weights loaded, Initializing the speculative decoder...")
    model_type = "qwen3_awq"

    spec_length = 8
    print(f"[Debug] Initializing the speculative decoder, Speculative length: {spec_length}")
    if not init_speculative_decoder(draft_config, draft_weights, model_type, spec_length):
        print("Speculative decoder initialization failed")
        return
    print("\u3010 Debug \u3011 Speculative decoder initialized successfully")

    system_prompt = "\u4f60\u662f\u4e00\u4e2a\u6709\u7528\u7684AI\u52a9\u624b\u3002"
    user_input = "\u7ed9\u6211\u8bb2\u8ff0\u4e00\u4e2a\u6709\u8da3\u7684\u9ed1\u6697\u4e4b\u9b42\u6545\u4e8b\u3002"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"System prompt: {system_prompt}")
    print(f"User input: {user_input}")
    print(f"Rendered chat template: {prompt[:100]}...")

    input_ids = tokenizer.encode(prompt)
    print(f"Input token count: {len(input_ids)}")

    print("\n=== Use standard decoding ===")
    q = queue.Queue()
    callback = create_callback(q)

    start_time = time.time()

    def run_standard_generation():
        try:
            generate_text_stream(
                input_ids,
                callback,
                max_length=60,
                temperature=0.8,
                top_p=0.9,
                top_k=50
            )
        except Exception as e:
            print(f"Standard decoding failed: {e}")
        finally:
            q.put(None)

    thread = threading.Thread(target=run_standard_generation)
    thread.start()

    # accumulate in the main thread token and decode
    output_ids = []
    last_output = ""
    print("Output:", end="", flush=True)

    try:
        while True:
            token = q.get(timeout=60)
            if token is None:
                break

            output_ids.append(token)
            try:
                new_text = tokenizer.decode(output_ids)
                diff = new_text[len(last_output):]
                last_output = new_text

                if diff:
                    print(diff, end="", flush=True)
            except Exception as e:
                print(f"Decoding error: {e}", end="", flush=True)
    except queue.Empty:
        print("\nTimed out while waiting, Forcing generation to stop")

    print()
    standard_time = time.time() - start_time
    standard_tokens = len(output_ids)
    print(f"Standard decoding generated {standard_tokens} items token, Elapsed time {standard_time:.2f} seconds")
    if standard_tokens > 0:
        print(f"Standard decoding speed: {standard_tokens / standard_time:.2f} tokens/s")
    else:
        print("Standard decoding did not generate any token")

    print("\n=== Use speculative decoding ===")
    q = queue.Queue()
    callback = create_callback(q)

    start_time = time.time()

    def run_speculative_generation():
        try:
            generate_text_stream_speculative(
                input_ids,
                callback,
                max_length=60,
                temperature=0.8,
                top_p=0.9,
                top_k=50
            )
        except Exception as e:
            print(f"Speculative decoding failed: {e}")
        finally:
            q.put(None)

    thread = threading.Thread(target=run_speculative_generation)
    thread.start()

    # accumulate in the main thread token and decode
    output_ids = []
    last_output = ""
    print("Output:", end="", flush=True)

    try:
        while True:
            token = q.get(timeout=60)
            if token is None:
                break

            output_ids.append(token)
            try:
                new_text = tokenizer.decode(output_ids)
                diff = new_text[len(last_output):]
                last_output = new_text

                if diff:
                    print(diff, end="", flush=True)
            except Exception as e:
                print(f"Decoding error: {e}", end="", flush=True)
    except queue.Empty:
        print("\nTimed out while waiting, Forcing generation to stop")

    print()
    spec_time = time.time() - start_time
    spec_tokens = len(output_ids)
    print(f"Speculative decoding generated {spec_tokens} items token, Elapsed time {spec_time:.2f} seconds")
    if spec_tokens > 0:
        print(f"Speculative decoding speed: {spec_tokens / spec_time:.2f} tokens/s")
    else:
        print("Speculative decoding did not generate any token")

    print(f"\n=== Performance comparison ===")
    if standard_tokens > 0 and spec_tokens > 0:
        speedup = standard_time / spec_time
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("Cannot compute the speedup: Standard or speculative decoding generated too few token")

    print("\nData collection complete! You can inspect it with the following command logits Distribution visualization:")
    print("python tools/analysis/visualize_logits.py")

if __name__ == "__main__":
    main()
