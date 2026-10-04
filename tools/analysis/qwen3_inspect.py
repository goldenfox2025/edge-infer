#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Inspect Qwen3-1.7B with Transformers, print tensor shapes, and run inference."""

import os
import argparse
from pathlib import Path
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Tuple, Optional, Union

def print_header(title: str) -> None:
    """Print a heading with a separator."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_section(title: str) -> None:
    """Print a section heading with a separator."""
    print("\n" + "-" * 80)
    print(f" {title} ".center(80, "-"))
    print("-" * 80)

def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    """Print detailed tensor information."""
    print(f"{name}:")
    print(f"  - Shape: {tensor.shape}")
    print(f"  - Dtype: {tensor.dtype}")
    print(f"  - Device: {tensor.device}")
    print(f"  - Min/Max: {tensor.min().item():.6f} / {tensor.max().item():.6f}")
    print(f"  - Mean/Std: {tensor.mean().item():.6f} / {tensor.std().item():.6f}")

def inspect_model_structure(model: AutoModelForCausalLM) -> None:
    """Inspect the model structure and print tensor shapes for each layer."""
    print_header("Model structure overview")

    print(f"Model type: {model.__class__.__name__}")
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Model device: {next(model.parameters()).device}")

    config = model.config
    print_section("Model configuration")
    print(f"Hidden size (hidden_size): {config.hidden_size}")
    print(f"Layer count (num_hidden_layers): {config.num_hidden_layers}")
    print(f"Attention head count (num_attention_heads): {config.num_attention_heads}")
    print(f"KV head count (num_key_value_heads): {config.num_key_value_heads}")
    print(f"Intermediate size (intermediate_size): {config.intermediate_size}")
    print(f"Head dimension (head_dim): {config.head_dim}")
    print(f"Maximum position embeddings (max_position_embeddings): {config.max_position_embeddings}")
    print(f"Vocabulary size (vocab_size): {config.vocab_size}")
    print(f"RMS Normalization epsilon (rms_norm_eps): {config.rms_norm_eps}")
    print(f"RoPE theta: {config.rope_theta}")

    print_section("Embedding layer")
    embed_tokens = model.model.embed_tokens
    print_tensor_info("embed_tokens.weight", embed_tokens.weight)

    print_section(f"Transformer layer details ({len(model.model.layers)} layers total)")
    layer = model.model.layers[0]

    print("\nAttention module:")
    print(f"  - q_proj.weight: {layer.self_attn.q_proj.weight.shape}")
    print(f"  - k_proj.weight: {layer.self_attn.k_proj.weight.shape}")
    print(f"  - v_proj.weight: {layer.self_attn.v_proj.weight.shape}")
    print(f"  - o_proj.weight: {layer.self_attn.o_proj.weight.shape}")

    print("\nMLP module:")
    print(f"  - gate_proj.weight: {layer.mlp.gate_proj.weight.shape}")
    print(f"  - up_proj.weight: {layer.mlp.up_proj.weight.shape}")
    print(f"  - down_proj.weight: {layer.mlp.down_proj.weight.shape}")

    print("\nNormalization layers:")
    print_tensor_info("input_layernorm.weight", layer.input_layernorm.weight)
    print_tensor_info("post_attention_layernorm.weight", layer.post_attention_layernorm.weight)

    print_section("Final layer normalization and LM head")
    print_tensor_info("norm.weight", model.model.norm.weight)
    print(f"lm_head.weight: {model.lm_head.weight.shape}")

    print_section("Weight-sharing check")
    is_shared = torch.equal(model.model.embed_tokens.weight, model.lm_head.weight)
    print(f"Embedding layer and LM head share weights: {is_shared}")

    print_section("Names and shapes of all layers")
    for name, param in model.named_parameters():
        if 'layers' not in name or 'layers.0' in name:
            print(f"{name}: {param.shape}")

def run_inference(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
    """Run inference and return generated text."""
    print_header("Run inference")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"Input text: {text}")

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(f"Input shape: {inputs.input_ids.shape}")

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generation_time = time.time() - start_time

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nGeneration time: {generation_time:.2f} seconds")
    print(f"Generated token Count: {outputs.shape[1] - inputs.input_ids.shape[1]}")
    print(f"Generated per second token count: {(outputs.shape[1] - inputs.input_ids.shape[1]) / generation_time:.2f}")

    return generated_text

def inspect_attention_patterns(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> None:
    """Inspect attention patterns."""
    print_header("Attention pattern inspection")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )

        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attentions = outputs.attentions

            print(f"Attention weight count: {len(attentions)}")
            for i, attn in enumerate(attentions):
                print(f"layer {i+1} Attention weight shape: {attn.shape}")

        else:
            print("The model did not return attention weights. The architecture may not support attention output, or it may be disabled.")
    except Exception as e:
        print(f"Failed to inspect attention patterns: {e}")

def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        default=os.environ.get("MODEL_PATH", str(repo_root / "models" / "Qwen3-1.7B")),
        help="Local model directory or Hugging Face model ID (default: MODEL_PATH or models/Qwen3-1.7B).",
    )
    args = parser.parse_args()
    try:

        model_path = args.model_path

        print_header("Load the model and tokenizer")
        print(f"Model path: {model_path}")

        print("Load the tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load the tokenizer locally: {e}")
            print("Trying from Hugging Face load the tokenizer directly...")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

        print("Load the model...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
        except ValueError as e:
            print(f"The standard loading method failed: {e}")
            print("Trying to use trust_remote_code=True and revision='main' parameters...")

            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-1.7B",
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                revision="main"
            )

        inspect_model_structure(model)

        prompt = "\u8bf7\u7b80\u5355\u4ecb\u7ecd\u4e00\u4e0b\u91cf\u5b50\u8ba1\u7b97"
        generated_text = run_inference(model, tokenizer, prompt)

        print_section("Generated text")
        print(generated_text)

        inspect_attention_patterns(model, tokenizer, "\u4eba\u5de5\u667a\u80fd\u662f\u4ec0\u4e48\uff1f")

    except Exception as e:
        import traceback
        print_header("Execution failed")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nDetailed error information:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
