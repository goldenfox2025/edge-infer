"""Load supported local text checkpoints for the frontend and validation tools."""

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


MODEL_TYPES = ("llama", "qwen", "qwen_bf16", "qwen_awq", "qwen3_bf16", "qwen3_awq")


def _checkpoint_file(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Checkpoint shard names must be relative paths")
    path = (root / name).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Checkpoint shards must stay inside the model directory") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_model(model_path, model_type):
    """Read config and safetensors weights; never fetch remote model files."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    root = Path(model_path).resolve()
    with (root / "config.json").open(encoding="utf-8") as source:
        config = json.load(source)
    if not isinstance(config, dict):
        raise ValueError("Model configuration must be a JSON object")

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as source:
            index = json.load(source)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("The safetensors index must contain a nonempty weight_map")
        shards = {}
        for name, shard in weight_map.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Indexed tensor names must be nonempty strings")
            path = _checkpoint_file(root, shard)
            shards.setdefault(path, []).append(name)
    else:
        shards = {_checkpoint_file(root, "model.safetensors"): None}

    awq = model_type.endswith("_awq")
    bf16 = awq or model_type.endswith("_bf16")
    verbose = os.environ.get("EDGE_INFER_VERBOSE_WEIGHTS") == "1"
    weights = {}
    for path, names in shards.items():
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys() if names is None else names:
                tensor = source.get_tensor(name)
                if awq and name.endswith((".qweight", ".qzeros", ".scales")):
                    weights[name] = tensor
                else:
                    weights[name] = tensor.to(torch.bfloat16 if bf16 else torch.float32)
                if verbose:
                    print(f"Loaded {name}: {tuple(weights[name].shape)}, {weights[name].dtype}")

    if not weights:
        raise ValueError("The checkpoint contains no tensors")
    if config.get("tie_word_embeddings"):
        embedding = weights.get("model.embed_tokens.weight")
        if embedding is not None:
            weights.setdefault("lm_head.weight", embedding)
    if awq:
        quantization = config.get("quantization_config", {})
        if not isinstance(quantization, dict):
            raise ValueError("quantization_config must be a JSON object")
        config["group_size"] = quantization.get("group_size", config.get("group_size", 128))
    return config, weights, model_type
