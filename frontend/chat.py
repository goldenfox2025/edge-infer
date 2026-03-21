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

# 检查是否安装了可选依赖
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

# 检查是否安装了tokenizers库
TOKENIZERS_AVAILABLE = False
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    pass

# -------------------------------
# 模型加载相关函数
# -------------------------------

def load_llama_model(model_path: str):
    """加载 Llama 模型及配置"""
    model_path = Path(model_path)
    weights = {}

    # 加载模型权重
    with safe_open(model_path / "model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # 如果是 bfloat16，则转换为 float32
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            weights[key] = tensor
            print(f"Loaded tensor {key} with shape {weights[key].shape}")

    # 加载配置文件
    with open(model_path / "config.json", 'r') as f:
        config = json.load(f)
        # 若缺少 embedding_table，则采用 lm_head 权重
        if "model.embed_tokens.weight" not in weights:
            config["tie_word_embeddings"] = True
            weights["model.embed_tokens.weight"] = weights["lm_head.weight"]
        print("Config loaded:", config)

    return config, weights, "llama"

def load_qwen_model(model_path: str, keep_bf16=True, is_awq=False):
    """加载 Qwen 模型及配置，可选保持 BF16 精度或加载 AWQ 量化模型"""
    model_path = Path(model_path)
    weights = {}

    # 加载模型权重
    with safe_open(model_path / "model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)

            # 如果是AWQ量化模型，需要区分量化权重和非量化权重
            if is_awq:
                # 检查是否是量化权重（qweight、scales、qzeros）
                if any(suffix in key for suffix in [".qweight", ".scales", ".qzeros"]):
                    # 量化权重保持原始格式
                    weights[key] = tensor
                    print(f"Loaded AWQ quantized tensor {key} with shape {tensor.shape} and dtype {tensor.dtype}")
                else:
                    # 非量化权重根据keep_bf16参数决定是否转换
                    if tensor.dtype == torch.bfloat16 and keep_bf16:
                        weights[key] = tensor
                        print(f"Loaded AWQ non-quantized bf16 tensor {key} with shape {tensor.shape}")
                    else:
                        weights[key] = tensor.to(torch.float32)
                        print(f"Loaded AWQ non-quantized fp32 tensor {key} with shape {weights[key].shape}")
            # 非AWQ模型处理
            elif tensor.dtype == torch.bfloat16 and keep_bf16:
                weights[key] = tensor
                print(f"Loaded bf16 tensor {key} with shape {tensor.shape}")
            else:
                weights[key] = tensor.to(torch.float32)
                print(f"Loaded tensor {key} with shape {weights[key].shape}")

    # 加载配置文件
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

        # 如果是AWQ模型，添加量化相关配置
        if is_awq:
            # 检查是否有AWQ相关配置
            if "quantization_config" in config:
                quant_config = config["quantization_config"]
                if "group_size" in quant_config:
                    config["group_size"] = quant_config["group_size"]
                    print(f"Using group_size from config: {config['group_size']}")
                else:
                    config["group_size"] = 128  # 默认值
                    print(f"Using default group_size: {config['group_size']}")
            else:
                config["group_size"] = 128  # 默认值
                print(f"Using default group_size: {config['group_size']}")

    # 根据模型类型返回不同的标识符
    if is_awq:
        model_type = "qwen_awq"
    else:
        model_type = "qwen_bf16" if keep_bf16 else "qwen"

    return config, weights, model_type

def load_qwen3_model(model_path: str, keep_bf16=True, is_awq=False):
    """加载 Qwen3 模型及配置，可选保持 BF16 精度或加载 AWQ 量化模型"""
    model_path = Path(model_path)
    weights = {}
    verbose_weights = os.environ.get("LLM_INFER_VERBOSE_WEIGHTS") == "1"

    # 首先检查和加载索引文件
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        print(f"找到模型索引文件: {index_path}")
        with open(index_path, 'r') as f:
            index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            
        # 按文件分组权重，以便一次性加载每个文件的所有权重
        weights_by_file = {}
        for key, file_name in weight_map.items():
            if file_name not in weights_by_file:
                weights_by_file[file_name] = []
            weights_by_file[file_name].append(key)
            
        # 加载每个文件中的权重
        for file_name, keys in weights_by_file.items():
            file_path = model_path / file_name
            if verbose_weights:
                print(f"从文件加载权重: {file_path}")
            
            with safe_open(file_path, framework="pt") as f:
                for key in keys:
                    tensor = f.get_tensor(key)
                    
                    # 如果是AWQ量化模型，需要区分量化权重和非量化权重
                    if is_awq:
                        # 检查是否是量化权重（qweight、scales、qzeros）
                        if any(suffix in key for suffix in [".qweight", ".scales", ".qzeros"]):
                            # 量化权重保持原始格式
                            weights[key] = tensor
                            if verbose_weights:
                                print(f"加载AWQ量化张量 {key}，形状 {tensor.shape}，数据类型 {tensor.dtype}")
                        else:
                            # 非量化权重根据keep_bf16参数决定是否转换
                            if tensor.dtype == torch.bfloat16 and keep_bf16:
                                weights[key] = tensor
                                if verbose_weights:
                                    print(f"加载AWQ非量化bf16张量 {key}，形状 {tensor.shape}")
                            else:
                                weights[key] = tensor.to(torch.float32)
                                if verbose_weights:
                                    print(f"加载AWQ非量化fp32张量 {key}，形状 {weights[key].shape}")
                    # 非AWQ模型处理
                    elif tensor.dtype == torch.bfloat16 and keep_bf16:
                        weights[key] = tensor
                        if verbose_weights:
                            print(f"加载bf16张量 {key}，形状 {tensor.shape}")
                    else:
                        weights[key] = tensor.to(torch.float32)
                        if verbose_weights:
                            print(f"加载张量 {key}，形状 {weights[key].shape}")
    else:
        # 尝试从单个文件加载，保持向后兼容性
        safetensors_path = model_path / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"未找到模型权重文件: 既没有索引文件 {index_path}，也没有单一权重文件 {safetensors_path}")
            
        if verbose_weights:
            print(f"未找到索引文件，尝试从单一文件加载: {safetensors_path}")
        with safe_open(safetensors_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # 如果是AWQ量化模型，需要区分量化权重和非量化权重
                if is_awq:
                    # 检查是否是量化权重（qweight、scales、qzeros）
                    if any(suffix in key for suffix in [".qweight", ".scales", ".qzeros"]):
                        # 量化权重保持原始格式
                        weights[key] = tensor
                        if verbose_weights:
                            print(f"加载AWQ量化张量 {key}，形状 {tensor.shape}，数据类型 {tensor.dtype}")
                    else:
                        # 非量化权重根据keep_bf16参数决定是否转换
                        if tensor.dtype == torch.bfloat16 and keep_bf16:
                            weights[key] = tensor
                            if verbose_weights:
                                print(f"加载AWQ非量化bf16张量 {key}，形状 {tensor.shape}")
                        else:
                            weights[key] = tensor.to(torch.float32)
                            if verbose_weights:
                                print(f"加载AWQ非量化fp32张量 {key}，形状 {weights[key].shape}")
                # 非AWQ模型处理
                elif tensor.dtype == torch.bfloat16 and keep_bf16:
                    weights[key] = tensor
                    if verbose_weights:
                        print(f"加载bf16张量 {key}，形状 {tensor.shape}")
                else:
                    weights[key] = tensor.to(torch.float32)
                    if verbose_weights:
                        print(f"加载张量 {key}，形状 {weights[key].shape}")

    # 加载配置文件
    config_path = model_path / "config.json"
    if verbose_weights:
        print(f"正在读取配置文件: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_str = f.read()
            if verbose_weights:
                print(f"原始配置内容的前100个字符: {config_str[:100]}...")
            config = json.loads(config_str)
            if verbose_weights:
                print("成功加载配置文件")
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        raise

    # 打印原始配置内容
    if verbose_weights:
        print("原始配置内容:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    # 创建C++接口所需的配置字典
    cpp_config = {}
    
    # 直接复制所有原始配置（确保所有键都存在）
    for key, value in config.items():
        cpp_config[key] = value
    
    # 特殊映射键名，同时保留原键名和映射后的键名
    key_mapping = {
        "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads"
    }
    
    # 添加映射后的键名
    for orig_key, new_key in key_mapping.items():
        if orig_key in config:
            cpp_config[new_key] = config[orig_key]
            if verbose_weights:
                print(f"添加映射键: {orig_key} -> {new_key}: {config[orig_key]}")

    
    # 添加量化相关配置
    if is_awq:
        cpp_config["quant_type"] = 1
        
        # 检查是否有AWQ相关配置
        if "quantization_config" in config:
            quant_config = config["quantization_config"]
            if "group_size" in quant_config:
                cpp_config["group_size"] = quant_config["group_size"]
                if verbose_weights:
                    print(f"使用配置中的group_size: {cpp_config['group_size']}")
            else:
                cpp_config["group_size"] = 128  # 默认值
                if verbose_weights:
                    print("使用默认group_size: 128")
        else:
            cpp_config["group_size"] = 128  # 默认值
            if verbose_weights:
                print("使用默认group_size: 128")
    else:
        cpp_config["quant_type"] = 0
    
    # 打印最终配置
    if verbose_weights:
        print("最终配置:")
        for key, value in cpp_config.items():
            print(f"  {key}: {value}")

    # 根据模型类型返回不同的标识符
    if is_awq:
        model_type = "qwen3_awq"
    else:
        model_type = "qwen3_bf16" if keep_bf16 else "qwen3"

    return cpp_config, weights, model_type

def load_tokenizer(model_path: str, model_type: str):
    """根据模型类型加载对应的 tokenizer"""
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
# 回调函数：仅返回 token_id（轻量化回调）
# -------------------------------
def create_callback(q: queue.Queue):
    """
    创建用于处理生成 token 的回调函数。
    该回调函数只将 token_id 放入队列，不进行解码和统计。
    """
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
# 终端聊天实现
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='LLaMA/Qwen/Qwen3 模型聊天')
    parser.add_argument('--model_path', type=str, default="./models/Qwen3-1.7B-AWQ", help='模型路径')
    parser.add_argument('--model_type', type=str, default="qwen3_awq", 
                       choices=['llama', 'qwen', 'qwen_bf16', 'qwen_awq', 'qwen3_bf16', 'qwen3_awq'], 
                       help='模型类型')
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'], help='运行设备 (cuda 或 cpu)') # qwen不支持cpu 会强制使用cuda
    parser.add_argument('--system_prompt', type=str, default="You are a helpful AI assistant.", help='系统提示词')
    parser.add_argument('--max_length', type=int, default=424, help='生成文本的最大长度')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--top_p', type=float, default=1, help='top-p 采样阈值')
    parser.add_argument('--top_k', type=int, default=20, help='top-k 采样阈值')
    args = parser.parse_args()

    # 加载模型和 tokenizer
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

    # 计算并打印模型大小
    total_params = sum(t.numel() for t in weights.values())
    total_bytes = sum(t.element_size() * t.numel() for t in weights.values())
    print("\nModel size: {} parameters, {:.2f} MB".format(total_params, total_bytes / (1024 * 1024)))

    # 打印模型配置信息
    print("\nModel Configuration:")
    print(f"Model Type: {model_type}")
    if model_type.startswith("qwen"):
        precision = "BF16" if "bf16" in model_type else "FP32"
        if "awq" in model_type:
            precision += " (AWQ Quantized)"
        print(f"Precision: {precision}")
    
    # 显示模型配置
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

    # 导入 C++ 模型桥接接口
    sys.path.append("./build")
    from model_bridge import init_model, generate_text_stream, set_default_device, get_default_device

    # 导入设备配置模块
    sys.path.append("./interface")
    import device_config

    # 设置默认设备
    print(f"\n设置默认设备为: {args.device}")
    if not set_default_device(args.device):
        print(f"设置设备 {args.device} 失败，使用默认设备", file=sys.stderr)

    # 打印当前设备
    current_device = get_default_device()
    print(f"当前使用设备: {current_device}")

    # 初始化模型
    if not init_model(config, weights, model_type):
        print("Model initialization failed.", file=sys.stderr)
        exit(1)
    print("\nModel initialized successfully.\n")

    # 记录是否是首次对话
    first_chat = True

    print("聊天已启动。输入'quit'或'exit'退出。\n")

    while True:
        user_message = input("User: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break
        if not user_message:
            continue

        # 根据模型类型构造对话文本（仅传输当前消息，历史记录由本地维护 kvcache）
        if model_type.startswith("qwen"):
            if first_chat:
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": user_message}
                ]
                first_chat = False
            else:
                messages = [{"role": "user", "content": user_message}]

            # 使用 apply_chat_template 应用聊天模板
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

        # 生成回复：使用 minimal callback 只返回 token_id
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
            q.put(None)  # 生成结束标记

        thread = threading.Thread(target=run_generation)
        thread.start()

        print("Assistant: ", end="", flush=True)

        # 在主线程中累积 token_id 并进行解码与统计
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
            # 计算每个token的生成时间（仅用于调试）
            if last_token_time is not None:
                token_time = current_time - last_token_time
                # 如果需要打印每个token的生成时间，可以取消下面的注释
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
