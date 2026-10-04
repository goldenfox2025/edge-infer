#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Quantize Qwen3 with AutoAWQ and inspect GEMV and GEMM tensor layouts."""

import torch
import os
import shutil
from transformers import AutoTokenizer

try:
    from awq import AutoAWQForCausalLM

    from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
except ImportError:
    print("Error: Please install autoawq (pip install autoawq)")
    print("You may also need: pip install torch transformers accelerate safetensors")
    exit()

def main():

    model_path = "./models/Qwen3-4B"
    quant_path = "./models/Qwen3-4B-AWQ"

    if os.path.exists(quant_path):
        print(f"Found an old quantization directory: {quant_path}")
        try:
            shutil.rmtree(quant_path)
            print("Removed the old quantization directory.")
        except OSError as e:
            print(f"Error: Cannot remove the old directory {quant_path}: {e}")
            print("Remove this directory manually and try again.")
            return

    print(f"Loading the tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load the tokenizer: {e}")
        return

    print("Starting a new quantization run...")
    os.makedirs(quant_path, exist_ok=True)

    print(f"Loading the original model: {model_path}")
    try:
        model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    except Exception as e:
         print(f"Failed to load the original model: {e}")
         return

    print("Preparing calibration data")
    print("Using local calibration data")
    samples = [
        "\u4eba\u5de5\u667a\u80fd\u662f\u4e00\u79cd\u80fd\u591f\u6a21\u62df\u4eba\u7c7b\u667a\u80fd\u7684\u8ba1\u7b97\u673a\u7cfb\u7edf\uff0c\u5b83\u53ef\u4ee5\u5b66\u4e60\u3001\u63a8\u7406\u548c\u81ea\u6211\u5b8c\u5584\u3002\u4eba\u5de5\u667a\u80fd\u7684\u5e94\u7528\u8303\u56f4\u975e\u5e38\u5e7f\u6cdb\uff0c\u5305\u62ec\u81ea\u7136\u8bed\u8a00\u5904\u7406\u3001\u8ba1\u7b97\u673a\u89c6\u89c9\u3001\u673a\u5668\u4eba\u6280\u672f\u7b49\u591a\u4e2a\u9886\u57df\u3002\u968f\u7740\u6280\u672f\u7684\u4e0d\u65ad\u53d1\u5c55\uff0c\u4eba\u5de5\u667a\u80fd\u6b63\u5728\u6539\u53d8\u6211\u4eec\u7684\u751f\u6d3b\u548c\u5de5\u4f5c\u65b9\u5f0f\u3002",
        "\u91cf\u5b50\u8ba1\u7b97\u5229\u7528\u91cf\u5b50\u529b\u5b66\u539f\u7406\uff0c\u5982\u53e0\u52a0\u548c\u7ea0\u7f20\uff0c\u6765\u5904\u7406\u4fe1\u606f\uff0c\u6709\u671b\u89e3\u51b3\u4f20\u7edf\u8ba1\u7b97\u673a\u96be\u4ee5\u89e3\u51b3\u7684\u95ee\u9898\u3002\u91cf\u5b50\u8ba1\u7b97\u673a\u53ef\u4ee5\u540c\u65f6\u5904\u7406\u591a\u79cd\u72b6\u6001\uff0c\u8fd9\u4f7f\u5f97\u5b83\u4eec\u5728\u67d0\u4e9b\u7279\u5b9a\u4efb\u52a1\u4e0a\u6bd4\u4f20\u7edf\u8ba1\u7b97\u673a\u5feb\u5f97\u591a\u3002\u79d1\u5b66\u5bb6\u4eec\u6b63\u5728\u52aa\u529b\u514b\u670d\u91cf\u5b50\u8ba1\u7b97\u9762\u4e34\u7684\u6280\u672f\u6311\u6218\uff0c\u5982\u91cf\u5b50\u9000\u76f8\u5e72\u548c\u9519\u8bef\u6821\u6b63\u7b49\u95ee\u9898\u3002",
        "\u673a\u5668\u5b66\u4e60\u662f\u4eba\u5de5\u667a\u80fd\u7684\u4e00\u4e2a\u5b50\u9886\u57df\uff0c\u5b83\u4f7f\u8ba1\u7b97\u673a\u7cfb\u7edf\u80fd\u591f\u4ece\u6570\u636e\u4e2d\u5b66\u4e60\uff0c\u800c\u65e0\u9700\u660e\u786e\u7f16\u7a0b\u3002\u673a\u5668\u5b66\u4e60\u7b97\u6cd5\u53ef\u4ee5\u5206\u4e3a\u76d1\u7763\u5b66\u4e60\u3001\u65e0\u76d1\u7763\u5b66\u4e60\u548c\u5f3a\u5316\u5b66\u4e60\u7b49\u591a\u79cd\u7c7b\u578b\u3002\u8fd9\u4e9b\u7b97\u6cd5\u5df2\u7ecf\u5728\u56fe\u50cf\u8bc6\u522b\u3001\u8bed\u97f3\u8bc6\u522b\u3001\u63a8\u8350\u7cfb\u7edf\u7b49\u591a\u4e2a\u9886\u57df\u53d6\u5f97\u4e86\u663e\u8457\u6210\u529f\u3002",
        "\u6df1\u5ea6\u5b66\u4e60\u662f\u673a\u5668\u5b66\u4e60\u7684\u4e00\u79cd\u65b9\u6cd5\uff0c\u5b83\u4f7f\u7528\u591a\u5c42\u795e\u7ecf\u7f51\u7edc\u6765\u63d0\u53d6\u6570\u636e\u4e2d\u7684\u9ad8\u7ea7\u7279\u5f81\u3002\u6df1\u5ea6\u5b66\u4e60\u6a21\u578b\uff0c\u5982\u5377\u79ef\u795e\u7ecf\u7f51\u7edc\u548c\u5faa\u73af\u795e\u7ecf\u7f51\u7edc\uff0c\u5df2\u7ecf\u5728\u8ba1\u7b97\u673a\u89c6\u89c9\u548c\u81ea\u7136\u8bed\u8a00\u5904\u7406\u7b49\u9886\u57df\u53d6\u5f97\u4e86\u7a81\u7834\u6027\u8fdb\u5c55\u3002\u8fd9\u4e9b\u6a21\u578b\u9700\u8981\u5927\u91cf\u7684\u6570\u636e\u548c\u8ba1\u7b97\u8d44\u6e90\u6765\u8bad\u7ec3\uff0c\u4f46\u5b83\u4eec\u7684\u6027\u80fd\u901a\u5e38\u8d85\u8fc7\u4f20\u7edf\u7684\u673a\u5668\u5b66\u4e60\u65b9\u6cd5\u3002",
        "\u81ea\u7136\u8bed\u8a00\u5904\u7406\u662f\u4eba\u5de5\u667a\u80fd\u7684\u4e00\u4e2a\u5206\u652f\uff0c\u4e13\u6ce8\u4e8e\u4f7f\u8ba1\u7b97\u673a\u7406\u89e3\u3001\u89e3\u91ca\u548c\u751f\u6210\u4eba\u7c7b\u8bed\u8a00\u3002\u81ea\u7136\u8bed\u8a00\u5904\u7406\u6280\u672f\u5305\u62ec\u6587\u672c\u5206\u7c7b\u3001\u60c5\u611f\u5206\u6790\u3001\u673a\u5668\u7ffb\u8bd1\u548c\u95ee\u7b54\u7cfb\u7edf\u7b49\u3002\u8fd1\u5e74\u6765\uff0c\u57fa\u4e8eTransformer\u67b6\u6784\u7684\u5927\u578b\u8bed\u8a00\u6a21\u578b\uff0c\u5982GPT\u548cBERT\uff0c\u6781\u5927\u5730\u63d0\u9ad8\u4e86\u81ea\u7136\u8bed\u8a00\u5904\u7406\u7684\u80fd\u529b\u3002",
        "\u8ba1\u7b97\u673a\u89c6\u89c9\u662f\u4eba\u5de5\u667a\u80fd\u7684\u4e00\u4e2a\u9886\u57df\uff0c\u5b83\u4f7f\u8ba1\u7b97\u673a\u80fd\u591f\u4ece\u56fe\u50cf\u6216\u89c6\u9891\u4e2d\u83b7\u53d6\u4fe1\u606f\u5e76\u7406\u89e3\u89c6\u89c9\u4e16\u754c\u3002\u8ba1\u7b97\u673a\u89c6\u89c9\u6280\u672f\u5305\u62ec\u56fe\u50cf\u5206\u7c7b\u3001\u76ee\u6807\u68c0\u6d4b\u3001\u56fe\u50cf\u5206\u5272\u548c\u4eba\u8138\u8bc6\u522b\u7b49\u3002\u8fd9\u4e9b\u6280\u672f\u5df2\u7ecf\u5728\u81ea\u52a8\u9a7e\u9a76\u3001\u533b\u7597\u8bca\u65ad\u548c\u5b89\u5168\u76d1\u63a7\u7b49\u9886\u57df\u5f97\u5230\u4e86\u5e7f\u6cdb\u5e94\u7528\u3002",
        "\u5f3a\u5316\u5b66\u4e60\u662f\u4e00\u79cd\u673a\u5668\u5b66\u4e60\u65b9\u6cd5\uff0c\u5b83\u901a\u8fc7\u4e0e\u73af\u5883\u4ea4\u4e92\u5e76\u4ece\u53cd\u9988\u4e2d\u5b66\u4e60\u6765\u4f18\u5316\u51b3\u7b56\u3002\u5728\u5f3a\u5316\u5b66\u4e60\u4e2d\uff0c\u667a\u80fd\u4f53\u901a\u8fc7\u5c1d\u8bd5\u4e0d\u540c\u7684\u884c\u52a8\u5e76\u89c2\u5bdf\u7ed3\u679c\u6765\u5b66\u4e60\u6700\u4f73\u7b56\u7565\u3002\u8fd9\u79cd\u65b9\u6cd5\u5df2\u7ecf\u5728\u6e38\u620f\u3001\u673a\u5668\u4eba\u63a7\u5236\u548c\u8d44\u6e90\u7ba1\u7406\u7b49\u9886\u57df\u53d6\u5f97\u4e86\u6210\u529f\u3002",
        "\u795e\u7ecf\u7f51\u7edc\u662f\u4e00\u79cd\u53d7\u4eba\u8111\u542f\u53d1\u7684\u8ba1\u7b97\u6a21\u578b\uff0c\u7531\u76f8\u4e92\u8fde\u63a5\u7684\u8282\u70b9\uff08\u795e\u7ecf\u5143\uff09\u7ec4\u6210\uff0c\u7528\u4e8e\u6a21\u5f0f\u8bc6\u522b\u548c\u51b3\u7b56\u3002\u795e\u7ecf\u7f51\u7edc\u53ef\u4ee5\u5b66\u4e60\u590d\u6742\u7684\u975e\u7ebf\u6027\u5173\u7cfb\uff0c\u8fd9\u4f7f\u5b83\u4eec\u5728\u5904\u7406\u56fe\u50cf\u3001\u8bed\u97f3\u548c\u6587\u672c\u7b49\u6570\u636e\u65f6\u975e\u5e38\u6709\u6548\u3002\u6df1\u5ea6\u795e\u7ecf\u7f51\u7edc\u5305\u542b\u591a\u4e2a\u9690\u85cf\u5c42\uff0c\u80fd\u591f\u5b66\u4e60\u6570\u636e\u7684\u5c42\u6b21\u8868\u793a\u3002"
    ] * 16
    print(f"Prepared {len(samples)} calibration samples")

    print("Configuring quantization")
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMV"
    }
    # ---------------------

    print("Quantizing the model...")
    try:
        model.quantize(tokenizer, quant_config=quant_config, calib_data=samples)
    except Exception as e:
        print(f"Quantization failed: {e}")
        raise

    print(f"Saving the quantized model to: {quant_path}")
    try:
        model.save_quantized(quant_path)
        tokenizer.save_pretrained(quant_path)
        print(f"Quantized model save attempted at: {os.path.abspath(quant_path)}")
    except Exception as e:
        print(f"Failed to save the model: {e}")
        raise

    print("\nLoading the newly quantized model to inspect tensor shapes...")
    try:
        model_quant = AutoAWQForCausalLM.from_quantized(quant_path, device_map="auto", trust_remote_code=True)
        print("Quantized model loaded successfully! Inspecting layers now...")

        print("\n--- Starting inspection Quantized Layers ( find WQLinear_GEMV and WQLinear_GEMM) ---")
        found_gemv = 0
        found_gemm = 0
        max_layers_to_print_each = 5

        for name, module in model_quant.named_modules():

            if isinstance(module, WQLinear_GEMV):
                found_gemv += 1
                if found_gemv <= max_layers_to_print_each:
                    print(f"\nLayer (Type: WQLinear_GEMV): {name}")
                    print(f"  qweight shape: {module.qweight.shape} (dtype: {module.qweight.dtype})")
                    print(f"  scales shape: {module.scales.shape} (dtype: {module.scales.dtype})")
                    print(f"  qzeros shape: {module.qzeros.shape} (dtype: {module.qzeros.dtype})")
                    if hasattr(module, 'bias') and module.bias is not None:
                        print(f"  bias shape: {module.bias.shape} (dtype: {module.bias.dtype})")
                    else:
                        print(f"  bias: None")

            elif isinstance(module, WQLinear_GEMM):
                 found_gemm += 1
                 if found_gemm <= max_layers_to_print_each:
                     print(f"\nLayer (Type: WQLinear_GEMM): {name}")

                     print(f"  qweight shape: {module.qweight.shape} (dtype: {module.qweight.dtype})")
                     print(f"  scales shape: {module.scales.shape} (dtype: {module.scales.dtype})")
                     print(f"  qzeros shape: {module.qzeros.shape} (dtype: {module.qzeros.dtype})")
                     if hasattr(module, 'bias') and module.bias is not None:
                         print(f"  bias shape: {module.bias.shape} (dtype: {module.bias.dtype})")
                     else:
                         print(f"  bias: None")
                     # ------------------------------------

        if found_gemv == 0 and found_gemm == 0:
            print("\nWarning: Not found in the loaded model WQLinear_GEMV or WQLinear_GEMM layer.")
            print("Please check AutoAWQ version, whether quantization replaced the linear layers successfully.")
        else:
            print(f"\nFound a total of {found_gemv} items WQLinear_GEMV layer ( showing the first {min(found_gemv, max_layers_to_print_each)} items ).")
            print(f"Found a total of {found_gemm} items WQLinear_GEMM layer ( showing the first {min(found_gemm, max_layers_to_print_each)} items ).")

        print("--- Inspection complete ---")

    except Exception as e:
        print(f"\nFailed to load or inspect the newly quantized model: {e}")
        print("Check that quantization and saving completed successfully, and AutoAWQ is installed correctly.")
        return

    prompt = "\u8bb2\u4e2a\u5173\u4e8e\u673a\u5668\u4eba\u5b66\u4e60\u7ed8\u753b\u7684\u77ed\u6545\u4e8b\u3002"
    print(f"\nTest prompt: {prompt}")
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:

        device = "cuda:0" # Default
        if hasattr(model_quant, 'device'):
            device = model_quant.device
        elif hasattr(model_quant, 'hf_device_map'):

             device_list = list(set(model_quant.hf_device_map.values()))
             if device_list:
                 device = device_list[0]
        print(f"The model will be moved to device: {device}")
    except Exception as e:
        print(f"Failed to determine the model device: {e}, falling back to cpu")
        device = "cpu"

    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f"Input moved to device: {inputs.input_ids.device}")

    print("Generating text...")
    try:

        model_quant.to(device)

        with torch.no_grad():

            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

            outputs = model_quant.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generation succeeded! Generated text:\n{generated_text}")
    except Exception as e:
        print(f"Text generation with the newly quantized model failed: {e}")

        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()