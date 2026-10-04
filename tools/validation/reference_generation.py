"""Run a manual Transformers reference generation and report elapsed time."""

import argparse
import os
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        default=os.environ.get("MODEL_PATH", str(repo_root / "models" / "Qwen2.5-1.5B")),
        help="Local model directory or Hugging Face model ID (default: MODEL_PATH or models/Qwen2.5-1.5B).",
    )
    model_name = parser.parse_args().model_path

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "\u8bb2\u4e2a\u6545\u4e8b"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    import time

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    start_time = time.time()
    print("Running generation test (streaming)...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=200,
        temperature=1,
        top_k=40,
        streamer=streamer,
    )

    elapsed_time = time.time() - start_time

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    speed = len(generated_ids[0]) / elapsed_time
    print(f"Token count: {len(generated_ids[0])}")
    print(f"\nSpeed: {speed:.2f} tokens/sec")
    print(f"Generation time: {elapsed_time:.2f} seconds")

    # print("\nModel Structure:")
    # print(model)

if __name__ == "__main__":
    main()
