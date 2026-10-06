#!/usr/bin/env python3
"""Export local Qwen3 Torch logits and optionally compare native greedy tokens.

All model/tokenizer loads are local-only; remote code and automatic downloads
are disabled. Native token checks use the production Model/Session API. That
API does not expose logits. An optional test-only module compares copied native
logits without changing that API.
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys


TRANSFORMERS_VERSION = "4.51.3"
BF16_ABSOLUTE_TOLERANCE = 0.015625
BF16_RELATIVE_TOLERANCE = 0.01
DEFAULT_PROMPTS = (
    "Explain why a KV cache speeds up decoding.",
    "Write one short sentence about a fox.",
)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_files(root):
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    index = root / "model.safetensors.index.json"
    files = [config]
    if index.is_file():
        mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("The local safetensors index requires a nonempty weight_map")
        names = set(mapping.values())
        files.append(index)
    else:
        names = {"model.safetensors"}
    for name in sorted(names):
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError("Checkpoint shard names must be relative paths")
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Checkpoint shards must remain inside the local directory") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
    return files


def token_list(text):
    try:
        values = [int(value.strip()) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Token IDs must be comma-separated integers") from error
    if not values or any(value < 0 or value > 0xFFFFFFFF for value in values):
        raise argparse.ArgumentTypeError("Token IDs must be nonempty uint32 values")
    return values


def compare_tokens(expected, actual):
    first = next((index for index, (left, right) in enumerate(zip(expected, actual))
                  if left != right), None)
    if first is None and len(expected) != len(actual):
        first = min(len(expected), len(actual))
    return {"passed": first is None, "first_mismatch": first,
            "expected": list(expected), "actual": list(actual)}


def cache_length(cache):
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2])


def torch_reference(model, prompt, steps, teacher_checks, eos, torch, np):
    tokens = torch.tensor([prompt], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        result = model(input_ids=tokens, use_cache=True, return_dict=True)
        if not bool(torch.isfinite(result.logits).all()):
            raise RuntimeError("Torch produced nonfinite prefill logits")
        prefill = result.logits[0].float().cpu().numpy()
        past = result.past_key_values
        growth = [cache_length(past)]
        current = result.logits[0, -1]
        generated, incremental, teacher_prefixes, teacher_tokens, teacher_logits = [], [], [], [], []
        for step in range(steps):
            if not bool(torch.isfinite(current).all()):
                raise RuntimeError(f"Torch produced nonfinite logits at greedy step {step}")
            incremental.append(current.float().cpu().numpy())
            selected = int(current.argmax().item())
            generated.append(selected)
            if selected == eos or step + 1 == steps:
                break
            result = model(input_ids=torch.tensor([[selected]], device=model.device),
                           past_key_values=past, use_cache=True, return_dict=True)
            past = result.past_key_values
            growth.append(cache_length(past))
            current = result.logits[0, -1]

        # Independent full-prefix forwards give the expected first token for
        # fresh native generation requests. They do not assume cached BF16
        # attention and full-prefix attention round identically.
        for step in range(min(teacher_checks, len(generated))):
            prefix = prompt + generated[:step]
            result = model(input_ids=torch.tensor([prefix], device=model.device),
                           use_cache=False, return_dict=True)
            logits = result.logits[0, -1]
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"Torch produced nonfinite teacher-forced logits at step {step}")
            teacher_prefixes.append(prefix)
            teacher_tokens.append(int(logits.argmax().item()))
            teacher_logits.append(logits.float().cpu().numpy())
    expected_growth = list(range(len(prompt), len(prompt) + len(growth)))
    if growth != expected_growth:
        raise RuntimeError(f"Unexpected Torch KV growth: {growth}, expected {expected_growth}")
    arrays = {"prefill_logits": prefill.astype("<f4", copy=False),
              "incremental_last_logits": np.stack(incremental).astype("<f4", copy=False),
              "teacher_forced_last_logits": np.stack(teacher_logits).astype("<f4", copy=False)}
    emitted = generated[:-1] if generated[-1] == eos else generated
    record = {"prompt_ids": prompt, "predicted_token_ids_including_eos": generated,
              "emitted_token_ids": emitted, "kv_lengths": growth,
              "teacher_forced_prefixes": teacher_prefixes,
              "teacher_forced_first_token_ids": teacher_tokens,
              "incremental_vs_full_prefix_first_token_ids": compare_tokens(
                  generated[:len(teacher_tokens)], teacher_tokens)}
    return record, arrays


def native_reference(root, module_dir, records, eos, capacity, modes):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(module_dir))
    from frontend.checkpoint import load_model
    import model_bridge

    config, weights, _ = load_model(root, "qwen3_bf16")
    original = os.environ.get("EDGE_INFER_ENABLE_QWEN3_GRAPH")
    reports = []
    try:
        for mode in modes:
            os.environ["EDGE_INFER_ENABLE_QWEN3_GRAPH"] = "1" if mode == "graph" else "0"
            model = model_bridge.Model(config, weights, "qwen3_bf16", device="cuda")
            first, second = model.new_session(capacity=capacity), model.new_session(capacity=capacity)
            report = {"mode": mode, "capacity": capacity, "prompts": [], "reset_replay": []}
            for index, record in enumerate(records):
                session = first if index % 2 == 0 else second
                actual = []
                prompt = record["prompt_ids"]
                session.generate(prompt, actual.append,
                                 max_length=len(prompt) + len(record["predicted_token_ids_including_eos"]),
                                 temperature=1.0, top_p=1.0, top_k=1)
                checks = []
                for prefix, expected in zip(record["teacher_forced_prefixes"],
                                            record["teacher_forced_first_token_ids"]):
                    predicted = []
                    session.generate(prefix, predicted.append, max_length=len(prefix) + 1,
                                     temperature=1.0, top_p=1.0, top_k=1)
                    checks.append(compare_tokens([] if expected == eos else [expected], predicted))
                report["prompts"].append({"full_greedy": compare_tokens(record["emitted_token_ids"], actual),
                                          "teacher_forced_first_tokens": checks})
            # Reuse both native sessions after interleaved requests and an
            # explicit reset; output callbacks are copied before the next call.
            for index, record in enumerate(records):
                session = first if index % 2 == 0 else second
                session.reset()
                actual = []
                session.generate(record["prompt_ids"], actual.append,
                                 max_length=len(record["prompt_ids"]) +
                                     len(record["predicted_token_ids_including_eos"]),
                                 temperature=1.0, top_p=1.0, top_k=1)
                report["reset_replay"].append(compare_tokens(record["emitted_token_ids"], actual))
            reports.append(report)
            del session, first, second, model
            gc.collect()
    finally:
        if original is None:
            os.environ.pop("EDGE_INFER_ENABLE_QWEN3_GRAPH", None)
        else:
            os.environ["EDGE_INFER_ENABLE_QWEN3_GRAPH"] = original
    return {"module": str(model_bridge.__file__), "module_sha256": file_hash(Path(model_bridge.__file__)),
            "emitted_tokens_compared_per_mode": sum(len(record["emitted_token_ids"]) for record in records),
            "scope": "Exact emitted greedy tokens and first tokens for fresh teacher-forced prefixes; no native logits API",
            "modes": reports}


def native_passed(report):
    return report["emitted_tokens_compared_per_mode"] > 0 and all(check["passed"] for mode in report["modes"]
               for check in mode["reset_replay"] +
                   [prompt["full_greedy"] for prompt in mode["prompts"]] +
                   [check for prompt in mode["prompts"] for check in prompt["teacher_forced_first_tokens"]])


def compare_logits(expected, actual, np):
    if expected.shape != actual.shape:
        return {"passed": False, "expected_shape": list(expected.shape), "actual_shape": list(actual.shape)}
    expected = expected.astype(np.float64)
    actual = actual.astype(np.float64)
    finite = np.isfinite(expected) & np.isfinite(actual)
    error = np.abs(expected - actual)
    tolerance = BF16_ABSOLUTE_TOLERANCE + BF16_RELATIVE_TOLERANCE * np.maximum(np.abs(expected), np.abs(actual))
    outside = ~(finite & (error <= tolerance))
    expected_tokens = expected.argmax(axis=-1).tolist()
    actual_tokens = actual.argmax(axis=-1).tolist()
    tokens_match = bool(finite.all()) and expected_tokens == actual_tokens
    denominator = float(np.linalg.norm(expected) * np.linalg.norm(actual)) if finite.all() else 0.0
    cosine = (float(np.vdot(expected, actual) / denominator) if denominator
              else (1.0 if finite.all() and not expected.any() and not actual.any() else None))
    return {"passed": bool(finite.all() and not outside.any() and tokens_match),
            "shape": list(expected.shape), "max_absolute_error": float(error.max()) if finite.all() else None,
            "mean_absolute_error": float(error.mean()) if finite.all() else None,
            "cosine_similarity": cosine, "outside_tolerance": int(outside.sum()),
            "compared_values": int(expected.size), "nonfinite_values": int((~finite).sum()),
            "reference_argmax": expected_tokens, "native_argmax": actual_tokens,
            "argmax_matches": tokens_match}


def native_logits_reference(root, module_dir, records, output, capacity, modes, np):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(module_dir))
    from frontend.checkpoint import load_model
    import _edge_qwen3_logits_test as diagnostic

    config, weights, _ = load_model(root, "qwen3_bf16")
    model = diagnostic.Model(config, weights)
    reports = []
    for mode in modes:
        sessions = [model.new_session(capacity=capacity, graph=mode == "graph") for _ in records]
        prefill, incremental, growth = [], [], []
        for session, record in zip(sessions, records):
            logits = session.prefill(record["prompt_ids"])
            prefill.append(logits)
            incremental.append([logits[-1].copy()])
            growth.append([session.context_size])
        # Interleave distinct prefixes over shared immutable weights. Continuing
        # each session after another prefill/decode exposes accidental KV aliasing.
        for step in range(1, max(len(record["predicted_token_ids_including_eos"]) for record in records)):
            for index, (session, record) in enumerate(zip(sessions, records)):
                tokens = record["predicted_token_ids_including_eos"]
                if step < len(tokens):
                    incremental[index].append(session.decode(tokens[step - 1])[0])
                    growth[index].append(session.context_size)
        mode_report = {"mode": mode, "capacity": capacity, "prompts": []}
        for index, (session, record) in enumerate(zip(sessions, records)):
            with np.load(output / record["fixture"], allow_pickle=False) as fixture:
                expected_prefill = fixture["prefill_logits"]
                expected_incremental = fixture["incremental_last_logits"]
                expected_teacher = fixture["teacher_forced_last_logits"]
            observed_incremental = np.stack(incremental[index])
            teacher = [session.prefill(prefix)[-1].copy() for prefix in record["teacher_forced_prefixes"]]
            observed_teacher = np.stack(teacher)
            session.reset()
            replay = session.prefill(record["prompt_ids"])
            checks = {"prefill": compare_logits(expected_prefill, prefill[index], np),
                      "incremental": compare_logits(expected_incremental, observed_incremental, np),
                      "teacher_forced_first_rows": compare_logits(expected_teacher, observed_teacher, np),
                      "reset_replayed_prefill": compare_logits(expected_prefill, replay, np)}
            artifact = output / f"native-{mode}-prompt-{index:03d}.npz"
            np.savez(artifact, prefill_logits=prefill[index], incremental_last_logits=observed_incremental,
                     teacher_forced_last_logits=observed_teacher, reset_prefill_logits=replay)
            mode_report["prompts"].append({"checks": checks, "kv_lengths": growth[index],
                "kv_growth_matches": growth[index] == record["kv_lengths"],
                "fixture": artifact.name, "fixture_sha256": file_hash(artifact)})
            for stage, result in checks.items():
                print(f"Native {mode} prompt {index} {stage}: max_abs={result.get('max_absolute_error')}, "
                      f"mean_abs={result.get('mean_absolute_error')}, outside={result.get('outside_tolerance')}, "
                      f"argmax={result.get('argmax_matches')}")
        reports.append(mode_report)
        del session, sessions
        gc.collect()
    passed = all(prompt["kv_growth_matches"] and all(check["passed"] for check in prompt["checks"].values())
                 for mode in reports for prompt in mode["prompts"])
    return {"passed": passed, "module": str(diagnostic.__file__), "module_sha256": file_hash(Path(diagnostic.__file__)),
            "absolute_tolerance": BF16_ABSOLUTE_TOLERANCE, "relative_tolerance": BF16_RELATIVE_TOLERANCE,
            "tolerance_formula": "abs(actual-reference) <= 0.015625 + 0.01 * max(abs(reference),abs(actual)); exact row argmax also required",
            "scope": "All copied BF16 logits for recorded local-checkpoint prompts and teacher-forced continuations; private interleaved eager/graph histories and reset",
            "modes": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True, help="Full publisher Git SHA; local payload hashes are recorded separately")
    parser.add_argument("--expected-model-sha256", help="Verify the single local model.safetensors payload")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--teacher-forced-checks", type=int, default=4)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", action="append", help="Repeated local-tokenizer chat prompts; thinking disabled")
    prompts.add_argument("--prompt-ids", action="append", type=token_list, help="Repeated comma-separated token-ID prompts")
    parser.add_argument("--native-module-dir", type=Path, help="Optional directory containing built production model_bridge")
    parser.add_argument("--native-logit-module-dir", type=Path, help="Optional directory containing the test-only _edge_qwen3_logits_test module")
    parser.add_argument("--native-capacity", type=int, default=128)
    parser.add_argument("--native-mode", choices=("both", "eager", "graph"), default="both")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.model_revision):
        parser.error("--model-revision must be a full 40-character Git SHA")
    if args.expected_model_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_model_sha256):
        parser.error("--expected-model-sha256 must contain 64 hexadecimal characters")
    if args.steps < 1 or args.teacher_forced_checks < 1 or args.native_capacity < 1:
        parser.error("Step, teacher-forced check and capacity counts must be positive")
    root = args.checkpoint.resolve()
    files = checkpoint_files(root)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be absent or an empty directory")
    hashes = {str(file.relative_to(root)): {"bytes": file.stat().st_size, "sha256": file_hash(file)}
              for file in files}
    if args.expected_model_sha256:
        actual = hashes.get("model.safetensors", {}).get("sha256")
        if actual != args.expected_model_sha256.lower():
            raise ValueError(f"Single-file checkpoint checksum mismatch: {actual}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError("This reference tool supports text Qwen3 checkpoints only")
    if config.get("quantization_config"):
        raise ValueError("This BF16 reference tool requires a dense checkpoint")
    eos = config.get("eos_token_id")
    if not isinstance(eos, int) or isinstance(eos, bool):
        raise ValueError("Native token comparison requires a single integer model EOS token")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Pinned reference requires transformers=={TRANSFORMERS_VERSION}; found {transformers.__version__}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The CUDA reference requires an available Torch CUDA device")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.prompt_ids:
        prompt_ids = args.prompt_ids
    else:
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        prompt_ids = [tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
            enable_thinking=False) for text in args.prompt or DEFAULT_PROMPTS]
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            path = root / name
            if path.is_file():
                hashes[name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    for prompt in prompt_ids:
        if not prompt or any(token >= config["vocab_size"] for token in prompt):
            raise ValueError("Prompt token IDs must be nonempty and within the model vocabulary")
        if len(prompt) + args.steps > config["max_position_embeddings"]:
            raise ValueError("Prompt and continuation budget exceed the model position limit")
        if (args.native_module_dir or args.native_logit_module_dir) and len(prompt) + args.steps > args.native_capacity:
            raise ValueError("Prompt and continuation budget exceed --native-capacity")
    model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager").to(args.device).eval()
    source_path = Path(inspect.getsourcefile(type(model)))
    manifest = {"model_id": args.model_id, "model_revision": args.model_revision.lower(),
                "provenance_note": "Model ID/revision are supplied provenance; the local bytes are identified by SHA256",
                "checkpoint_files": hashes, "torch_version": torch.__version__,
                "transformers_version": transformers.__version__, "numpy_version": np.__version__,
                "reference_source": str(source_path), "reference_source_sha256": file_hash(source_path),
                "dtype": "bfloat16", "attention_implementation": "eager", "tf32": False,
                "deterministic_algorithms": True, "device": args.device,
                "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                "torch_cuda_version": torch.version.cuda, "steps": args.steps,
                "teacher_forced_checks": args.teacher_forced_checks,
                "scope": "Small deterministic full-checkpoint Torch reference and optional native greedy-token checks; no performance claim",
                "prompts": []}
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, prompt in enumerate(prompt_ids):
        record, arrays = torch_reference(model, prompt, args.steps, args.teacher_forced_checks, eos, torch, np)
        fixture = args.output / f"prompt-{index:03d}.npz"
        np.savez(fixture, **arrays)
        record["fixture"] = fixture.name
        record["fixture_sha256"] = file_hash(fixture)
        record["array_shapes"] = {name: list(value.shape) for name, value in arrays.items()}
        records.append(record)
        print(f"Torch prompt {index}: {len(prompt)} prompt tokens, {len(record['emitted_token_ids'])} emitted greedy tokens")
    manifest["prompts"] = records
    del model
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    failed = False
    if args.native_module_dir:
        modes = ("eager", "graph") if args.native_mode == "both" else (args.native_mode,)
        try:
            manifest["native"] = native_reference(root, args.native_module_dir.resolve(), records,
                                                  eos, args.native_capacity, modes)
            failed = not native_passed(manifest["native"])
            manifest["native"]["passed"] = not failed
        except Exception as error:
            failed = True
            manifest["native"] = {"passed": False, "error_type": type(error).__name__, "error": str(error)}
    if args.native_logit_module_dir:
        modes = ("eager", "graph") if args.native_mode == "both" else (args.native_mode,)
        try:
            manifest["native_logits"] = native_logits_reference(root, args.native_logit_module_dir.resolve(), records,
                                                                 args.output, args.native_capacity, modes, np)
            failed = failed or not manifest["native_logits"]["passed"]
        except Exception as error:
            failed = True
            manifest["native_logits"] = {"passed": False, "error_type": type(error).__name__, "error": str(error)}
    (args.output / "source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failed:
        print(f"Native comparison failed; exact diagnostics saved in {args.output / 'source.json'}", file=sys.stderr)
        return 1
    print(f"Reference and provenance saved in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
