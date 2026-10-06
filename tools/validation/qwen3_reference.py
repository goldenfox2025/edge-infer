#!/usr/bin/env python3
"""Export local Qwen3 Torch logits and optionally compare native greedy tokens.

All model/tokenizer loads are local-only; remote code and automatic downloads
are disabled. Native token checks use the production Model/Session API. That
API does not expose logits. An optional test-only module compares copied native
logits without changing that API.
Eager and SDPA references retain separate provenance and the same strict bound.
Verified saved fixtures can be reused without loading a Torch model or tokenizer.
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
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
    root = root.resolve()
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token_ids(values, vocabulary, label, *, allow_empty=False):
    require(isinstance(values, list) and (values or allow_empty), f"Invalid {label}")
    require(all(type(value) is int and 0 <= value < vocabulary for value in values),
            f"{label} must contain integer vocabulary IDs, not booleans")
    return values


def contained_file(directory, name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name and
            not Path(name).is_absolute(), "Artifact filenames must be relative contained paths")
    path = (directory / name).resolve()
    require(path.is_relative_to(directory.resolve()) and path.is_file(),
            f"Artifact must be a file inside its recorded directory: {name}")
    return path


def saved_reference(manifest_path, root, hashes, config, requested, software, np):
    """Validate saved reference bytes and contexts before any native work.

    This function imports no Torch, initializes no model, and accepts no
    tolerance override. The caller supplies installed software/source identity.
    """
    manifest_path = manifest_path.resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "source.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict), "Reference manifest must be an object")
    for name, expected in requested.items():
        if name != "prompt_ids":
            require(type(manifest.get(name)) is type(expected) and manifest.get(name) == expected,
                    f"Reference provenance mismatch: {name}")
    for name, expected in software.items():
        require(name in manifest and type(manifest[name]) is type(expected) and manifest[name] == expected,
                f"Installed reference software/source differs: {name}")
    require(manifest.get("dtype") == "bfloat16" and manifest.get("tf32") is False and
            manifest.get("deterministic_algorithms") is True, "Unsupported reference numerical policy")
    checkpoint = manifest.get("checkpoint_files")
    require(isinstance(checkpoint, dict) and hashes.keys() <= checkpoint.keys(),
            "Reference is missing checkpoint/config hashes")
    for name, identity in checkpoint.items():
        require(isinstance(identity, dict) and type(identity.get("bytes")) is int and
                identity["bytes"] > 0 and isinstance(identity.get("sha256"), str) and
                re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]),
                "Malformed checkpoint identity")
        if name in hashes:
            actual = hashes[name]
        else:
            require(name in {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"},
                    "Unrecognized checkpoint provenance file")
            path = contained_file(root, name)
            actual = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        require(identity == actual, f"Checkpoint/config bytes differ: {name}")
    vocabulary = config["vocab_size"]
    require(type(vocabulary) is int and vocabulary > 0, "Invalid checkpoint vocabulary")
    eos, steps, teacher_checks = config["eos_token_id"], manifest["steps"], manifest["teacher_forced_checks"]
    require(type(eos) is int and 0 <= eos < vocabulary and type(steps) is int and steps > 0 and
            type(teacher_checks) is int and teacher_checks > 0 and
            type(config["max_position_embeddings"]) is int and config["max_position_embeddings"] > 0,
            "Invalid reference generation metadata")
    records = manifest.get("prompts")
    require(isinstance(records, list) and records, "Reference requires at least one recorded prompt")
    fixtures = []
    for index, record in enumerate(records):
        require(isinstance(record, dict), "Prompt record must be an object")
        prompt = token_ids(record.get("prompt_ids"), vocabulary, "prompt IDs")
        generated = token_ids(record.get("predicted_token_ids_including_eos"), vocabulary,
                              "teacher continuation")
        require(len(generated) <= steps and eos not in generated[:-1] and
                (len(generated) == steps or generated[-1] == eos), "Inconsistent continuation/EOS length")
        emitted = token_ids(record.get("emitted_token_ids"), vocabulary, "emitted IDs", allow_empty=True)
        require(emitted == (generated[:-1] if generated[-1] == eos else generated),
                "Emitted tokens differ from recorded continuation")
        growth = record.get("kv_lengths")
        require(isinstance(growth, list) and all(type(value) is int for value in growth) and
                growth == list(range(len(prompt), len(prompt) + len(generated))), "Invalid recorded KV growth")
        prefixes = record.get("teacher_forced_prefixes")
        count = min(teacher_checks, len(generated))
        require(isinstance(prefixes, list) and len(prefixes) == count, "Invalid teacher-prefix count")
        for step, prefix in enumerate(prefixes):
            token_ids(prefix, vocabulary, "teacher prefix")
            require(prefix == prompt + generated[:step], "Teacher prefix differs from continuation")
        teacher = token_ids(record.get("teacher_forced_first_token_ids"), vocabulary, "teacher argmax IDs")
        require(len(teacher) == count, "Invalid teacher argmax count")
        require(record.get("incremental_vs_full_prefix_first_token_ids") ==
                compare_tokens(generated[:count], teacher), "Inconsistent cached/full-prefix token report")
        path = contained_file(manifest_path.parent, record.get("fixture"))
        require(path.suffix == ".npz" and isinstance(record.get("fixture_sha256"), str) and
                re.fullmatch(r"[0-9a-f]{64}", record["fixture_sha256"]) and
                file_hash(path) == record["fixture_sha256"], f"Reference fixture checksum differs: {index}")
        shapes = {"prefill_logits": [len(prompt), vocabulary],
                  "incremental_last_logits": [len(generated), vocabulary],
                  "teacher_forced_last_logits": [count, vocabulary]}
        declared = record.get("array_shapes")
        require(isinstance(declared, dict) and declared.keys() == shapes.keys() and
                all(isinstance(shape, list) and all(type(size) is int for size in shape)
                    for shape in declared.values()) and declared == shapes, "Invalid declared reference shapes")
        with np.load(path, allow_pickle=False) as fixture:
            require(set(fixture.files) == set(shapes), "Unexpected reference array keys")
            arrays = {name: fixture[name] for name in shapes}
            for name, array in arrays.items():
                require(array.dtype == np.dtype("<f4") and list(array.shape) == shapes[name] and
                        bool(np.isfinite(array).all()), f"Invalid reference logits: {index}:{name}")
            require(arrays["incremental_last_logits"].argmax(-1).tolist() == generated and
                    arrays["teacher_forced_last_logits"].argmax(-1).tolist() == teacher,
                    "Logit argmax differs from recorded teacher tokens")
            require(arrays["prefill_logits"][-1].tobytes() ==
                    arrays["incremental_last_logits"][0].tobytes(), "First incremental row differs from prefill")
        require(len(prompt) + steps <= config["max_position_embeddings"],
                "Saved reference exceeds checkpoint position budget")
        fixtures.append(path)
    if requested.get("prompt_ids") is not None:
        require([record["prompt_ids"] for record in records] == requested["prompt_ids"],
                "Requested prompts differ from saved reference")
    for name in ("native", "native_logits"):
        report = manifest.get(name, {})
        require(isinstance(report, dict) and ("passed" not in report or type(report["passed"]) is bool),
                "Invalid prior native result status")
    return manifest, fixtures, {"manifest": str(manifest_path), "manifest_sha256": file_hash(manifest_path),
        "policy": "Verified saved reference; no Torch model or tokenizer loaded",
        "prior_native_passed": manifest.get("native", {}).get("passed"),
        "prior_native_logits_passed": manifest.get("native_logits", {}).get("passed"),
        "prior_absolute_tolerance": manifest.get("native_logits", {}).get("absolute_tolerance"),
        "prior_relative_tolerance": manifest.get("native_logits", {}).get("relative_tolerance")}


def cache_length(cache):
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2])


def configured_attention(model, requested):
    actual = model.config._attn_implementation
    require(isinstance(actual, str) and actual == requested,
            f"Configured reference attention differs: requested {requested}, actual {actual}")
    return actual


def torch_reference(model, prompt, steps, teacher_checks, eos, torch, np):
    tokens = torch.tensor([prompt], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        result = model(input_ids=tokens, use_cache=True, return_dict=True, output_attentions=False)
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
                           past_key_values=past, use_cache=True, return_dict=True,
                           output_attentions=False)
            past = result.past_key_values
            growth.append(cache_length(past))
            current = result.logits[0, -1]

        # Independent full-prefix forwards give the expected first token for
        # fresh native generation requests. They do not assume cached BF16
        # attention and full-prefix attention round identically.
        for step in range(min(teacher_checks, len(generated))):
            prefix = prompt + generated[:step]
            result = model(input_ids=torch.tensor([prefix], device=model.device),
                           use_cache=False, return_dict=True, output_attentions=False)
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


def compare_exact_logits(expected, actual, np):
    matching_shape = expected.shape == actual.shape and expected.dtype == actual.dtype
    finite = bool(np.isfinite(expected).all() and np.isfinite(actual).all())
    return {"passed": matching_shape and finite and expected.tobytes(order="C") == actual.tobytes(order="C"),
            "expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
            "finite": finite, "contract": "Exact dtype, shape and bytes; no tolerance"}


def native_logits_reference(root, module_dir, records, output, capacity, modes, np):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(module_dir))
    from frontend.checkpoint import load_model
    import _edge_qwen3_logits_test as diagnostic

    config, weights, _ = load_model(root, "qwen3_bf16")
    model = diagnostic.Model(config, weights)
    reports, prefills_by_mode = [], {}
    for mode in modes:
        sessions = [model.new_session(capacity=capacity, graph=mode == "graph") for _ in records]
        prefill, incremental, growth = [], [], []
        for session, record in zip(sessions, records):
            logits = session.prefill(record["prompt_ids"])
            prefill.append(logits.copy())
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
                "reset_replay_exact": compare_exact_logits(prefill[index], replay, np),
                "fixture": artifact.name, "fixture_sha256": file_hash(artifact)})
            for stage, result in checks.items():
                print(f"Native {mode} prompt {index} {stage}: max_abs={result.get('max_absolute_error')}, "
                      f"mean_abs={result.get('mean_absolute_error')}, outside={result.get('outside_tolerance')}, "
                      f"argmax={result.get('argmax_matches')}")
        reports.append(mode_report)
        prefills_by_mode[mode] = prefill
        del session, sessions
        gc.collect()
    cross_mode = [compare_exact_logits(eager, graph, np) for eager, graph in
                  zip(prefills_by_mode.get("eager", []), prefills_by_mode.get("graph", []))]
    strict_passed = all(all(check["passed"] for check in prompt["checks"].values())
                        for mode in reports for prompt in mode["prompts"])
    invariants_passed = all(prompt["kv_growth_matches"] and prompt["reset_replay_exact"]["passed"]
                            for mode in reports for prompt in mode["prompts"]) and all(
                                check["passed"] for check in cross_mode)
    return {"passed": strict_passed and invariants_passed, "strict_reference_passed": strict_passed,
            "deterministic_invariants_passed": invariants_passed,
            "cross_mode_prefill_exact": {"executed": {"eager", "graph"} <= set(modes), "checks": cross_mode},
            "module": str(diagnostic.__file__), "module_sha256": file_hash(Path(diagnostic.__file__)),
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
    parser.add_argument("--reference-attention", choices=("eager", "sdpa"), default="eager",
                        help="Explicit Torch attention backend; strict logit tolerances are unchanged")
    parser.add_argument("--reuse-reference", type=Path,
                        help="Verified source.json or its directory; rerun native checks without loading a Torch model")
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
    if args.reuse_reference and args.prompt:
        parser.error("--reuse-reference takes recorded token IDs; use --prompt-ids to assert their identity")
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
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Pinned reference requires transformers=={TRANSFORMERS_VERSION}; found {transformers.__version__}")
    if not args.reuse_reference and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The CUDA reference requires an available Torch CUDA device")
    source_path = Path(inspect.getsourcefile(Qwen3ForCausalLM))
    software = {"torch_version": str(torch.__version__), "transformers_version": str(transformers.__version__),
                "numpy_version": str(np.__version__), "torch_cuda_version": torch.version.cuda,
                "reference_source_sha256": file_hash(source_path)}
    if args.reuse_reference:
        requested = {"model_id": args.model_id, "model_revision": args.model_revision.lower(),
                     "attention_implementation": args.reference_attention, "device": args.device,
                     "steps": args.steps, "teacher_forced_checks": args.teacher_forced_checks,
                     "prompt_ids": args.prompt_ids}
        previous, fixtures, reuse = saved_reference(args.reuse_reference, root, hashes, config,
                                                   requested, software, np)
        prompt_ids = [record["prompt_ids"] for record in previous["prompts"]]
    elif args.prompt_ids:
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
        token_ids(prompt, config["vocab_size"], "prompt IDs")
        if len(prompt) + args.steps > config["max_position_embeddings"]:
            raise ValueError("Prompt and continuation budget exceed the model position limit")
        if (args.native_module_dir or args.native_logit_module_dir) and len(prompt) + args.steps > args.native_capacity:
            raise ValueError("Prompt and continuation budget exceed --native-capacity")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    if args.reuse_reference:
        manifest = {name: value for name, value in previous.items()
                    if name not in {"native", "native_logits", "reference_reuse"}}
        manifest["reference_reuse"] = reuse
        for index, (record, path) in enumerate(zip(previous["prompts"], fixtures)):
            copied = dict(record)
            fixture = args.output / f"prompt-{index:03d}.npz"
            shutil.copyfile(path, fixture)
            require(file_hash(fixture) == record["fixture_sha256"], "Reference bytes changed during copy")
            copied["fixture"] = fixture.name
            records.append(copied)
        print(f"Reused {len(records)} verified {args.reference_attention} references from {reuse['manifest']}; "
              "no Torch model or tokenizer loaded")
    else:
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16,
            attn_implementation=args.reference_attention).to(args.device).eval()
        actual_attention = configured_attention(model, args.reference_attention)
        manifest = {"model_id": args.model_id, "model_revision": args.model_revision.lower(),
                "provenance_note": "Model ID/revision are supplied provenance; the local bytes are identified by SHA256",
                "checkpoint_files": hashes, "torch_version": software["torch_version"],
                "transformers_version": software["transformers_version"], "numpy_version": software["numpy_version"],
                "reference_source": str(source_path), "reference_source_sha256": file_hash(source_path),
                "dtype": "bfloat16", "attention_implementation": actual_attention, "tf32": False,
                "deterministic_algorithms": True, "device": args.device,
                "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                "torch_cuda_version": torch.version.cuda, "steps": args.steps,
                "teacher_forced_checks": args.teacher_forced_checks,
                "scope": "Small deterministic full-checkpoint Torch reference and optional native greedy-token checks; no performance claim",
                "prompts": []}
        for index, prompt in enumerate(prompt_ids):
            record, arrays = torch_reference(model, prompt, args.steps, args.teacher_forced_checks, eos, torch, np)
            fixture = args.output / f"prompt-{index:03d}.npz"
            np.savez(fixture, **arrays)
            record["fixture"] = fixture.name
            record["fixture_sha256"] = file_hash(fixture)
            record["array_shapes"] = {name: list(value.shape) for name, value in arrays.items()}
            records.append(record)
            print(f"Torch prompt {index}: {len(prompt)} prompt tokens, {len(record['emitted_token_ids'])} emitted greedy tokens")
        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    manifest["prompts"] = records
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
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Refused invalid or unmatched reference evidence: {error}", file=sys.stderr)
        raise SystemExit(2)
