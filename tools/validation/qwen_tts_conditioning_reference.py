"""Generate native conditioning fixtures from a pinned Qwen3-TTS reference.

Torch is needed only by this developer tool. No checkpoint, model package or
reference code is downloaded automatically. The reference class is loaded from
the recorded Git object, not an arbitrary working-tree Python module.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REFERENCE_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
REFERENCE_FILE = "qwen_tts/core/models/modeling_qwen3_tts.py"
TEXT_IDS = [0, 77091, 151671, 151672]
CODEC_IDS = [0, 17, 2047, 127]


def read_bf16(path, shape, torch):
    data = path.read_bytes()
    expected = 2
    for dimension in shape:
        expected *= dimension
    if len(data) != expected:
        raise ValueError(f"BF16 tensor size mismatch: {path.name}")
    return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape).float()


def load_subset(directory, torch):
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    if source.get("dtype") != "BF16" or source.get("text_ids") != TEXT_IDS or source.get("codec_ids") != CODEC_IDS:
        raise ValueError("Unsupported conditioning subset metadata")
    roles = source["roles"]
    tensors = {}
    for role in ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias", "text_input"):
        tensors[role] = read_bf16(directory / f"{role}.bf16", roles[role]["shape"], torch)
    group_count = len([key for key in roles if key.startswith("codec_")])
    for group in range(group_count):
        role = f"codec_{group}"
        tensors[role] = read_bf16(directory / f"{role}.bf16", roles[role]["shape"], torch)
    return source, tensors, group_count


def load_checkpoint(directory, torch, model_id=None, model_revision=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
    from qwen_tts_checkpoint import build_manifest

    manifest = build_manifest(directory, model_id=model_id, model_revision=model_revision,
                              reference_revision=REFERENCE_REVISION)
    root = directory.resolve()
    tensors = {}
    roles = {"fc1_weight": "text_projection.fc1.weight", "fc1_bias": "text_projection.fc1.bias",
             "fc2_weight": "text_projection.fc2.weight", "fc2_bias": "text_projection.fc2.bias"}

    def read(entry, ids=None):
        if entry["dtype"] != "BF16":
            raise ValueError("Recorded fixture export currently requires BF16 checkpoint weights")
        file = root / entry["source_file"]
        with file.open("rb") as stream:
            if ids is None:
                stream.seek(entry["file_offset"])
                data = stream.read(entry["nbytes"])
                shape = entry["shape"]
            else:
                width = entry["shape"][1]
                chunks = []
                for token in ids:
                    if token < 0 or token >= entry["shape"][0]:
                        raise ValueError("Reference row index exceeds the checkpoint vocabulary")
                    stream.seek(entry["file_offset"] + token * width * 2)
                    chunk = stream.read(width * 2)
                    if len(chunk) != width * 2:
                        raise ValueError("Truncated checkpoint embedding row")
                    chunks.append(chunk)
                data = b"".join(chunks)
                shape = [len(ids), width]
        if len(data) != (entry["nbytes"] if ids is None else shape[0] * shape[1] * 2):
            raise ValueError("Truncated checkpoint tensor")
        return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape).float()

    for role, name in roles.items():
        tensors[role] = read(manifest["tensors"][name])
    tensors["text_input"] = read(manifest["tensors"]["text_embedding"], TEXT_IDS)
    group_count = len([key for key in manifest["tensors"] if key.startswith("codec_embedding.")])
    for group in range(group_count):
        tensors[f"codec_{group}"] = read(manifest["tensors"][f"codec_embedding.{group}"], CODEC_IDS)
    return {"checkpoint_manifest": manifest, "text_ids": TEXT_IDS, "codec_ids": CODEC_IDS}, tensors, group_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--weights-subset", type=Path)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-id", help="Record publisher/model ID for a complete local checkpoint")
    parser.add_argument("--model-revision", help="Record a full checkpoint Git SHA for a complete local checkpoint")
    args = parser.parse_args()
    if args.weights_subset and (args.model_id or args.model_revision):
        parser.error("Subset provenance is read from source.json; model options require --checkpoint")
    import torch
    import numpy as np

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the selected reference configuration")
    source = subprocess.run(
        ["git", "-C", str(args.reference_source), "show", f"{REFERENCE_REVISION}:{REFERENCE_FILE}"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    tree = ast.parse(source)
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "Qwen3TTSTalkerResizeMLP"]
    if len(definitions) != 1:
        raise ValueError("The pinned reference projection class is missing")
    namespace = {"nn": torch.nn, "ACT2FN": {"silu": torch.nn.functional.silu}}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), REFERENCE_FILE, "exec"), namespace)
    provenance, tensors, groups = (load_checkpoint(args.checkpoint, torch, args.model_id, args.model_revision) if args.checkpoint
                                   else load_subset(args.weights_subset, torch))
    text_features = tensors["fc1_weight"].shape[1]
    intermediate = tensors["fc1_weight"].shape[0]
    hidden = tensors["fc2_weight"].shape[0]
    rows = len(TEXT_IDS)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name, tensor in tensors.items():
        tensor.contiguous().numpy().astype("<f4", copy=False).tofile(args.output / f"{name}.f32")
    ids = np.repeat(np.arange(rows, dtype="<u4")[:, None], groups, axis=1)
    ids.tofile(args.output / "codec_ids.u32")
    (args.output / "dimensions.txt").write_text(f"{text_features} {intermediate} {hidden} {groups} {rows} {rows}\n")
    for label, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        model = namespace["Qwen3TTSTalkerResizeMLP"](text_features, intermediate, hidden, "silu", bias=True)
        model.load_state_dict({"linear_fc1.weight": tensors["fc1_weight"], "linear_fc1.bias": tensors["fc1_bias"],
                               "linear_fc2.weight": tensors["fc2_weight"], "linear_fc2.bias": tensors["fc2_bias"]})
        model = model.to(device=args.device, dtype=dtype).eval()
        with torch.no_grad():
            projected = model(tensors["text_input"].to(device=args.device, dtype=dtype))
            embedded = torch.cat([tensors[f"codec_{index}"].to(device=args.device, dtype=dtype)[:, None, :]
                                  for index in range(groups)], dim=1).sum(dim=1)
            composed = embedded + projected
        projected.float().cpu().numpy().astype("<f4", copy=False).tofile(args.output / f"text_expected_{label}.f32")
        composed.float().cpu().numpy().astype("<f4", copy=False).tofile(args.output / f"frame_expected_{label}.f32")
    provenance.update({"reference_revision": REFERENCE_REVISION,
                       "reference_class": "Qwen3TTSTalkerResizeMLP",
                       "reference_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                       "torch_version": torch.__version__, "device": args.device,
                       "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                       "scope": "selected real embedding rows and complete text projection; no transformer/codec inference"})
    provenance["fixture_sha256"] = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                    for file in sorted(args.output.iterdir()) if file.is_file() and file.name != "source.json"}
    (args.output / "source.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {rows}-row float32/BF16 conditioning reference with {groups} codebooks")


if __name__ == "__main__":
    main()
