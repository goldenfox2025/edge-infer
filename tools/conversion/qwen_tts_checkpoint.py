#!/usr/bin/env python3
"""Inspect a local Qwen3-TTS 0.6B checkpoint and export conditioning metadata.

Only JSON and safetensors headers are read. Tensor payloads are neither loaded nor
converted. The manifest describes text/codec embeddings and text projection;
talker, code predictor and codec execution remain outside this stage.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from typing import Mapping, Sequence


MAX_HEADER_BYTES = 64 * 1024 * 1024
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


class CheckpointError(ValueError):
    """The checkpoint does not satisfy the supported metadata contract."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: bytes, label: str):
    def reject_nonfinite(value):
        raise CheckpointError(f"Non-finite JSON value: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Invalid JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointError(f"Expected JSON object in {label}")
    return value


def _positive(config: Mapping, name: str, label: str) -> int:
    value = config.get(name)
    if type(value) is not int or value <= 0:
        raise CheckpointError(f"{label}.{name} must be an explicit positive integer")
    return value


def _sha(value, label):
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value)):
        raise CheckpointError(f"{label} must be a full 40-character Git revision")
    return value.lower() if value else None


def _relative_parts(relative: str) -> PurePosixPath:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise CheckpointError(f"Unsafe checkpoint shard path: {relative!r}")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(part in (".", "..") for part in parts.parts):
        raise CheckpointError(f"Unsafe checkpoint shard path: {relative!r}")
    return parts


@dataclass(frozen=True)
class SafetensorsMetadata:
    source_file: str
    data_start: int
    file_size: int
    tensors: Mapping[str, Mapping]


def parse_safetensors_header(header: bytes, *, source_file: str, file_size: int) -> SafetensorsMetadata:
    """Parse header bytes with the independently known complete file size.

    This also supports offline metadata fixtures. Passing a size is not evidence
    that a tensor payload has been downloaded or its checksum verified.
    """
    _relative_parts(source_file)
    if not 0 < len(header) <= MAX_HEADER_BYTES or not header.startswith(b"{"):
        raise CheckpointError(f"Invalid safetensors header in {source_file}")
    data_start = 8 + len(header)
    if type(file_size) is not int or file_size < data_start:
        raise CheckpointError(f"Truncated safetensors file: {source_file}")
    decoded = _json(header, source_file)
    tensors = {}
    for name, entry in decoded.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or any(not isinstance(v, str) for v in entry.values()):
                raise CheckpointError(f"Invalid safetensors metadata in {source_file}")
            continue
        if not isinstance(entry, dict):
            raise CheckpointError(f"Invalid tensor entry: {name}")
        dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise CheckpointError(f"Unsupported safetensors dtype for {name}: {dtype}")
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise CheckpointError(f"Invalid tensor shape: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int or v < 0 for v in offsets):
            raise CheckpointError(f"Invalid tensor offsets: {name}")
        begin, end = offsets
        if end < begin or end - begin != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise CheckpointError(f"Tensor byte extent does not match dtype/shape: {name}")
        if end > file_size - data_start:
            raise CheckpointError(f"Tensor extends beyond file: {name}")
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
    nonempty = sorted((v["data_offsets"][0], v["data_offsets"][1], k)
                      for k, v in tensors.items() if v["data_offsets"][1] > v["data_offsets"][0])
    previous_end = 0
    for begin, end, name in nonempty:
        if begin != previous_end:
            raise CheckpointError(f"Overlapping or non-contiguous tensor data: {name}")
        previous_end = end
    if previous_end != file_size - data_start:
        raise CheckpointError(f"Safetensors payload extent is inconsistent: {source_file}")
    return SafetensorsMetadata(source_file, data_start, file_size, tensors)


def inspect_safetensors(path: Path, *, source_file: str) -> SafetensorsMetadata:
    """Read the local header and validate offsets against the complete file size."""
    try:
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise CheckpointError(f"Truncated safetensors length: {source_file}")
            length = struct.unpack("<Q", length_bytes)[0]
            if not 0 < length <= MAX_HEADER_BYTES:
                raise CheckpointError(f"Invalid safetensors header length: {source_file}")
            header = stream.read(length)
            if len(header) != length:
                raise CheckpointError(f"Truncated safetensors header: {source_file}")
            return parse_safetensors_header(header, source_file=source_file, file_size=os.fstat(stream.fileno()).st_size)
    except OSError as error:
        raise CheckpointError(f"Cannot inspect {source_file}: {error}") from error


def normalize_config(config: Mapping) -> dict:
    if not isinstance(config, Mapping):
        raise CheckpointError("Expected a Qwen3-TTS configuration object")
    if config.get("model_type") != "qwen3_tts" or config.get("tts_model_size") != "0b6":
        raise CheckpointError("This stage supports only model_type=qwen3_tts, tts_model_size=0b6")
    variant = config.get("tts_model_type")
    if variant not in ("base", "custom_voice"):
        raise CheckpointError("This stage supports only Base and CustomVoice variants")
    if config.get("tokenizer_type") != "qwen3_tts_tokenizer_12hz":
        raise CheckpointError("This stage requires the 12Hz speech tokenizer")
    talker = config.get("talker_config")
    predictor = talker.get("code_predictor_config") if isinstance(talker, dict) else None
    if not isinstance(talker, dict) or not isinstance(predictor, dict):
        raise CheckpointError("Missing talker/code_predictor configuration")
    common = {
        "hidden_size": 1024, "intermediate_size": 3072, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 128, "num_code_groups": 16,
    }
    normalized = {}
    for label, values, extra in (
        ("talker", talker, {"num_hidden_layers": 28, "vocab_size": 3072,
                            "text_hidden_size": 2048, "text_vocab_size": 151936}),
        ("code_predictor", predictor, {"num_hidden_layers": 5, "vocab_size": 2048}),
    ):
        expected = {**common, **extra}
        for name, supported in expected.items():
            value = _positive(values, name, label)
            if value != supported:
                raise CheckpointError(f"Unsupported 0.6B configuration: {label}.{name}={value}; expected {supported}")
        if values.get("hidden_act") != "silu" or values.get("attention_bias") is not False:
            raise CheckpointError(f"Unsupported activation/attention bias in {label}")
        epsilon = values.get("rms_norm_eps")
        if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
            raise CheckpointError(f"Invalid RMSNorm epsilon in {label}")
        normalized[label] = {name: values[name] for name in expected}
        normalized[label].update({
            "query_width": values["num_attention_heads"] * values["head_dim"],
            "key_value_width": values["num_key_value_heads"] * values["head_dim"],
            "hidden_act": "silu", "attention_bias": False, "rms_norm_eps": epsilon,
            "rope_theta": values.get("rope_theta"), "rope_scaling": values.get("rope_scaling"),
        })
    def tokens(values, names, size, label):
        result = {}
        for name in names:
            value = values.get(name)
            if type(value) is not int or not 0 <= value < size:
                raise CheckpointError(f"Invalid token ID: {label}.{name}")
            result[name] = value
        return result

    normalized["text_tokens"] = tokens(
        config, ("assistant_token_id", "im_start_token_id", "im_end_token_id",
                 "tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id"),
        talker["text_vocab_size"], "text",
    )
    normalized["codec_tokens"] = tokens(
        talker, ("codec_bos_id", "codec_eos_token_id", "codec_pad_id", "codec_think_id",
                 "codec_nothink_id", "codec_think_bos_id", "codec_think_eos_id"),
        talker["vocab_size"], "codec",
    )
    for source, destination in (("codec_language_id", "languages"), ("spk_id", "speakers")):
        mapping = talker.get(source)
        if not isinstance(mapping, dict):
            raise CheckpointError(f"Missing talker.{source}")
        if any(not isinstance(name, str) or type(value) is not int or not 0 <= value < talker["vocab_size"]
               for name, value in mapping.items()):
            raise CheckpointError(f"Invalid token mapping in talker.{source}")
        normalized[destination] = dict(mapping)
    dialects = talker.get("spk_is_dialect", {})
    if not isinstance(dialects, dict) or any(not isinstance(value, (bool, str)) for value in dialects.values()):
        raise CheckpointError("Invalid talker.spk_is_dialect mapping")
    speaker_encoder = config.get("speaker_encoder_config")
    if speaker_encoder is not None and not isinstance(speaker_encoder, dict):
        raise CheckpointError("Invalid speaker_encoder_config")
    normalized["speaker_dialects"] = dict(dialects)
    normalized["speaker_encoder"] = speaker_encoder
    return {"variant": variant, "model_size": "0b6", "num_code_groups": 16, **normalized}


def conditioning_weights(config: Mapping) -> dict[str, tuple[str, list[int]]]:
    talker = config["talker"]
    text, hidden = talker["text_hidden_size"], talker["hidden_size"]
    weights = {
        "text_embedding": ("talker.model.text_embedding.weight", [talker["text_vocab_size"], text]),
        "text_projection.fc1.weight": ("talker.text_projection.linear_fc1.weight", [text, text]),
        "text_projection.fc1.bias": ("talker.text_projection.linear_fc1.bias", [text]),
        "text_projection.fc2.weight": ("talker.text_projection.linear_fc2.weight", [hidden, text]),
        "text_projection.fc2.bias": ("talker.text_projection.linear_fc2.bias", [hidden]),
        "codec_embedding.0": ("talker.model.codec_embedding.weight", [talker["vocab_size"], hidden]),
    }
    for group in range(1, config["num_code_groups"]):
        weights[f"codec_embedding.{group}"] = (
            f"talker.code_predictor.model.codec_embedding.{group - 1}.weight",
            [config["code_predictor"]["vocab_size"], hidden],
        )
    return weights


def _namespace(name):
    plain = name.removeprefix("speech_tokenizer.")
    if plain.startswith(("decoder.", "encoder.")):
        return "codec_" + plain.split(".")[0]
    if name.startswith("speaker_encoder."):
        return "speaker_encoder"
    if name.startswith("talker.code_predictor."):
        return "code_predictor"
    if name.startswith("talker."):
        return "talker"
    return "unknown"


def _merge(files):
    merged, groups = {}, {}
    for metadata in files:
        for name, tensor in metadata.tensors.items():
            if name in merged:
                raise CheckpointError(f"Duplicate tensor across shards: {name}")
            merged[name] = (metadata, tensor)
            group = _namespace(name)
            groups[group] = groups.get(group, 0) + 1
    return merged, groups


def manifest_from_metadata(config: Mapping, main_files: Sequence[SafetensorsMetadata], *,
                           codec_config: Mapping | None = None,
                           codec_files: Sequence[SafetensorsMetadata] = (),
                           model_id: str | None = None, model_revision: str | None = None,
                           reference_revision: str | None = None) -> dict:
    """Build a deterministic metadata manifest; this does not validate payload values."""
    normalized = normalize_config(config)
    main, main_groups = _merge(main_files)
    _, codec_groups = _merge(codec_files)
    tensors = {}
    for role, (name, shape) in conditioning_weights(normalized).items():
        if name not in main:
            raise CheckpointError(f"Missing conditioning tensor: {name}")
        metadata, tensor = main[name]
        if tensor["dtype"] != "BF16" or tensor["shape"] != shape:
            raise CheckpointError(f"Conditioning tensor requires BF16 {shape}: {name}")
        begin, end = tensor["data_offsets"]
        tensors[role] = {
            "name": name, "dtype": "BF16", "shape": shape, "source_file": metadata.source_file,
            "data_offset": begin, "file_offset": metadata.data_start + begin, "nbytes": end - begin,
        }
    codec = {"layout": "separate" if codec_files else "absent", "execution": "not_implemented"}
    if main_groups.get("codec_decoder") or main_groups.get("codec_encoder"):
        codec["layout"] = "mixed_and_separate" if codec_files else "mixed_main"
    if codec_config is not None:
        if codec_config.get("model_type") != "qwen3_tts_tokenizer_12hz":
            raise CheckpointError("Unsupported speech tokenizer model_type")
        decoder = codec_config.get("decoder_config")
        if not isinstance(decoder, dict):
            raise CheckpointError("Missing speech tokenizer decoder configuration")
        groups = _positive(decoder, "num_quantizers", "codec.decoder")
        if groups != normalized["num_code_groups"]:
            raise CheckpointError("Speech tokenizer codebook count does not match talker code groups")
        codec.update({
            "num_codebooks": groups, "codebook_size": _positive(decoder, "codebook_size", "codec.decoder"),
            "input_sample_rate": _positive(codec_config, "input_sample_rate", "codec"),
            "output_sample_rate": _positive(codec_config, "output_sample_rate", "codec"),
            "decode_upsample_rate": _positive(codec_config, "decode_upsample_rate", "codec"),
            "encode_downsample_rate": _positive(codec_config, "encode_downsample_rate", "codec"),
            "decoder_config": dict(decoder),
        })
    return {
        "format": "edge-infer.qwen-tts-conditioning", "version": 1, "stage": "conditioning",
        "source": {"model_id": model_id, "model_revision": _sha(model_revision, "model_revision"),
                   "reference_revision": _sha(reference_revision, "reference_revision")},
        "config": normalized,
        "tensor_layout": {"storage": "row_major", "linear_weights": "out_in", "conversion": "none"},
        "tensors": tensors,
        "files": [{"path": entry.source_file, "size": entry.file_size, "tensor_data_start": entry.data_start}
                  for entry in sorted([*main_files, *codec_files], key=lambda entry: entry.source_file)],
        "namespaces": {"main": main_groups, "codec": codec_groups},
        "codec": codec,
        "execution": {"conditioning_weights": "mapped", "talker": "metadata_only",
                      "code_predictor": "metadata_only", "waveform_generation": False,
                      "payload_values_verified": False},
    }


def _local_path(directory: Path, relative: str) -> Path:
    _relative_parts(relative)
    candidate = directory / relative
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError as error:
        raise CheckpointError(f"Checkpoint shard resolves outside its directory: {relative}") from error
    return candidate


def _checkpoint_files(directory: Path, root: Path) -> list[SafetensorsMetadata]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = _json(index_path.read_bytes(), str(index_path))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointError("Safetensors index has no weight_map")
        filenames = sorted(set(weight_map.values())) if all(isinstance(v, str) for v in weight_map.values()) else []
        if not filenames:
            raise CheckpointError("Invalid safetensors shard names in weight_map")
    else:
        weight_map = None
        filenames = ["model.safetensors"]
    files = []
    for filename in filenames:
        if not filename.endswith(".safetensors"):
            raise CheckpointError(f"Only safetensors shards are accepted: {filename}")
        path = _local_path(directory, filename)
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise CheckpointError(f"Checkpoint shard resolves outside checkpoint root: {filename}") from error
        files.append(inspect_safetensors(path, source_file=path.relative_to(root).as_posix()))
    merged, _ = _merge(files)
    if weight_map is not None:
        for name, filename in weight_map.items():
            if name not in merged or merged[name][0].source_file != (directory / filename).relative_to(root).as_posix():
                raise CheckpointError(f"Safetensors index does not match shard contents: {name}")
        if set(merged) != set(weight_map):
            raise CheckpointError("Safetensors index does not describe every shard tensor")
    return files


def build_manifest(checkpoint_dir, *, model_id=None, model_revision=None, reference_revision=None) -> dict:
    """Inspect a local, complete safetensors checkpoint; no network is used."""
    directory = Path(checkpoint_dir).resolve()
    try:
        config_bytes = (directory / "config.json").read_bytes()
        config = _json(config_bytes, "config.json")
        normalize_config(config)
        main_files = _checkpoint_files(directory, directory)
        codec_directory = directory / "speech_tokenizer"
        codec_config, codec_files = None, []
        if (codec_directory / "config.json").exists():
            codec_config = _json((codec_directory / "config.json").read_bytes(), "speech_tokenizer/config.json")
            if (codec_directory / "model.safetensors").exists() or (codec_directory / "model.safetensors.index.json").exists():
                codec_files = _checkpoint_files(codec_directory, directory)
        manifest = manifest_from_metadata(
            config, main_files, codec_config=codec_config, codec_files=codec_files,
            model_id=model_id, model_revision=model_revision, reference_revision=reference_revision,
        )
        manifest["source"]["config_file_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        return manifest
    except OSError as error:
        raise CheckpointError(f"Cannot read checkpoint: {error}") from error


def manifest_json(manifest: Mapping) -> str:
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Local checkpoint directory; no model downloads are performed")
    parser.add_argument("--output", type=Path, help="Manifest destination (default: standard output)")
    parser.add_argument("--model-id", help="Checkpoint publisher/model ID recorded as provenance")
    parser.add_argument("--model-revision", help="Full Hugging Face checkpoint Git SHA")
    parser.add_argument("--reference-revision", help="Full official Qwen3-TTS source Git SHA")
    args = parser.parse_args()
    try:
        manifest = build_manifest(args.checkpoint, model_id=args.model_id,
                                  model_revision=args.model_revision, reference_revision=args.reference_revision)
        output = manifest_json(manifest)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8", newline="\n")
        else:
            sys.stdout.write(output)
    except (CheckpointError, OSError) as error:
        parser.exit(2, f"Checkpoint inspection failed: {error}\n")


if __name__ == "__main__":
    main()
