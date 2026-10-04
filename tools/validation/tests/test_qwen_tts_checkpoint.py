"""Offline checks using recorded official configs and safetensors headers.

The fixture contains metadata only. These tests do not download/load weights or
establish talker, codec or waveform correctness.
"""

import copy
from dataclasses import replace
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "conversion"))
import qwen_tts_checkpoint as checkpoint


FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "qwen_tts_0b6_metadata.json").read_text(encoding="utf-8"))
BASE = "Qwen3-TTS-12Hz-0.6B-Base"
CUSTOM = "Qwen3-TTS-12Hz-0.6B-CustomVoice"
CODEC = "Qwen3-TTS-Tokenizer-12Hz"


def recorded_metadata(name, source_file="model.safetensors"):
    recorded = FIXTURE["models"][name]
    header = json.dumps(recorded["header"], separators=(",", ":")).encode()
    header += b" " * (recorded["header_length"] - len(header))
    return checkpoint.parse_safetensors_header(
        header, source_file=source_file, file_size=recorded["file"]["size"]
    )


def tiny_safetensors(path, name="tiny", *, header=None, payload=b"\0\0\0\0"):
    header = header or {name: {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class ConditioningManifestTests(unittest.TestCase):
    def setUp(self):
        self.codec_config = copy.deepcopy(FIXTURE["models"][CODEC]["config"])
        self.codec_metadata = recorded_metadata(CODEC, "speech_tokenizer/model.safetensors")

    def manifest(self, name, *, config=None, metadata=None, **kwargs):
        model = FIXTURE["models"][name]
        return checkpoint.manifest_from_metadata(
            config or model["config"], [metadata or recorded_metadata(name)],
            codec_config=self.codec_config, codec_files=[self.codec_metadata],
            model_id=model["model_id"], model_revision=model["revision"],
            reference_revision=FIXTURE["reference_revision"], **kwargs,
        )

    def test_real_base_and_customvoice_configs_and_source_offsets(self):
        for name, variant, tensor_count in ((BASE, "base", 478), (CUSTOM, "custom_voice", 402)):
            with self.subTest(name=name):
                manifest = self.manifest(name)
                self.assertEqual(manifest["config"]["variant"], variant)
                self.assertEqual(len(recorded_metadata(name).tensors), tensor_count)
                self.assertEqual(len(manifest["tensors"]), 21)
                self.assertEqual(manifest["config"]["talker"]["head_dim"], 128)
                self.assertEqual(manifest["config"]["talker"]["query_width"], 2048)
                self.assertEqual(manifest["config"]["talker"]["hidden_size"], 1024)
                self.assertEqual(manifest["config"]["code_predictor"]["num_hidden_layers"], 5)
                self.assertEqual(manifest["codec"]["num_codebooks"], 16)
                self.assertEqual(manifest["codec"]["output_sample_rate"], 24000)
                self.assertEqual(manifest["codec"]["decode_upsample_rate"], 1920)
                self.assertFalse(manifest["execution"]["waveform_generation"])
                self.assertEqual(manifest["execution"]["talker"], "metadata_only")
                embedding = manifest["tensors"]["text_embedding"]
                self.assertEqual(embedding["shape"], [151936, 2048])
                self.assertEqual(embedding["nbytes"], 622329856)
                self.assertEqual(embedding["file_offset"],
                                 embedding["data_offset"] + 8 + FIXTURE["models"][name]["header_length"])
                self.assertEqual(manifest["tensors"]["codec_embedding.0"]["shape"], [3072, 1024])
                self.assertEqual(manifest["tensors"]["codec_embedding.15"]["shape"], [2048, 1024])
                self.assertEqual(manifest["source"]["model_revision"], FIXTURE["models"][name]["revision"])
        self.assertEqual(len(self.manifest(CUSTOM)["config"]["speakers"]), 9)
        self.assertEqual(self.manifest(BASE)["config"]["speakers"], {})

    def test_required_conditioning_tensor_missing(self):
        metadata = recorded_metadata(CUSTOM)
        entries = dict(metadata.tensors)
        del entries["talker.text_projection.linear_fc2.bias"]
        with self.assertRaisesRegex(checkpoint.CheckpointError, "Missing conditioning tensor"):
            self.manifest(CUSTOM, metadata=replace(metadata, tensors=entries))

    def test_same_byte_count_but_wrong_shape_rejected(self):
        metadata = recorded_metadata(CUSTOM)
        entries = copy.deepcopy(metadata.tensors)
        entries["talker.model.text_embedding.weight"]["shape"] = [2048, 151936]
        with self.assertRaisesRegex(checkpoint.CheckpointError, "requires BF16"):
            self.manifest(CUSTOM, metadata=replace(metadata, tensors=entries))

    def test_float16_cannot_be_relabelled_as_bfloat16(self):
        metadata = recorded_metadata(BASE)
        entries = copy.deepcopy(metadata.tensors)
        entries["talker.text_projection.linear_fc1.weight"]["dtype"] = "F16"
        with self.assertRaisesRegex(checkpoint.CheckpointError, "requires BF16"):
            self.manifest(BASE, metadata=replace(metadata, tensors=entries))

    def test_unsupported_configuration_and_implicit_head_dim(self):
        modifications = (
            lambda c: c.update(model_type="qwen3"),
            lambda c: c.update(tts_model_size="1b7"),
            lambda c: c.update(tts_model_type="voice_design"),
            lambda c: c["talker_config"].pop("head_dim"),
            lambda c: c["talker_config"].update(head_dim=64),
            lambda c: c["talker_config"]["code_predictor_config"].update(num_code_groups=32),
            lambda c: c.update(tts_bos_token_id=151936),
            lambda c: c["talker_config"].update(spk_is_dialect=None),
            lambda c: c.update(speaker_encoder_config=[]),
        )
        for change in modifications:
            config = copy.deepcopy(FIXTURE["models"][CUSTOM]["config"])
            change(config)
            with self.subTest(change=change), self.assertRaises(checkpoint.CheckpointError):
                self.manifest(CUSTOM, config=config)

    def test_codec_codebook_count_mismatch(self):
        self.codec_config["decoder_config"]["num_quantizers"] = 8
        with self.assertRaisesRegex(checkpoint.CheckpointError, "codebook count"):
            self.manifest(BASE)

    def test_mixed_namespace_codec_is_detected(self):
        entries = copy.deepcopy(FIXTURE["models"][CUSTOM]["header"])
        end = max(v["data_offsets"][1] for k, v in entries.items() if k != "__metadata__")
        entries["speech_tokenizer.decoder.example.weight"] = {
            "dtype": "F32", "shape": [1], "data_offsets": [end, end + 4]
        }
        header = json.dumps(entries, separators=(",", ":")).encode()
        metadata = checkpoint.parse_safetensors_header(
            header, source_file="model.safetensors", file_size=8 + len(header) + end + 4
        )
        manifest = self.manifest(CUSTOM, metadata=metadata)
        self.assertEqual(manifest["codec"]["layout"], "mixed_and_separate")
        self.assertEqual(manifest["namespaces"]["main"]["codec_decoder"], 1)
        self.assertNotIn("speech_tokenizer.decoder.example.weight", manifest["tensors"])

    def test_manifest_json_is_deterministic_and_provenance_is_explicit(self):
        manifest = self.manifest(CUSTOM)
        reordered = dict(reversed(list(manifest.items())))
        self.assertEqual(checkpoint.manifest_json(manifest), checkpoint.manifest_json(reordered))
        without_provenance = checkpoint.manifest_from_metadata(
            FIXTURE["models"][CUSTOM]["config"], [recorded_metadata(CUSTOM)]
        )
        self.assertIsNone(without_provenance["source"]["model_revision"])
        with self.assertRaisesRegex(checkpoint.CheckpointError, "full 40-character"):
            checkpoint.manifest_from_metadata(
                FIXTURE["models"][CUSTOM]["config"], [recorded_metadata(CUSTOM)], model_revision="main"
            )


class HeaderAndLocalPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def inspect(self, path):
        return checkpoint.inspect_safetensors(path, source_file=path.name)

    def test_small_complete_file_reads_only_header_metadata(self):
        path = self.root / "model.safetensors"
        tiny_safetensors(path)
        metadata = self.inspect(path)
        self.assertEqual(metadata.tensors["tiny"]["shape"], [2])
        self.assertEqual(metadata.file_size, metadata.data_start + 4)

    def test_truncated_length_and_header(self):
        path = self.root / "model.safetensors"
        for data in (b"", b"1234567", struct.pack("<Q", 50) + b"{}"):
            path.write_bytes(data)
            with self.subTest(data=data), self.assertRaisesRegex(checkpoint.CheckpointError, "Truncated"):
                self.inspect(path)

    def test_huge_declared_header_is_rejected_before_allocation(self):
        path = self.root / "model.safetensors"
        path.write_bytes(struct.pack("<Q", checkpoint.MAX_HEADER_BYTES + 1))
        with self.assertRaisesRegex(checkpoint.CheckpointError, "header length"):
            self.inspect(path)

    def test_truncated_payload_and_wrong_byte_extent(self):
        path = self.root / "model.safetensors"
        tiny_safetensors(path, payload=b"\0\0")
        with self.assertRaisesRegex(checkpoint.CheckpointError, "beyond file"):
            self.inspect(path)
        tiny_safetensors(path, header={"tiny": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 4]}})
        with self.assertRaisesRegex(checkpoint.CheckpointError, "byte extent"):
            self.inspect(path)

    def test_overlapping_or_gapped_tensors(self):
        path = self.root / "model.safetensors"
        for second in ([2, 6], [8, 12]):
            header = {"first": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                      "second": {"dtype": "BF16", "shape": [2], "data_offsets": second}}
            tiny_safetensors(path, header=header, payload=b"\0" * second[1])
            with self.subTest(second=second), self.assertRaisesRegex(checkpoint.CheckpointError, "non-contiguous"):
                self.inspect(path)

    def test_duplicate_json_keys_and_nonfinite_config_rejected(self):
        with self.assertRaisesRegex(checkpoint.CheckpointError, "Duplicate JSON"):
            checkpoint.parse_safetensors_header(b'{"x":{},"x":{}}', source_file="x", file_size=100)
        with self.assertRaisesRegex(checkpoint.CheckpointError, "Non-finite"):
            checkpoint._json(b'{"epsilon": NaN}', "config")

    def test_unsafe_source_path_and_invalid_dtype_are_controlled_errors(self):
        with self.assertRaisesRegex(checkpoint.CheckpointError, "Unsafe checkpoint"):
            checkpoint.parse_safetensors_header(b'{}', source_file="../outside.safetensors", file_size=10)
        header = b'{"x":{"dtype":[],"shape":[1],"data_offsets":[0,1]}}'
        with self.assertRaisesRegex(checkpoint.CheckpointError, "Unsupported safetensors dtype"):
            checkpoint.parse_safetensors_header(header, source_file="x.safetensors", file_size=9 + len(header))

    def test_sharded_index_and_wrong_shard_assignment(self):
        tiny_safetensors(self.root / "one.safetensors", "one")
        tiny_safetensors(self.root / "two.safetensors", "two")
        index = self.root / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"one": "one.safetensors", "two": "two.safetensors"}}))
        self.assertEqual(len(checkpoint._checkpoint_files(self.root, self.root)), 2)
        index.write_text(json.dumps({"weight_map": {"one": "two.safetensors"}}))
        with self.assertRaisesRegex(checkpoint.CheckpointError, "does not match"):
            checkpoint._checkpoint_files(self.root, self.root)

    def test_duplicate_tensor_names_across_shards(self):
        tiny_safetensors(self.root / "one.safetensors", "same")
        tiny_safetensors(self.root / "two.safetensors", "same")
        index = self.root / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"same": "one.safetensors", "other": "two.safetensors"}}))
        with self.assertRaisesRegex(checkpoint.CheckpointError, "Duplicate tensor across shards"):
            checkpoint._checkpoint_files(self.root, self.root)

    def test_shard_path_traversal_and_pickle_are_rejected(self):
        index = self.root / "model.safetensors.index.json"
        for filename in ("../outside.safetensors", "/absolute.safetensors", "C:/absolute.safetensors",
                         "nested\\outside.safetensors", "pytorch_model.bin"):
            index.write_text(json.dumps({"weight_map": {"some.weight": filename}}))
            with self.subTest(filename=filename), self.assertRaises(checkpoint.CheckpointError):
                checkpoint._checkpoint_files(self.root, self.root)

    def test_local_builder_reports_missing_conditioning_weights(self):
        (self.root / "config.json").write_text(json.dumps(FIXTURE["models"][CUSTOM]["config"]))
        tiny_safetensors(self.root / "model.safetensors")
        with self.assertRaisesRegex(checkpoint.CheckpointError, "Missing conditioning tensor"):
            checkpoint.build_manifest(self.root)


if __name__ == "__main__":
    unittest.main()
