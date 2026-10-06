"""Test the local checkpoint boundary with real CPU safetensors files."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors import SafetensorError
from safetensors.torch import save_file

from frontend.checkpoint import load_model


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "model"
        self.root.mkdir()
        self.write_json("config.json", {"tie_word_embeddings": True})
        self.embedding = torch.tensor([[1.25, -2.0], [3.0, 4.5]], dtype=torch.float32)

    def write_json(self, name, content):
        (self.root / name).write_text(json.dumps(content), encoding="utf-8")

    def write_weights(self, name="model.safetensors", tensors=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors if tensors is not None else {
            "model.embed_tokens.weight": self.embedding
        }, str(path))
        return path

    def write_index(self, weights):
        self.write_json("model.safetensors.index.json", {"weight_map": weights})

    def test_single_file_values_and_preserved_tie_declaration(self):
        self.write_weights()
        config, weights, kind = load_model(self.root, "qwen")
        self.assertTrue(config["tie_word_embeddings"])
        self.assertEqual(kind, "qwen")
        torch.testing.assert_close(weights["model.embed_tokens.weight"], self.embedding)
        self.assertNotIn("lm_head.weight", weights)

    def test_indexed_files_follow_recorded_tensor_mapping(self):
        head = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        self.write_weights("shards/embedding.safetensors")
        self.write_weights("output.safetensors", {"lm_head.weight": head})
        self.write_index({
            "model.embed_tokens.weight": "shards/embedding.safetensors",
            "lm_head.weight": "output.safetensors",
        })
        _, weights, _ = load_model(self.root, "qwen")
        torch.testing.assert_close(weights["model.embed_tokens.weight"], self.embedding)
        torch.testing.assert_close(weights["lm_head.weight"], head)

    def test_bf16_and_awq_preserve_payload_contracts(self):
        packed = torch.tensor([[0x10203040, -1]], dtype=torch.int32)
        scales = torch.tensor([[0.125, 0.25]], dtype=torch.float16)
        self.write_weights(tensors={
            "model.embed_tokens.weight": self.embedding,
            "model.layers.0.self_attn.q_proj.qweight": packed,
            "model.layers.0.self_attn.q_proj.scales": scales,
        })
        self.write_json("config.json", {"quantization_config": {"group_size": 64}})
        config, weights, _ = load_model(self.root, "qwen3_awq")
        self.assertEqual(config["group_size"], 64)
        torch.testing.assert_close(weights["model.layers.0.self_attn.q_proj.qweight"], packed)
        torch.testing.assert_close(weights["model.layers.0.self_attn.q_proj.scales"], scales)
        torch.testing.assert_close(weights["model.embed_tokens.weight"], self.embedding.to(torch.bfloat16))

    def test_indexed_tensor_must_exist_in_the_recorded_shard(self):
        self.write_weights()
        self.write_index({"missing.weight": "model.safetensors"})
        with self.assertRaises(SafetensorError):
            load_model(self.root, "qwen")

    def test_declared_awq_formats_are_checked_before_tensor_loading(self):
        for declared in (
            {"quant_method": "gptq"}, {"bits": 8}, {"w_bit": True},
            {"zero_point": False}, {"zero_point": 1}, {"version": "GEMM"},
            {"version": "MARLIN"}, {"group_size": 0}, {"group_size": True},
            {"group_size": "64"}, {"q_group_size": False},
            {"q_group_size": 2147483648}, {"group_size": 32, "q_group_size": 64},
        ):
            with self.subTest(declared=declared):
                for config in (declared, {"quantization_config": declared}):
                    self.write_json("config.json", config)
                    with self.assertRaises(ValueError):
                        load_model(self.root, "qwen3_awq")
        self.write_json("config.json", {
            "group_size": 32, "quantization_config": {"group_size": 64}
        })
        with self.assertRaisesRegex(ValueError, "agree"):
            load_model(self.root, "qwen3_awq")

    def test_explicit_awq_metadata_is_preserved(self):
        self.write_weights()
        quantization = {
            "quant_method": "awq", "bits": 4, "zero_point": True,
            "version": "gemv", "group_size": 64,
        }
        self.write_json("config.json", {"quantization_config": quantization})
        config, _, _ = load_model(self.root, "qwen3_awq")
        self.assertEqual(config["group_size"], 64)
        self.assertEqual(config["quantization_config"], quantization)
        self.write_json("config.json", {
            "q_group_size": 64, "quantization_config": {**quantization, "q_group_size": 64}
        })
        config, _, _ = load_model(self.root, "qwen3_awq")
        self.assertEqual(config["group_size"], 64)
        self.assertEqual(config["q_group_size"], 64)

    def test_missing_shard_rejects(self):
        self.write_index({"model.embed_tokens.weight": "missing.safetensors"})
        with self.assertRaises(FileNotFoundError):
            load_model(self.root, "qwen")

    def test_parent_and_absolute_shards_cannot_read_valid_outside_files(self):
        outside = Path(self.directory.name) / "outside.safetensors"
        save_file({"model.embed_tokens.weight": self.embedding}, str(outside))
        for name in ("../outside.safetensors", str(outside.resolve())):
            with self.subTest(name=name):
                self.write_index({"model.embed_tokens.weight": name})
                with self.assertRaises(ValueError):
                    load_model(self.root, "qwen")

    def test_symlink_cannot_escape_checkpoint_directory(self):
        outside = Path(self.directory.name) / "outside.safetensors"
        save_file({"model.embed_tokens.weight": self.embedding}, str(outside))
        try:
            (self.root / "linked.safetensors").symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symlink creation is unavailable: {error}")
        self.write_index({"model.embed_tokens.weight": "linked.safetensors"})
        with self.assertRaises(ValueError):
            load_model(self.root, "qwen")

    def test_truncated_actual_payload_rejects(self):
        path = self.write_weights()
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(SafetensorError):
            load_model(self.root, "qwen")

    def test_invalid_configuration_and_index_headers_reject(self):
        self.write_weights()
        self.write_json("config.json", [])
        with self.assertRaises(ValueError):
            load_model(self.root, "qwen")
        self.write_json("config.json", {})
        for index in ({}, {"weight_map": {}}, {"weight_map": []},
                      {"weight_map": {"": "model.safetensors"}},
                      {"weight_map": {"embedding": 3}}):
            with self.subTest(index=index):
                self.write_json("model.safetensors.index.json", index)
                with self.assertRaises(ValueError):
                    load_model(self.root, "qwen")


if __name__ == "__main__":
    unittest.main()
