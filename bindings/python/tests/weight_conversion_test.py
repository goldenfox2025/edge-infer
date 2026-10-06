"""Check native checkpoint casts and logical copies of strided caller weights."""

import sys
import unittest

import numpy as np

sys.path.insert(0, sys.argv.pop(1))
import _edge_weight_conversion_test as conversion

try:
    import torch
except ImportError:
    torch = None


class WeightConversionTest(unittest.TestCase):
    def test_numpy_float_views_copy_logical_values(self):
        matrix = np.arange(30, dtype=np.float64).reshape(5, 6)
        for view in (matrix[:, ::2], matrix.T, matrix[::-1, ::-1]):
            with self.subTest(strides=view.strides):
                before = view.copy()
                shape, values = conversion.float_values(view)
                self.assertEqual(shape, list(view.shape))
                np.testing.assert_array_equal(np.array(values, dtype=np.float32).reshape(shape),
                                              before.astype(np.float32))
                np.testing.assert_array_equal(view, before)

    def test_packed_numpy_views_preserve_signed_bits_and_order(self):
        matrix = np.array([[0x12345678, -1, 7, 11], [-2147483648, 0, 17, 23],
                           [31, 37, 41, 43]], dtype=np.int32)
        for view in (matrix[:, ::2], matrix.T, matrix[::-1, ::-1]):
            shape, values = conversion.packed_int_values(view)
            self.assertEqual(shape, list(view.shape))
            np.testing.assert_array_equal(np.array(values, dtype=np.int32).reshape(shape), view)

    def test_dense_decoder_mapping_preserves_layout_and_all_biases(self):
        projection = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
        weights = {"model.embed_tokens.weight": np.ones((5, 2), dtype=np.float32),
                   "model.norm.weight": np.ones(2, dtype=np.float32),
                   "lm_head.weight": np.arange(10, dtype=np.float32).reshape(5, 2),
                   "lm_head.bias": np.arange(5, dtype=np.float32)}
        for namespace, names in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                                 ("mlp", ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                weights[f"model.layers.0.{namespace}.{name}.weight"] = projection
                weights[f"model.layers.0.{namespace}.{name}.bias"] = np.arange(3, dtype=np.float32)
        mapped = conversion.mapped_dense(weights, n_layers=1)
        for name in ("wq0", "wk0", "wv0", "wo0", "w_gate0", "w_up0", "w_down0"):
            shape, strides, values = mapped[name]
            self.assertEqual(shape, [2, 3])
            self.assertEqual(strides, [1, 2])
            np.testing.assert_array_equal(np.array(values).reshape(shape), projection.T)
        for source, expected in weights.items():
            if source.endswith(".bias"):
                target = source.removeprefix("model.")
                shape, _, values = mapped[target]
                np.testing.assert_array_equal(np.array(values).reshape(shape), expected)
        if torch is not None:
            qwen3 = dict(weights)
            qwen3["model.layers.0.self_attn.q_norm.weight"] = np.ones(2, dtype=np.float32)
            qwen3["model.layers.0.self_attn.k_norm.weight"] = np.ones(2, dtype=np.float32)
            mapped_bf16 = conversion.mapped_dense(qwen3, qk_norm=True, bf16=True, n_layers=1)
            self.assertIn("q_norm0", mapped_bf16)
            self.assertIn("k_norm0", mapped_bf16)
            for source, expected in weights.items():
                if source.endswith(".bias"):
                    shape, _, values = mapped_bf16[source.removeprefix("model.")]
                    np.testing.assert_array_equal(np.array(values).reshape(shape), expected)

    def test_decoder_mapping_rejects_malformed_keys_and_collisions(self):
        value = np.ones((2, 2), dtype=np.float32)
        for key in ("model.layers.0junk.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.q_proj.weight.extra",
                    "model.layers.00.self_attn.q_proj.weight",
                    "model.layers.1.self_attn.q_proj.weight"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                conversion.mapped_dense({key: value}, n_layers=1)
        with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
            conversion.mapped_dense({"lm_head.weight": value, "model.lm_head.weight": value})
        with self.assertRaisesRegex(ValueError, "AWQ model"):
            conversion.mapped_dense({"lm_head.qweight": np.ones((2, 2), dtype=np.int32)})
        with self.assertRaisesRegex(ValueError, "rank mismatch"):
            conversion.mapped_dense({"lm_head.bias": value})

    def test_awq_metadata_admits_only_the_fixed_native_contract(self):
        accepted = ({}, {"group_size": 64}, {"quantization_config": {"q_group_size": 32}},
                    {"group_size": 64, "quantization_config": {
                        "quant_method": "AWQ", "version": "GEMV", "bits": 4,
                        "w_bit": 4, "zero_point": True, "group_size": 64}})
        self.assertEqual([conversion.awq_group_size(config) for config in accepted], [128, 64, 32, 64])
        rejected = ({"version": "GEMM"}, {"version": "Marlin"}, {"bits": 8}, {"bits": True},
                    {"bits": 4.0}, {"zero_point": False}, {"zero_point": 1},
                    {"quant_method": "gptq"}, {"group_size": True}, {"group_size": 0},
                    {"group_size": 2 ** 31},
                    {"quantization_config": None},
                    {"group_size": 64, "quantization_config": {"group_size": 128}},
                    {"quantization_config": {"group_size": 64, "q_group_size": 128}})
        for config in rejected:
            with self.subTest(config=config), self.assertRaises(ValueError):
                conversion.awq_group_size(config)

    def test_awq_packed_source_dtype_is_explicit(self):
        for dtype in (np.float32, np.int64, np.uint32, np.bool_):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "int32 dtype"):
                conversion.mapped_awq({"lm_head.qweight": np.ones((2, 2), dtype=dtype)})

    @unittest.skipIf(torch is None, "Torch is needed for BF16 checkpoint casts")
    def test_mixed_dense_awq_mapping_preserves_packing_and_rejects_incomplete_sets(self):
        packed = np.array([[0x12345678, -1], [-2147483648, 7], [11, 13]], dtype=np.int32)
        weights = {
            "model.layers.0.self_attn.q_proj.qweight": packed[:, ::-1],
            "model.layers.0.self_attn.q_proj.scales": np.array([[0.125], [0.25], [0.5]], dtype=np.float32),
            "model.layers.0.self_attn.q_proj.qzeros": np.array([[-1], [7], [11]], dtype=np.int32),
            "model.layers.0.self_attn.q_proj.bias": np.array([1, 2, 3], dtype=np.float32),
            "model.layers.0.self_attn.k_proj.weight": np.arange(8, dtype=np.float32).reshape(2, 4),
            "model.layers.0.self_attn.q_norm.weight": np.ones(4, dtype=np.float32),
        }
        dense, qweight, scales, zeros = conversion.mapped_awq(weights, qk_norm=True)
        shape, strides, values = qweight["wq0"]
        self.assertEqual(shape, [3, 2])
        self.assertEqual(strides, [2, 1])
        np.testing.assert_array_equal(np.array(values, dtype=np.int32).reshape(shape), packed[:, ::-1])
        np.testing.assert_array_equal(np.array(zeros["wq0"][2], dtype=np.int32).reshape(3, 1),
                                      weights["model.layers.0.self_attn.q_proj.qzeros"])
        self.assertEqual(scales["wq0"][0], [3, 1])
        self.assertEqual(dense["wk0"][0], [4, 2])
        self.assertIn("q_norm0", dense)
        self.assertIn("layers.0.self_attn.q_proj.bias", dense)
        for key in ("model.layers.0.self_attn.q_proj.qzeros", "model.layers.0.self_attn.q_proj.qweight"):
            incomplete = dict(weights)
            del incomplete[key]
            with self.assertRaises(ValueError):
                conversion.mapped_awq(incomplete, qk_norm=True)
        collision = dict(weights)
        collision["model.layers.0.self_attn.q_proj.weight"] = np.ones((3, 16), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "overlap"):
            conversion.mapped_awq(collision, qk_norm=True)

    @unittest.skipIf(torch is None, "Torch is needed for BF16 checkpoint casts")
    def test_numpy_bf16_cast_preserves_dtype_and_reversed_views(self):
        source = np.arange(24, dtype=np.float64).reshape(4, 6) / 7
        for view in (source, source.T, source[::-1, ::-1]):
            shape, bits = conversion.bf16_bits(view)
            expected = torch.as_tensor(view.copy()).to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)
            np.testing.assert_array_equal(np.array(bits, dtype=np.uint16).reshape(shape), expected)

    @unittest.skipIf(torch is None, "Torch is needed for BF16 checkpoint casts")
    def test_small_fp16_values_are_preserved_by_ordinary_bf16_cast(self):
        source = torch.tensor([[0.25, 0.00025, -0.000125, 0.0],
                               [0.125, 0.0005, -0.0005, -0.0]], dtype=torch.float16)
        for view in (source, source.T, source[:, ::2]):
            before = view.clone()
            shape, bits = conversion.bf16_bits(view)
            expected = before.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)
            self.assertEqual(shape, list(view.shape))
            np.testing.assert_array_equal(np.array(bits, dtype=np.uint16).reshape(shape), expected)
            torch.testing.assert_close(view, before, rtol=0, atol=0)
        expected_small = float(source[0, 1].to(torch.bfloat16))
        self.assertLess(expected_small, 0.001)

    @unittest.skipIf(torch is None, "Torch is needed for BF16 checkpoint casts")
    def test_bf16_transpose_slice_and_empty_preserve_payload(self):
        source = torch.arange(30, dtype=torch.float32).reshape(5, 6).to(torch.bfloat16)
        for view in (source.T, source[1::2, ::2], source[:0]):
            shape, bits = conversion.bf16_bits(view)
            expected = view.contiguous().view(torch.int16).numpy().view(np.uint16)
            self.assertEqual(shape, list(view.shape))
            np.testing.assert_array_equal(np.array(bits, dtype=np.uint16).reshape(shape), expected)

    @unittest.skipIf(torch is None, "Torch is needed for Tensor caller inputs")
    def test_torch_float_and_packed_views_detach_and_materialize(self):
        source = torch.arange(30, dtype=torch.float64).reshape(5, 6).requires_grad_()
        view = source.T[:, ::2]
        shape, values = conversion.float_values(view)
        np.testing.assert_array_equal(np.array(values, dtype=np.float32).reshape(shape),
                                      view.detach().float().numpy())
        self.assertTrue(source.requires_grad)
        self.assertIsNone(source.grad)
        packed = torch.arange(30, dtype=torch.int32).reshape(5, 6).T[:, ::2]
        shape, values = conversion.packed_int_values(packed)
        np.testing.assert_array_equal(np.array(values, dtype=np.int32).reshape(shape), packed.numpy())


if __name__ == "__main__":
    unittest.main()
