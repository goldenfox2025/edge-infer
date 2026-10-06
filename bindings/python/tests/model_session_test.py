"""Exercise prepared Python models and session lifetimes on the FP32 CPU path."""

import gc
import sys
import unittest

import numpy as np

sys.path.insert(0, sys.argv.pop(1))
import model_bridge as bridge


def checkpoint():
    config = {
        "vocab_size": 8, "hidden_size": 4, "num_hidden_layers": 1,
        "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 4,
        "intermediate_size": 8, "max_position_embeddings": 16,
        "bos_token_id": 1, "eos_token_id": 7,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    }
    weights = {
        "model.embed_tokens.weight": np.tile(np.arange(1, 5, dtype=np.float32), (8, 1)),
        "model.norm.weight": np.ones(4, dtype=np.float32),
        "lm_head.weight": np.zeros((8, 4), dtype=np.float32),
    }
    weights["lm_head.weight"][2] = 1
    prefix = "model.layers.0."
    for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        weights[prefix + name] = np.ones(4, dtype=np.float32)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        weights[prefix + "self_attn." + name + ".weight"] = np.zeros((4, 4), dtype=np.float32)
    for name in ("gate_proj", "up_proj"):
        weights[prefix + "mlp." + name + ".weight"] = np.zeros((8, 4), dtype=np.float32)
    weights[prefix + "mlp.down_proj.weight"] = np.zeros((4, 8), dtype=np.float32)
    return config, weights


def collect(session, prompt=(1, 3), maximum=6):
    tokens = []
    session.generate(prompt, tokens.append, max_length=maximum)
    return tokens


class ModelSessionTest(unittest.TestCase):
    def setUp(self):
        self.config, self.weights = checkpoint()
        self.model = bridge.Model(self.config, self.weights, "qwen", device="cpu")

    def test_capacity_and_explicit_device(self):
        self.assertEqual(self.model.device, "cpu")
        self.assertEqual(self.model.max_context_length, 16)
        self.assertEqual(self.model.new_session().capacity, 16)
        self.assertEqual(self.model.new_session(6).capacity, 6)
        for capacity in (0, 17):
            with self.assertRaises(ValueError):
                self.model.new_session(capacity)
        with self.assertRaises(ValueError):
            bridge.Model(self.config, self.weights, "qwen", device="invalid")
        with self.assertRaises(ValueError):
            bridge.Model(self.config, self.weights, "qwen_bf16", device="cpu")

    def test_full_prompt_replaces_previous_request(self):
        session = self.model.new_session(8)
        self.assertEqual(collect(session), [2] * 4)
        self.assertEqual(collect(session, (5,), 3), [2] * 2)
        self.assertEqual(collect(session), [2] * 4)
        session.reset()
        self.assertEqual(collect(session), [2] * 4)

    def test_unsupported_checkpoint_semantics_reject_before_preparation(self):
        unsupported = (
            {"hidden_act": "gelu"},
            {"rope_scaling": {"rope_type": "linear", "factor": 2}},
            {"use_sliding_window": True, "sliding_window": 4},
            {"sliding_window": 4},
            {"layer_types": ["sliding_attention"]},
        )
        for settings in unsupported:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                bridge.Model(dict(self.config, **settings), self.weights, "qwen", device="cpu")
        allowed = dict(self.config, rope_scaling=None, use_sliding_window=False, sliding_window=4)
        self.assertEqual(bridge.Model(allowed, self.weights, "qwen", device="cpu").device, "cpu")

    def test_retains_prepared_weights_after_model_release(self):
        session = self.model.new_session(8)
        self.weights["lm_head.weight"].fill(0)
        del self.model
        gc.collect()
        self.assertEqual(collect(session), [2] * 4)

    def test_strided_numpy_checkpoint_weights_preserve_model_behavior(self):
        strided = {}
        for name, weight in self.weights.items():
            backing = np.full(weight.shape[:-1] + (weight.shape[-1] * 2,), -100.0, dtype=np.float32)
            view = backing[..., ::2]
            view[...] = weight
            self.assertFalse(view.flags.c_contiguous)
            strided[name] = view
        for model_type in ("qwen", "llama"):
            with self.subTest(model_type=model_type):
                model = bridge.Model(self.config, strided, model_type, device="cpu")
                self.assertEqual(collect(model.new_session(8)), [2] * 4)
        for name, view in strided.items():
            np.testing.assert_array_equal(view, self.weights[name])

    def test_independent_session_can_generate_inside_callback(self):
        first = self.model.new_session(8)
        second = self.model.new_session(8)
        nested = []

        def callback(token):
            if not nested:
                nested.extend(collect(second, (5,), 3))
            with self.assertRaisesRegex(RuntimeError, "busy"):
                first.reset()

        first.generate((1, 3), callback, max_length=6)
        self.assertEqual(nested, [2, 2])
        self.assertEqual(collect(first), [2] * 4)

    def test_callback_exception_preserves_original_and_releases_guard(self):
        session = self.model.new_session(8)
        original = ValueError("callback details")

        def callback(token):
            raise original

        with self.assertRaises(ValueError) as failure:
            session.generate((1, 3), callback, max_length=6)
        self.assertIs(failure.exception, original)
        self.assertEqual(collect(session), [2] * 4)

    def test_sampling_validation_and_context_bounds(self):
        session = self.model.new_session(6)
        with self.assertRaises((ValueError, RuntimeError)):
            session.generate((1, 3), lambda token: None, max_length=6, top_k=0)
        with self.assertRaises((ValueError, RuntimeError)):
            session.generate((1, 2, 3, 4, 5, 6, 1), lambda token: None, max_length=8)
        self.assertEqual(collect(session), [2] * 4)

    def test_compatibility_replacement_is_atomic(self):
        self.assertTrue(bridge.set_default_device("cpu"))
        self.assertTrue(bridge.init_model(self.config, self.weights, "qwen"))
        self.assertFalse(bridge.init_model({}, {}, "qwen"))
        tokens = []
        bridge.generate_text_stream((1, 3), tokens.append, max_length=6, top_k=1)
        self.assertEqual(tokens, [2] * 4)


if __name__ == "__main__":
    unittest.main()
