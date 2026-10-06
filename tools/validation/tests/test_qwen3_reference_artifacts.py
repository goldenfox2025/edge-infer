"""CPU falsification tests for saved Qwen3 numerical evidence; no Torch required."""

import copy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "qwen3_reference.py"
SPEC = importlib.util.spec_from_file_location("qwen3_reference", SOURCE)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


class FakeQwenSource:
    pass


class SavedReferenceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name).resolve()
        self.root, self.saved = self.base / "checkpoint", self.base / "saved"
        self.root.mkdir()
        self.saved.mkdir()
        self.config = {"model_type": "qwen3", "vocab_size": 8, "eos_token_id": 7,
                       "max_position_embeddings": 32}
        (self.root / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        (self.root / "model.safetensors").write_bytes(b"local fake weight payload; never loaded")
        self.hashes = {path.name: {"bytes": path.stat().st_size, "sha256": reference.file_hash(path)}
                       for path in reference.checkpoint_files(self.root)}
        self.software = {"torch_version": "test-torch", "transformers_version": reference.TRANSFORMERS_VERSION,
                         "numpy_version": np.__version__, "torch_cuda_version": None,
                         "reference_source_sha256": reference.file_hash(Path(__file__))}
        self.requested = {"model_id": "Qwen/Qwen3-0.6B", "model_revision": "a" * 40,
                          "attention_implementation": "eager", "device": "cpu", "steps": 2,
                          "teacher_forced_checks": 2, "prompt_ids": [[1, 2]]}
        prefill = np.zeros((2, 8), dtype="<f4")
        prefill[0, 1], prefill[1, 3] = 2, 2
        incremental = np.zeros((2, 8), dtype="<f4")
        incremental[0, 3], incremental[1, 4] = 2, 2
        self.arrays = {"prefill_logits": prefill, "incremental_last_logits": incremental,
                       "teacher_forced_last_logits": incremental.copy()}
        self.record = {"prompt_ids": [1, 2], "predicted_token_ids_including_eos": [3, 4],
                       "emitted_token_ids": [3, 4], "kv_lengths": [2, 3],
                       "teacher_forced_prefixes": [[1, 2], [1, 2, 3]],
                       "teacher_forced_first_token_ids": [3, 4],
                       "incremental_vs_full_prefix_first_token_ids": reference.compare_tokens([3, 4], [3, 4]),
                       "fixture": "prompt-000.npz", "array_shapes": {
                           name: list(array.shape) for name, array in self.arrays.items()}}
        self.manifest = {name: value for name, value in self.requested.items() if name != "prompt_ids"}
        self.manifest.update(self.software)
        self.manifest.update({"checkpoint_files": copy.deepcopy(self.hashes), "dtype": "bfloat16",
                              "tf32": False, "deterministic_algorithms": True, "prompts": [self.record],
                              "native_logits": {"passed": False, "absolute_tolerance": 0.015625,
                                                "relative_tolerance": 0.01}})
        self.write_arrays()

    def write_arrays(self):
        path = self.saved / "prompt-000.npz"
        np.savez(path, **self.arrays)
        self.record["fixture_sha256"] = reference.file_hash(path)

    def validate(self):
        manifest = self.saved / "source.json"
        manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
        return reference.saved_reference(manifest, self.root, self.hashes, self.config,
                                         self.requested, self.software, np)

    def test_valid_reference_preserves_prior_failure_and_tolerance(self):
        manifest, paths, reuse = self.validate()
        self.assertEqual(paths, [(self.saved / "prompt-000.npz").resolve()])
        self.assertFalse(reuse["prior_native_logits_passed"])
        self.assertEqual((reuse["prior_absolute_tolerance"], reuse["prior_relative_tolerance"]),
                         (reference.BF16_ABSOLUTE_TOLERANCE, reference.BF16_RELATIVE_TOLERANCE))
        self.assertEqual(manifest["prompts"], [self.record])
        self.assertEqual(reuse["manifest_sha256"], reference.file_hash(self.saved / "source.json"))

    def test_fixture_checksum_tamper_rejected(self):
        (self.saved / "prompt-000.npz").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.validate()

    def test_shape_dtype_nonfinite_and_object_arrays_rejected_even_with_new_hash(self):
        for name, value in (("wrong shape", np.zeros((1, 8), dtype="<f4")),
                            ("wrong dtype", np.zeros((2, 8), dtype=np.float64)),
                            ("nonfinite", np.full((2, 8), np.nan, dtype="<f4")),
                            ("object", np.zeros((2, 8), dtype=object))):
            with self.subTest(name=name):
                self.arrays["prefill_logits"] = value
                self.write_arrays()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_fixture_path_escape_and_absolute_paths_rejected(self):
        for name in ("../outside.npz", str((self.base / "outside.npz").resolve()),
                     "C:\\outside.npz", "nested\\outside.npz"):
            with self.subTest(name=name):
                self.record["fixture"] = name
                with self.assertRaises(ValueError):
                    self.validate()

    def test_symlink_escape_rejected(self):
        outside = self.base / "outside.npz"
        outside.write_bytes((self.saved / "prompt-000.npz").read_bytes())
        link = self.saved / "link.npz"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Host does not permit file symlinks")
        self.record["fixture"] = link.name
        with self.assertRaisesRegex(ValueError, "inside"):
            self.validate()

    def test_boolean_token_ids_and_lengths_rejected(self):
        for name in ("prompt_ids", "predicted_token_ids_including_eos", "emitted_token_ids",
                     "teacher_forced_first_token_ids", "kv_lengths"):
            with self.subTest(name=name):
                previous = self.record[name]
                self.record[name] = [True] + previous[1:]
                with self.assertRaises(ValueError):
                    self.validate()
                self.record[name] = previous
        self.record["teacher_forced_prefixes"][0][0] = True
        with self.assertRaises(ValueError):
            self.validate()

    def test_continuation_and_teacher_prefix_mismatch_rejected(self):
        self.record["teacher_forced_prefixes"][1][-1] = 5
        with self.assertRaisesRegex(ValueError, "prefix"):
            self.validate()
        self.record["teacher_forced_prefixes"][1][-1] = 3
        self.arrays["incremental_last_logits"][1, 5] = 3
        self.write_arrays()
        with self.assertRaisesRegex(ValueError, "argmax"):
            self.validate()

    def test_emitted_eos_kv_and_cached_full_prefix_report_mismatch_rejected(self):
        for name, value in (("emitted_token_ids", [3]), ("kv_lengths", [2, 4]),
                            ("predicted_token_ids_including_eos", [7, 4]),
                            ("incremental_vs_full_prefix_first_token_ids", {"passed": True})):
            with self.subTest(name=name):
                previous = self.record[name]
                self.record[name] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.record[name] = previous

    def test_initial_incremental_row_must_be_exact_prefill_last_row(self):
        self.arrays["incremental_last_logits"][0, 0] = 0.001
        self.write_arrays()
        with self.assertRaisesRegex(ValueError, "First incremental"):
            self.validate()

    def test_checkpoint_config_software_source_backend_and_prompt_mismatch_rejected(self):
        changes = {"model_id": "other", "model_revision": "b" * 40, "torch_version": "other",
                   "numpy_version": "other", "reference_source_sha256": "b" * 64,
                   "attention_implementation": "sdpa", "device": "cuda", "steps": True,
                   "tf32": True, "deterministic_algorithms": False}
        for name, value in changes.items():
            with self.subTest(name=name):
                previous = self.manifest[name]
                self.manifest[name] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.manifest[name] = previous
        for name in ("config.json", "model.safetensors"):
            with self.subTest(name=name):
                previous = self.manifest["checkpoint_files"][name]["sha256"]
                self.manifest["checkpoint_files"][name]["sha256"] = "b" * 64
                with self.assertRaisesRegex(ValueError, "bytes differ"):
                    self.validate()
                self.manifest["checkpoint_files"][name]["sha256"] = previous
        self.requested["prompt_ids"] = [[2, 1]]
        with self.assertRaisesRegex(ValueError, "Requested prompts"):
            self.validate()

    def test_exact_check_rejects_rounding_signed_zero_and_nonfinite(self):
        original = np.zeros((1, 8), dtype="<f4")
        for value in (1e-6, -0.0, np.nan, np.inf):
            with self.subTest(value=value):
                changed = original.copy()
                changed[0, 0] = value
                self.assertFalse(reference.compare_exact_logits(original, changed, np)["passed"])
        self.assertTrue(reference.compare_exact_logits(original, original.copy(), np)["passed"])
        changed = original.copy()
        changed[0, 0] = 1e-6
        self.assertTrue(reference.compare_logits(original, changed, np)["passed"])

    def test_native_exact_invariants_fail_even_when_strict_tolerance_passes(self):
        arrays = self.arrays
        for broken_reset, broken_graph in ((False, False), (True, False), (False, True)):
            with self.subTest(reset=broken_reset, graph=broken_graph):
                class Session:
                    def __init__(self, graph):
                        self.graph, self.context_size, self.replayed = graph, 0, False

                    def prefill(self, prefix):
                        self.context_size = len(prefix)
                        if len(prefix) == 2:
                            result = arrays["prefill_logits"].copy()
                        else:
                            result = np.zeros((len(prefix), 8), dtype="<f4")
                            result[-1] = arrays["teacher_forced_last_logits"][1]
                        if broken_reset and self.replayed or broken_graph and self.graph:
                            result[0, 0] += 1e-6
                        return result

                    def decode(self, token):
                        self.context_size += 1
                        return arrays["incremental_last_logits"][1:2].copy()

                    def reset(self):
                        self.context_size, self.replayed = 0, True

                class Model:
                    def __init__(self, config, weights):
                        pass

                    def new_session(self, capacity, graph):
                        return Session(graph)

                diagnostic = types.ModuleType("_edge_qwen3_logits_test")
                diagnostic.Model, diagnostic.__file__ = Model, str(SOURCE)
                checkpoint = types.ModuleType("frontend.checkpoint")
                checkpoint.load_model = lambda *args: ({}, {}, None)
                with patch.dict(sys.modules, {"_edge_qwen3_logits_test": diagnostic,
                                "frontend.checkpoint": checkpoint}), patch.object(sys, "path", sys.path.copy()):
                    result = reference.native_logits_reference(self.root, self.base, [self.record],
                        self.saved, 16, ("eager", "graph"), np)
                self.assertTrue(result["strict_reference_passed"])
                self.assertEqual(result["deterministic_invariants_passed"], not (broken_reset or broken_graph))
                self.assertEqual(result["passed"], not (broken_reset or broken_graph))
                self.assertTrue(result["cross_mode_prefill_exact"]["executed"])

    def test_configured_reference_backend_mismatch_rejected(self):
        model = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="sdpa"))
        self.assertEqual(reference.configured_attention(model, "sdpa"), "sdpa")
        with self.assertRaisesRegex(ValueError, "requested eager, actual sdpa"):
            reference.configured_attention(model, "eager")

    def test_all_reference_forwards_disable_attention_output_fallback(self):
        class Tensor:
            def __init__(self, values):
                self.values = np.asarray(values)

            def __getitem__(self, index):
                return Tensor(self.values[index])

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values.copy()

            def argmax(self):
                return Tensor(self.values.argmax())

            def item(self):
                return self.values.item()

        class Cache:
            def __init__(self, length):
                self.length = length

            def get_seq_length(self):
                return self.length

        class Model:
            device = "cpu"
            config = types.SimpleNamespace(_attn_implementation="sdpa", output_attentions=True)

            def __init__(self):
                self.calls, self.backends = [], []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                # Pinned Qwen3 inherits output_attentions from config and falls
                # back to eager when SDPA is asked to return attention matrices.
                output = kwargs.get("output_attentions", self.config.output_attentions)
                self.backends.append("eager" if output else self.config._attn_implementation)
                rows = kwargs["input_ids"].values.shape[1]
                logits = np.zeros((1, rows, 8), dtype="<f4")
                logits[..., 3] = 2
                past = kwargs.get("past_key_values")
                return types.SimpleNamespace(logits=Tensor(logits),
                    past_key_values=Cache(rows + (past.length if past else 0)))

        torch = types.SimpleNamespace(long=object(), inference_mode=nullcontext,
            tensor=lambda values, **kwargs: Tensor(values),
            isfinite=lambda tensor: np.isfinite(tensor.values))
        model = Model()
        record, arrays = reference.torch_reference(model, [1, 2], 3, 3, 7, torch, np)
        self.assertEqual(len(model.calls), 6)  # Prefill, two decodes and three fresh prefixes.
        self.assertTrue(all(call.get("output_attentions") is False for call in model.calls))
        self.assertEqual(model.backends, ["sdpa"] * 6)
        self.assertEqual(reference.configured_attention(model, "sdpa"), "sdpa")
        self.assertEqual(record["kv_lengths"], [2, 3, 4])
        self.assertEqual(arrays["incremental_last_logits"].shape, (3, 8))

    def test_reuse_main_loads_no_torch_model_preserves_failed_strict_gate(self):
        self.validate()
        torch = types.ModuleType("torch")
        class TorchVersion(str):
            pass

        # Real Torch 2.6 uses a str subclass, while JSON reloads plain strings.
        torch.__version__, torch.version = TorchVersion(self.software["torch_version"]), types.SimpleNamespace(cuda=None)
        transformers = types.ModuleType("transformers")
        transformers.__version__ = reference.TRANSFORMERS_VERSION
        transformers.AutoModelForCausalLM = Mock()
        transformers.AutoTokenizer = Mock()
        transformers.AutoModelForCausalLM.from_pretrained.side_effect = AssertionError("Unexpected model load")
        transformers.AutoTokenizer.from_pretrained.side_effect = AssertionError("Unexpected tokenizer load")
        model_source = types.ModuleType("transformers.models.qwen3.modeling_qwen3")
        model_source.Qwen3ForCausalLM = FakeQwenSource
        output = self.base / "fresh"
        argv = [str(SOURCE), "--checkpoint", str(self.root), "--output", str(output),
                "--model-revision", "a" * 40, "--device", "cpu", "--steps", "2",
                "--teacher-forced-checks", "2", "--reuse-reference", str(self.saved),
                "--native-logit-module-dir", str(self.base)]
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers,
                        "transformers.models.qwen3.modeling_qwen3": model_source}), \
                patch.object(sys, "argv", argv), \
                patch.object(reference, "native_logits_reference", return_value={"passed": False}) as native:
            self.assertEqual(reference.main(), 1)
        native.assert_called_once()
        result = json.loads((output / "source.json").read_text(encoding="utf-8"))
        self.assertFalse(result["native_logits"]["passed"])
        self.assertFalse(result["reference_reuse"]["prior_native_logits_passed"])
        self.assertEqual(reference.file_hash(output / "prompt-000.npz"), self.record["fixture_sha256"])
        transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        transformers.AutoTokenizer.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
