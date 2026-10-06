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
