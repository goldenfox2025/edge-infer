"""Exercise the production binding callback helper without a model or GPU."""

import gc
import sys
import threading
import traceback
import unittest
import weakref

sys.path.insert(0, sys.argv.pop(1))
import _edge_callback_boundary_test as boundary


class CallbackValueError(ValueError):
    pass


class CallbackBoundaryTest(unittest.TestCase):
    def test_tokens_on_caller(self):
        caller = threading.get_ident()
        received = []

        def callback(token):
            self.assertEqual(threading.get_ident(), caller)
            received.append(token)

        boundary.generate(callback, "normal")
        self.assertEqual(received, [2, 3])

    def test_worker_can_acquire_gil(self):
        caller = threading.get_ident()
        received = []

        def callback(token):
            self.assertNotEqual(threading.get_ident(), caller)
            received.append(token)

        boundary.generate(callback, "worker")
        self.assertEqual(received, [2, 3])

    def test_original_python_exception(self):
        for mode in ("normal", "worker", "swallow"):
            with self.subTest(mode=mode):
                calls = []
                original = CallbackValueError("callback details")

                def callback(token):
                    calls.append(token)
                    raise original

                try:
                    boundary.generate(callback, mode)
                except CallbackValueError as error:
                    self.assertIs(error, original)
                    self.assertEqual(str(error), "callback details")
                    frames = traceback.extract_tb(error.__traceback__)
                    self.assertIn("callback", [frame.name for frame in frames])
                else:
                    self.fail("The original callback exception was lost")
                self.assertEqual(calls, [2])

    def test_native_exception(self):
        with self.assertRaisesRegex(RuntimeError, "native generation failed"):
            boundary.generate(lambda token: None, "native_error")

    def test_nested_operations_rejected(self):
        def callback(token):
            with self.assertRaisesRegex(RuntimeError, "runtime is busy"):
                boundary.mutate_guarded()
            with self.assertRaisesRegex(RuntimeError, "runtime is busy"):
                boundary.generate_guarded(lambda token: None, "normal")

        boundary.generate_guarded(callback, "normal")
        boundary.mutate_guarded()

    def test_concurrent_operations_rejected(self):
        entered = threading.Event()
        resume = threading.Event()
        errors = []

        def callback(token):
            entered.set()
            if not resume.wait(timeout=5):
                raise RuntimeError("Test did not release the generation callback")

        def generate():
            try:
                boundary.generate_guarded(callback, "normal")
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=generate)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=5))
            with self.assertRaisesRegex(RuntimeError, "runtime is busy"):
                boundary.mutate_guarded()
            with self.assertRaisesRegex(RuntimeError, "runtime is busy"):
                boundary.generate_guarded(lambda token: None, "normal")
        finally:
            resume.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        boundary.mutate_guarded()

    def test_guard_resets_on_callback_failure(self):
        def callback(token):
            raise CallbackValueError("guarded callback failed")

        with self.assertRaisesRegex(CallbackValueError, "guarded callback failed"):
            boundary.generate_guarded(callback, "worker")
        boundary.mutate_guarded()

    def test_callback_not_retained(self):
        class Callback:
            def __call__(self, token):
                pass

        callback = Callback()
        callback_ref = weakref.ref(callback)
        references_before = sys.getrefcount(callback)
        for mode in ("normal", "worker", "swallow"):
            boundary.generate(callback, mode)
            self.assertEqual(sys.getrefcount(callback), references_before)
        del callback
        gc.collect()
        self.assertIsNone(callback_ref())


if __name__ == "__main__":
    unittest.main()
