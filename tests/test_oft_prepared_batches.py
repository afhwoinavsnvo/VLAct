"""Worker-prepared inputs must preserve OFT predictions, loss and updates."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from test_qwenoft_conditioning import FakeVLM, configuration, examples, oft
from starVLA.training import train_starvla as training


class OFTPreparedBatchTests(unittest.TestCase):
    def setUp(self):
        self.autocast = patch.object(torch, "autocast", lambda *_args, **_kwargs: nullcontext())
        self.autocast.start()
        self.addCleanup(self.autocast.stop)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)

    def model(self):
        torch.manual_seed(314)
        with patch.object(oft, "get_vlm_model", side_effect=lambda config: FakeVLM()):
            return oft.Qwenvl_OFT(configuration()).train()

    def prepared(self, model, batch):
        _, inputs = model._prepare_vlm_inputs(batch)
        return {"qwen_inputs": inputs, "examples": batch}

    def test_prepared_batch_preserves_predictions_gradients_and_optimizer_update(self):
        original = self.model()
        optimized = copy.deepcopy(original)
        batch = examples()
        prepared = self.prepared(original, batch)
        with patch.object(optimized.qwen_vl_interface, "build_qwenvl_inputs",
                          side_effect=AssertionError("worker-prepared inputs were recomputed")):
            left_loss = original(batch)["action_loss"]
            right_loss = optimized(prepared)["action_loss"]
            torch.testing.assert_close(left_loss, right_loss, rtol=0, atol=0)
            left_loss.backward()
            right_loss.backward()
            for (name, left), (_, right) in zip(original.named_parameters(), optimized.named_parameters()):
                with self.subTest(parameter=name):
                    if left.grad is None:
                        self.assertIsNone(right.grad)
                    else:
                        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
            for model in (original, optimized):
                optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
                optimizer.step()
            for left, right in zip(original.parameters(), optimized.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            np.testing.assert_array_equal(
                original.predict_action(batch)["normalized_actions"],
                optimized.predict_action(prepared)["normalized_actions"],
            )

    def test_action_diagnostic_keeps_prepared_inputs_and_original_action_labels(self):
        model = self.model()
        batch = examples()
        prepared = self.prepared(model, batch)
        trainer = training.VLATrainer.__new__(training.VLATrainer)
        trainer.model = model
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        trainer._get_next_batch = MagicMock(return_value=prepared)
        with patch.object(training, "_unwrap_model", return_value=model), \
                patch.object(model, "predict_action", wraps=model.predict_action) as predict, \
                patch.object(training.dist, "is_initialized", return_value=False):
            metrics = trainer.eval_action_model({"action_loss": 0.5})
        self.assertIs(predict.call_args.kwargs["examples"], prepared)
        self.assertEqual(metrics["action_loss"], 0.5)
        self.assertTrue(np.isfinite(metrics["mse_score"]))

    def test_prepared_inputs_reject_mismatched_batch_size(self):
        model = self.model()
        prepared = self.prepared(model, examples())
        prepared["qwen_inputs"]["input_ids"] = prepared["qwen_inputs"]["input_ids"][:1]
        with self.assertRaisesRegex(ValueError, "input_ids must match"):
            model(prepared)


if __name__ == "__main__":
    unittest.main()
