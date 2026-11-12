"""CPU framework integration with a tiny VLM and the actual OFT MLP head."""

import ast
import contextlib
import importlib
import itertools
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch


oft = importlib.import_module("starVLA.model.framework.QwenOFT")


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.tokens = torch.nn.Embedding(100, 8)
        self.positions = torch.nn.Embedding(64, 8)
        self.dropout = torch.nn.Dropout(.6)
        self.checkpointing_disabled = False

    def gradient_checkpointing_disable(self):
        self.checkpointing_disabled = True

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return self.dropout(self.tokens(input_ids) + self.positions(positions)[None])


class FakeVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBackbone()
        self.processor = SimpleNamespace(tokenizer=lambda *_args, **_kwargs: {"input_ids": [99]})
        self.instructions = []
        self.image_batches = []
        self.calls = []
        self.feature_dtype = torch.float32

    def build_qwenvl_inputs(self, images, instructions):
        self.instructions.append(tuple(instructions))
        self.image_batches.append(images)
        horizon = instructions[0].count("🔍")
        # The extra prefix placeholder must be excluded by last-H-token selection.
        ids = torch.tensor([[99, 1, 2] + [99] * horizon] * len(images))
        return {"input_ids": ids}

    def _features(self, input_ids, path):
        result = self.model(input_ids).to(self.feature_dtype)
        self.calls.append({
            "path": path, "grad_enabled": torch.is_grad_enabled(),
            "requires_grad": result.requires_grad, "training": self.training,
        })
        return result

    def encode_last_hidden_state(self, input_ids):
        return self._features(input_ids, "last_hidden")

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(hidden_states=(self._features(input_ids, "legacy"),))


class ConditionReadGuard(dict):
    """Disabled branches must not inspect values, even if keys are supplied."""

    def __init__(self, values, forbidden):
        super().__init__(values)
        self.forbidden = set(forbidden)

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"Disabled condition was read: {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"Disabled condition was read: {key}")
        return super().get(key, default)


def configuration(include_state=True, include_tcp=True, *, enabled=True, omit_conditioning=False):
    framework = {
        "name": "QwenOFT", "qwenvl": {"base_vlm": "fake-local-vlm"},
        "action_model": {"action_model_type": "MLP", "action_horizon": 32, "future_action_window_size": 31, "past_action_window_size": 0, "action_dim": 14},
    }
    if not omit_conditioning:
        framework["conditioning"] = {
            "enabled": enabled, "include_state": include_state, "include_tcp": include_tcp,
            "freeze_vlm": True, "state_dim": 14, "tcp_dim": 14, "encoder_hidden_dim": 16,
        }
    return OmegaConf.create({
        "framework": framework,
        "datasets": {"vla_data": {
            "include_state": include_state, "include_tcp": include_tcp,
            "include_time_budget": False, "image_size": None, "obs_image_size": None,
        }},
        "trainer": {"shortest_angular_joint_loss": False},
    })


def examples():
    result = []
    for index in range(2):
        result.append({
            "image": [Image.new("RGB", (4, 4), (index * 40, 10, 20))],
            "lang": f"pick the bottle {index}",
            "action": np.full((32, 14), index + .25, dtype=np.float32),
            "state": np.linspace(-1, 1, 14, dtype=np.float32)[None] + index * .1,
            "target_tcp_pose": np.linspace(-.5, .5, 14, dtype=np.float32)[None] + index * .2,
            "robot_tag": "new_embodiment",
        })
    return result


class QwenOFTConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.autocast = patch.object(torch, "autocast", lambda *_args, **_kwargs: contextlib.nullcontext())
        self.autocast.start()
        self.addCleanup(self.autocast.stop)

    def model(self, cfg=None):
        torch.manual_seed(314)
        with patch.object(oft, "get_vlm_model", side_effect=lambda config: FakeVLM()):
            return oft.Qwenvl_OFT(config=configuration() if cfg is None else cfg)

    def test_frozen_backbone_stays_eval_and_all_new_modules_and_head_learn(self):
        model = self.model().train()
        self.assertTrue(model.training)
        self.assertFalse(model.qwen_vl_interface.training)
        self.assertFalse(model.qwen_vl_interface.model.dropout.training)
        self.assertTrue(model.qwen_vl_interface.model.checkpointing_disabled)
        self.assertTrue(all(not p.requires_grad for p in model.qwen_vl_interface.parameters()))
        for mode in (False, True, True):
            model.train(mode)
            self.assertFalse(model.qwen_vl_interface.training)
        loss = model(examples())["action_loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.qwen_vl_interface.parameters():
            self.assertIsNone(parameter.grad)
        for name, module in (
            ("projection", model.conditioning.output_projection),
            ("state", model.conditioning.state_encoder),
            ("tcp", model.conditioning.tcp_encoder),
            ("head", model.action_model),
        ):
            gradients = [parameter.grad for parameter in module.parameters()]
            with self.subTest(module=name):
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        self.assertTrue(all(call["path"] == "last_hidden" for call in model.qwen_vl_interface.calls))
        self.assertTrue(all(not call["grad_enabled"] and not call["requires_grad"]
                            for call in model.qwen_vl_interface.calls))

    def test_four_switch_combinations_do_not_read_disabled_conditions_or_change_prompts(self):
        for state_on, tcp_on in itertools.product((False, True), repeat=2):
            with self.subTest(state=state_on, tcp=tcp_on):
                model = self.model(configuration(state_on, tcp_on)).train()
                forbidden = set()
                if not state_on:
                    forbidden.add("state")
                if not tcp_on:
                    forbidden.update(("target_tcp_pose", "tcp_pose", "target_pose"))
                batch = [ConditionReadGuard(sample, forbidden) for sample in examples()]
                self.assertTrue(torch.isfinite(model(batch)["action_loss"]))
                output = model.predict_action(batch)["normalized_actions"]
                self.assertEqual(output.shape, (2, 32, 14))
                self.assertEqual(output.dtype, np.float32)
                for prompts in model.qwen_vl_interface.instructions:
                    for index, prompt in enumerate(prompts):
                        self.assertTrue(prompt.startswith(f"pick the bottle {index} Please predict"))
                        self.assertNotIn("[STATE]", prompt)
                        self.assertEqual(prompt.count("🔍"), 32)

    def test_train_and_prediction_use_identical_fused_features_and_actions(self):
        model = self.model().train()
        fused, predictions = [], []
        feature_hook = model.action_model.model.register_forward_pre_hook(
            lambda _module, args: fused.append(args[0].detach().clone())
        )
        action_hook = model.action_model.model.register_forward_hook(
            lambda _module, _args, output: predictions.append(output.detach().clone())
        )
        try:
            model(examples())
            output = model.predict_action(examples())["normalized_actions"]
        finally:
            feature_hook.remove()
            action_hook.remove()
        self.assertEqual(fused[0].shape, (64, 8))
        torch.testing.assert_close(fused[0], fused[1], rtol=0, atol=0)
        torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)
        np.testing.assert_array_equal(output, predictions[0].reshape(2, 32, 14).numpy())

    def test_bfloat_backbone_features_are_cast_for_the_float32_trainable_head(self):
        model = self.model().train()
        model.qwen_vl_interface.feature_dtype = torch.bfloat16
        loss = model(examples())["action_loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.conditioning.output_projection.weight.grad)
        output = model.predict_action(examples())["normalized_actions"]
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.isfinite(output).all())

    def test_conditioned_train_and_prediction_share_bucket_sizes_and_exact_pixels(self):
        cfg = configuration()
        cfg.datasets.vla_data.image_size = 224
        cfg.datasets.vla_data.image_size_buckets = "[(288, 288), (384, 288)]"
        model = self.model(cfg).train()
        batch = examples()
        rng = np.random.default_rng(17)
        for sample in batch:
            sample["image"] = [
                Image.fromarray(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)),
                Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)),
            ]
        model(batch)
        model.predict_action(batch)
        train_images, predict_images = model.qwen_vl_interface.image_batches
        for train_views, predict_views in zip(train_images, predict_images):
            self.assertEqual([view.size for view in train_views], [(384, 288), (288, 288)])
            for train_view, predict_view in zip(train_views, predict_views):
                np.testing.assert_array_equal(np.asarray(train_view), np.asarray(predict_view))
        # The official loader may already have resized the training frames.
        # Reapplying the shared resize must retain those exact pixels.
        for sample, train_views in zip(batch, train_images):
            sample["image"] = train_views
        model(batch)
        for original_views, repeated_views in zip(train_images, model.qwen_vl_interface.image_batches[-1]):
            for original_view, repeated_view in zip(original_views, repeated_views):
                np.testing.assert_array_equal(np.asarray(original_view), np.asarray(repeated_view))

    def test_disabled_adapter_does_not_call_new_image_preprocessor(self):
        cfg = configuration(False, False, enabled=False)
        cfg.datasets.vla_data.image_size_buckets = "[(288, 288), (384, 288)]"
        model = self.model(cfg).eval()
        with patch.object(oft, "preprocess_image_batch", side_effect=AssertionError("legacy image path changed")):
            model(examples())
            model.predict_action(examples())
        for batch in model.qwen_vl_interface.image_batches:
            self.assertEqual([sample[0].size for sample in batch], [(4, 4), (4, 4)])

    def test_missing_conditions_fail_for_the_actual_training_and_prediction_paths(self):
        model = self.model()
        for key in ("state", "target_tcp_pose"):
            for method in (model.forward, model.predict_action):
                batch = examples()
                del batch[1][key]
                with self.subTest(key=key, method=method.__name__), self.assertRaisesRegex(ValueError, key):
                    method(batch)

    def test_condition_shape_and_finiteness_validation_reaches_both_paths(self):
        model = self.model()
        for key in ("state", "target_tcp_pose"):
            for invalid in (np.zeros(13, dtype=np.float32), np.zeros((2, 14), dtype=np.float32),
                            np.full((1, 14), np.nan, dtype=np.float32),
                            np.full((1, 14), np.inf, dtype=np.float32)):
                for method in (model.forward, model.predict_action):
                    batch = examples()
                    batch[0][key] = invalid
                    with self.subTest(key=key, shape=invalid.shape, method=method.__name__):
                        with self.assertRaises(ValueError):
                            method(batch)

    def test_tcp_aliases_match_canonical_target_and_conflicts_are_rejected(self):
        model = self.model().eval()
        canonical = model.predict_action(examples())["normalized_actions"]
        for alias in ("tcp_pose", "target_pose"):
            aliased = examples()
            for sample in aliased:
                sample[alias] = sample.pop("target_tcp_pose")
            with self.subTest(alias=alias):
                np.testing.assert_array_equal(canonical, model.predict_action(aliased)["normalized_actions"])
            identical = examples()
            for sample in identical:
                sample[alias] = sample["target_tcp_pose"].copy()
            np.testing.assert_array_equal(canonical, model.predict_action(identical)["normalized_actions"])
            for method in (model.forward, model.predict_action):
                conflict = examples()
                conflict[0][alias] = conflict[0]["target_tcp_pose"] + .1
                with self.subTest(alias=alias, method=method.__name__):
                    with self.assertRaisesRegex(ValueError, "Conflicting"):
                        method(conflict)

    def test_flat_condition_vectors_match_singleton_token_inputs(self):
        model = self.model().eval()
        expected = model.predict_action(examples())["normalized_actions"]
        flat = examples()
        for sample in flat:
            sample["state"] = sample["state"][0]
            sample["target_tcp_pose"] = sample["target_tcp_pose"][0]
        np.testing.assert_array_equal(expected, model.predict_action(flat)["normalized_actions"])

    def test_disabled_adapter_has_no_new_checkpoint_keys_and_loads_strictly(self):
        legacy = self.model(configuration(False, False, omit_conditioning=True))
        self.assertFalse(legacy.conditioning.enabled)
        self.assertFalse(any(name.startswith("conditioning.") for name in legacy.state_dict()))
        self.assertEqual({name.split(".")[0] for name in legacy.state_dict()},
                         {"qwen_vl_interface", "action_model"})
        explicit = self.model(configuration(False, False, enabled=False))
        result = explicit.load_state_dict(legacy.state_dict(), strict=True)
        self.assertFalse(result.missing_keys or result.unexpected_keys)
        batch = examples()
        for sample in batch:
            del sample["state"]
        legacy.eval()
        explicit.eval()
        np.testing.assert_array_equal(legacy.predict_action(batch)["normalized_actions"],
                                      explicit.predict_action(batch)["normalized_actions"])
        self.assertEqual(legacy.qwen_vl_interface.calls[-1]["path"], "legacy")

    def test_official_legacy_path_does_not_add_state_text(self):
        model = self.model(configuration(False, False, enabled=False)).eval()
        model.predict_action(examples())
        self.assertTrue(all("[STATE]" not in prompt for prompt in model.qwen_vl_interface.instructions[-1]))

    def test_disabled_model_is_numerically_identical_to_pinned_official_source(self):
        # Execute only the undecorated class from the unmodified official commit,
        # so the comparison cannot overwrite the live framework registry.
        revision = "4c41b666228b70d8ae166bb65ac8f6b392ea7919"
        repository = Path(__file__).resolve().parents[1]
        try:
            source = subprocess.run(
                ["git", "show", f"{revision}:starVLA/model/framework/QwenOFT.py"],
                cwd=repository, check=True, capture_output=True, text=True,
            ).stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            self.skipTest("Pinned official source is unavailable in this source archive")
        original_class = next(node for node in ast.parse(source).body
                              if isinstance(node, ast.ClassDef) and node.name == "Qwenvl_OFT")
        original_class.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[original_class], type_ignores=[]))
        namespace = dict(vars(oft))
        namespace["get_vlm_model"] = lambda config: FakeVLM()
        exec(compile(module, "official_oft.py", "exec"), namespace)
        torch.manual_seed(314)
        original = namespace["Qwenvl_OFT"](
            config=configuration(False, False, omit_conditioning=True),
        ).eval()
        current = self.model(configuration(False, False, omit_conditioning=True)).eval()
        self.assertEqual(original.state_dict().keys(), current.state_dict().keys())
        for name, tensor in original.state_dict().items():
            torch.testing.assert_close(tensor, current.state_dict()[name], rtol=0, atol=0)
        for shortest in (False, True):
            original.use_shortest_angular_joint_loss = shortest
            current.use_shortest_angular_joint_loss = shortest
            batch = examples()
            batch[0]["action_valid_mask"] = np.tile([1., 0.], 16).astype(np.float32)
            batch[1]["action_dim_valid_mask"] = np.asarray([1.] * 12 + [0., 0.], dtype=np.float32)
            expected = original(batch)
            actual = current(batch)
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            np.testing.assert_array_equal(
                current.predict_action(batch)["normalized_actions"],
                original.predict_action(batch)["normalized_actions"],
            )
        expected["action_loss"].backward()
        actual["action_loss"].backward()
        for (name, original_parameter), (_, current_parameter) in zip(
            original.named_parameters(), current.named_parameters(),
        ):
            with self.subTest(parameter=name):
                if original_parameter.grad is None:
                    self.assertIsNone(current_parameter.grad)
                else:
                    torch.testing.assert_close(current_parameter.grad, original_parameter.grad, rtol=0, atol=0)

    def test_wrap_loss_matches_official_boundary_and_masks_only_twelve_joints(self):
        model = self.model().eval()
        batch = examples()
        predictions = model.predict_action(batch)["normalized_actions"]
        for sample, prediction in zip(batch, predictions):
            sample["action"] = prediction.copy()
            sample["action"][:, 0] -= 2 * math.pi - .2
            sample["action"][:, 13] -= 1
        model.use_shortest_angular_joint_loss = True
        wrapped = model(batch)["action_loss"]
        model.use_shortest_angular_joint_loss = False
        ordinary = model(batch)["action_loss"]
        self.assertAlmostEqual(float(wrapped), (math.pi + 1) / 14, places=6)
        self.assertAlmostEqual(float(ordinary), (2 * math.pi - .2 + 1) / 14, places=6)


class OFTWrappedLossTests(unittest.TestCase):
    def model(self):
        with patch.object(oft, "get_vlm_model", side_effect=lambda config: FakeVLM()):
            model = oft.Qwenvl_OFT(config=configuration(False, False, enabled=False))
        model.use_shortest_angular_joint_loss = True
        return model

    def test_official_wrap_loss_and_independent_step_and_dimension_masks(self):
        model = self.model()
        prediction = torch.zeros(1, 2, 14, requires_grad=True)
        target = torch.zeros_like(prediction)
        target[0, 0, 0] = -(2 * math.pi - .2)
        target[0, 1, 13] = -1
        steps = torch.tensor([[1., 0.]])
        dims = torch.zeros(1, 14)
        dims[:, 0] = 1
        joint_loss = model._compute_action_loss(
            prediction, target, ["new_embodiment"], action_valid_mask=steps, action_dim_valid_mask=dims,
        )
        self.assertAlmostEqual(float(joint_loss), math.pi, places=6)
        steps = 1 - steps
        dims.zero_()
        dims[:, 13] = 1
        gripper_loss = model._compute_action_loss(
            prediction, target, ["new_embodiment"], action_valid_mask=steps, action_dim_valid_mask=dims,
        )
        self.assertEqual(float(gripper_loss), 1)
        gripper_loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[0, 1, 13]), 1)

    def test_official_all_masked_loss_is_differentiable_zero(self):
        model = self.model()
        prediction = torch.randn(1, 2, 14, requires_grad=True)
        loss = model._compute_action_loss(
            prediction, torch.zeros_like(prediction), ["new_embodiment"],
            action_valid_mask=torch.zeros(1, 2),
        )
        self.assertEqual(float(loss), 0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


if __name__ == "__main__":
    unittest.main()
