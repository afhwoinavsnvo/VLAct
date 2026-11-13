"""Opt-in performance settings preserve default precision and optimizer behavior."""

import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf
import torch

from starVLA.training import train_starvla as training


class TinyConditionedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_model = torch.nn.Linear(3, 2)
        self.conditioning = torch.nn.Linear(2, 1)
        self.offset = torch.nn.Parameter(torch.tensor([0.1]))
        self.qwen_vl_interface = torch.nn.Linear(3, 3).requires_grad_(False)

    def forward(self, values):
        return self.conditioning(torch.relu(self.action_model(values))) + self.offset


def configuration():
    return OmegaConf.create({"trainer": {
        "gradient_accumulation_steps": 2,
        "freeze_modules": "qwen_vl_interface",
        "learning_rate": {"base": 0.01, "action_model": 0.02, "conditioning": 0.03},
        "optimizer": {"betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 1e-8},
        "lr_scheduler_type": "constant", "num_warmup_steps": 0,
        "max_train_steps": 4, "scheduler_specific_kwargs": {},
    }})


class AcceleratorPrecisionOptionTests(unittest.TestCase):
    def setUp(self):
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous_tf32)

    def create(self, config):
        observed = []
        accelerator = SimpleNamespace(state="mock state", print=MagicMock())

        def construct(**kwargs):
            observed.append((torch.backends.cuda.matmul.allow_tf32, kwargs))
            return accelerator

        with patch.dict(os.environ, {"STARVLA_DISABLE_DEEPSPEED": "1"}), \
                patch.object(training, "Accelerator", side_effect=construct):
            result = training.create_accelerator(config)
        self.assertIs(result, accelerator)
        self.assertEqual(observed[0][1]["gradient_accumulation_steps"], 2)
        return observed[0][0]

    def test_true_and_false_take_effect_before_accelerator_construction(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                torch.backends.cuda.matmul.allow_tf32 = not enabled
                config = configuration()
                config.trainer.allow_tf32 = enabled
                self.assertEqual(self.create(config), enabled)
                self.assertEqual(torch.backends.cuda.matmul.allow_tf32, enabled)

    def test_omitted_or_null_setting_preserves_existing_global_precision(self):
        for original in (False, True):
            for supplied_null in (False, True):
                with self.subTest(original=original, supplied_null=supplied_null):
                    torch.backends.cuda.matmul.allow_tf32 = original
                    config = configuration()
                    if supplied_null:
                        config.trainer.allow_tf32 = None
                    self.assertEqual(self.create(config), original)
                    self.assertEqual(torch.backends.cuda.matmul.allow_tf32, original)

    def test_tracked_configuration_and_false_string_do_not_enable_tf32(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        config = configuration()
        config.trainer.allow_tf32 = "false"
        self.assertFalse(self.create(training.wrap_config(config)))


class OptimizerPerformanceOptionTests(unittest.TestCase):
    def constructor_arguments(self, config):
        model = TinyConditionedModel()
        optimizer = MagicMock()
        with patch.object(training.torch.optim, "AdamW", return_value=optimizer) as constructor, \
                patch.object(training, "get_scheduler", return_value="scheduler"), \
                patch.object(training.dist, "is_initialized", return_value=False):
            result, scheduler = training.setup_optimizer_and_scheduler(model, config)
        self.assertIs(result, optimizer)
        self.assertEqual(scheduler, "scheduler")
        groups = constructor.call_args.args[0]
        self.assertEqual({group["name"]: group["lr"] for group in groups},
                         {"action_model": 0.02, "conditioning": 0.03, "base": 0.01})
        selected = {id(parameter) for group in groups for parameter in group["params"]}
        self.assertEqual(selected, {id(parameter) for parameter in model.parameters() if parameter.requires_grad})
        return constructor.call_args.kwargs

    def test_omitted_or_null_fused_setting_preserves_pytorch_default(self):
        for supplied_null in (False, True):
            with self.subTest(supplied_null=supplied_null):
                config = configuration()
                if supplied_null:
                    config.trainer.optimizer.fused = None
                kwargs = self.constructor_arguments(config)
                self.assertNotIn("fused", kwargs)
                self.assertEqual(kwargs["betas"], (0.9, 0.95))
                self.assertEqual(kwargs["eps"], 1e-8)
                self.assertEqual(kwargs["weight_decay"], 1e-8)

    def test_explicit_true_and_false_are_forwarded_without_changing_parameter_groups(self):
        for requested, expected in ((True, True), (False, False), ("false", False)):
            with self.subTest(requested=requested):
                config = configuration()
                config.trainer.optimizer.fused = requested
                kwargs = self.constructor_arguments(training.wrap_config(config))
                self.assertIs(kwargs["fused"], expected)
                self.assertEqual(kwargs["lr"], 0.01)

    def test_cpu_fused_updates_match_default_adamw_and_keep_frozen_parameters(self):
        torch.manual_seed(314)
        baseline = TinyConditionedModel()
        fused = copy.deepcopy(baseline)
        frozen_before = {name: parameter.detach().clone() for name, parameter in fused.named_parameters()
                         if not parameter.requires_grad}
        values = torch.tensor([[1.0, -2.0, 3.0], [0.5, 0.25, -1.0]])
        target = torch.tensor([[0.2], [0.8]])
        default_config = configuration()
        fused_config = configuration()
        fused_config.trainer.optimizer.fused = True
        with patch.object(training.dist, "is_initialized", return_value=False):
            default_optimizer, default_scheduler = training.setup_optimizer_and_scheduler(baseline, default_config)
            fused_optimizer, fused_scheduler = training.setup_optimizer_and_scheduler(fused, fused_config)
        self.assertTrue(fused_optimizer.defaults["fused"])
        for _ in range(3):
            for model, optimizer, scheduler in ((baseline, default_optimizer, default_scheduler),
                                                 (fused, fused_optimizer, fused_scheduler)):
                loss = torch.nn.functional.mse_loss(model(values), target)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                try:
                    optimizer.step()
                except NotImplementedError as error:
                    if optimizer is fused_optimizer and "CPU" in str(error):
                        self.skipTest("This PyTorch build does not provide fused AdamW on CPU")
                    raise
                scheduler.step()
                optimizer.zero_grad()
            for left, right in zip(baseline.parameters(), fused.parameters()):
                torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
        self.assertEqual({group["name"]: group["lr"] for group in fused_optimizer.param_groups},
                         {"action_model": 0.02, "conditioning": 0.03, "base": 0.01})
        for name, parameter in fused.named_parameters():
            if name in frozen_before:
                torch.testing.assert_close(parameter, frozen_before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
