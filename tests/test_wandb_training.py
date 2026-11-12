"""Validate W&B wiring and the numerical meaning of logged action errors."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from omegaconf import OmegaConf

from starVLA.training import train_starvla as training


class WandbTrainingTests(unittest.TestCase):
    def trainer(self, output_dir, *, main=True):
        trainer = training.VLATrainer.__new__(training.VLATrainer)
        trainer.config = OmegaConf.create({
            "run_id": "oft-state-tcp", "output_dir": str(output_dir),
            "wandb_project": "VLAct_Robotwin_TCP", "wandb_entity": None,
            "datasets": {"vla_data": {"include_tcp": True}},
            "framework": {"conditioning": {"include_tcp": "${datasets.vla_data.include_tcp}"}},
            "trainer": {"logging_frequency": 10},
        })
        trainer.accelerator = SimpleNamespace(is_main_process=main, wait_for_everyone=MagicMock())
        trainer.optimizer = SimpleNamespace(param_groups=[{"name": "action_model"}, {"name": "conditioning"}])
        trainer.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [1e-4, 2e-4])
        trainer.completed_steps = 20
        trainer.vla_train_dataloader = [None] * 4
        return trainer

    def test_online_logging_resolves_account_project_and_training_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.trainer(folder)
            with patch.dict(os.environ, {"WANDB_MODE": "online", "WANDB_ENTITY": "research-team",
                                         "WANDB_PROJECT": "tcp-ablation", "WANDB_NAME": "state-tcp-42"}), \
                    patch.object(training.wandb, "init", return_value=SimpleNamespace(url="https://wandb.ai/example/run")) as init, \
                    patch.object(training, "resolve_wandb_entity", return_value="research-team"), \
                    patch.object(training.wandb, "log") as log, \
                    patch.object(training, "logger"), \
                    patch.object(training.dist, "is_initialized", return_value=False):
                trainer._init_wandb()
                trainer._log_metrics({"action_loss": 0.25})
            kwargs = init.call_args.kwargs
            self.assertEqual(kwargs["entity"], "research-team")
            self.assertEqual(kwargs["project"], "tcp-ablation")
            self.assertEqual(kwargs["name"], "state-tcp-42")
            self.assertIs(kwargs["config"]["framework"]["conditioning"]["include_tcp"], True)
            self.assertTrue(Path(kwargs["dir"]).is_dir())
            metrics = log.call_args.args[0]
            self.assertEqual(metrics["action_loss"], 0.25)
            self.assertEqual(metrics["learning_rate/action_model"], 1e-4)
            self.assertEqual(metrics["learning_rate/conditioning"], 2e-4)
            self.assertEqual(log.call_args.kwargs["step"], 20)

    def test_disabled_mode_never_initializes_or_logs_a_run(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.trainer(folder)
            with patch.dict(os.environ, {"WANDB_MODE": "disabled"}), \
                    patch.object(training.wandb, "init") as init, patch.object(training.wandb, "log") as log, \
                    patch.object(training, "logger"), patch.object(training.dist, "is_initialized", return_value=False):
                trainer._init_wandb()
                trainer._log_metrics({"action_loss": 0.25})
            init.assert_not_called()
            log.assert_not_called()

    def test_only_main_process_creates_a_run(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.trainer(folder, main=False)
            with patch.dict(os.environ, {"WANDB_MODE": "offline"}), patch.object(training.wandb, "init") as init:
                trainer._init_wandb()
            init.assert_not_called()
            trainer.accelerator.wait_for_everyone.assert_called_once()

    def test_default_account_is_not_the_placeholder_and_logs_follow_frequency(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.trainer(folder)
            trainer.config.wandb_entity = "your_name"
            trainer.completed_steps = 21
            with patch.dict(os.environ, {"WANDB_MODE": "offline", "WANDB_ENTITY": ""}), \
                    patch.object(training.wandb, "init", return_value=SimpleNamespace(url=None)) as init, \
                    patch.object(training.wandb, "log") as log, patch.object(training, "logger"):
                trainer._init_wandb()
                trainer._log_metrics({"action_loss": 0.25})
            self.assertIsNone(init.call_args.kwargs["entity"])
            log.assert_not_called()

    def test_training_batch_mse_and_mae_are_true_elementwise_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.trainer(folder)
            trainer._get_next_batch = lambda: [
                {"action": np.zeros((2, 2))}, {"action": np.zeros((1, 1))},
            ]
            model = SimpleNamespace(predict_action=lambda **kwargs: {
                "normalized_actions": [np.array([[1., 2.], [3., 4.]]), np.array([[5.]])],
            })
            trainer.model = model
            with patch.object(training, "_unwrap_model", return_value=model), \
                    patch.object(training.dist, "is_initialized", return_value=False):
                metrics = trainer.eval_action_model({})
            self.assertEqual(metrics["train_action_mse"], 11.0)
            self.assertEqual(metrics["train_action_mae"], 3.0)
            self.assertNotIn("mse_score", metrics)

    def test_wrapped_oft_does_not_consume_an_extra_batch_or_report_raw_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.trainer(folder)
            trainer.config.framework.name = "QwenOFT"
            trainer.config.datasets.vla_data.data_mix = "robotwin_wrap_32"
            trainer._get_next_batch = MagicMock(side_effect=AssertionError("No diagnostic batch for wrapped OFT"))
            metrics = {"action_loss": 0.76}
            self.assertFalse(trainer._action_error_diagnostics_enabled())
            self.assertIs(trainer.eval_action_model(metrics), metrics)
            self.assertEqual(metrics, {"action_loss": 0.76})
            trainer._get_next_batch.assert_not_called()

    def test_angular_loss_also_disables_raw_errors_without_a_wrap_mix_name(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.trainer(folder)
            trainer.config.framework.name = "QwenOFT"
            trainer.config.trainer.shortest_angular_joint_loss = True
            self.assertFalse(trainer._action_error_diagnostics_enabled())

    def test_main_loop_skips_disabled_diagnostics_at_an_old_eval_interval(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.trainer(folder)
            trainer.config.framework.name = "QwenOFT"
            trainer.config.datasets.vla_data.data_mix = "robotwin_wrap_32"
            trainer.config.trainer.eval_interval = 1
            trainer.config.trainer.max_train_steps = 1
            trainer.config.trainer.save_interval = 100
            trainer.completed_steps = 0
            trainer.accelerator.sync_gradients = True
            trainer.accelerator.is_local_main_process = False
            trainer.model = MagicMock()
            trainer._log_training_config = MagicMock()
            trainer._create_data_iterators = MagicMock()
            trainer._get_next_batch = MagicMock(return_value=[])
            trainer._train_step = MagicMock(return_value={"action_loss": 0.76})
            trainer._log_metrics = MagicMock()
            trainer._finalize_training = MagicMock()
            trainer.eval_action_model = MagicMock(side_effect=AssertionError("Raw error evaluation must be skipped"))
            trainer.train()
            trainer._get_next_batch.assert_called_once()
            trainer.eval_action_model.assert_not_called()
            self.assertEqual(trainer._log_metrics.call_args.args[0]["action_loss"], 0.76)

    def test_wandb_failure_precedes_any_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            config = OmegaConf.create({"run_id": "auth-check", "run_root_dir": folder})
            accelerator = SimpleNamespace(is_main_process=True)
            with patch.object(training, "initialize_wandb", side_effect=RuntimeError("entity not found")), \
                    patch.object(training, "build_framework") as build, patch.object(training, "logger"):
                with self.assertRaisesRegex(RuntimeError, "entity not found"):
                    training.main(config, accelerator)
            build.assert_not_called()

    def test_failed_actual_run_initialization_reports_early_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.trainer(folder)
            with patch.dict(os.environ, {"WANDB_MODE": "online"}), \
                    patch.object(training, "resolve_wandb_entity", return_value="chosen-team"), \
                    patch.object(training.wandb, "init", side_effect=RuntimeError("cannot create run")), \
                    patch.object(training.dist, "is_initialized", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "before model loading"):
                    trainer._init_wandb()


if __name__ == "__main__":
    unittest.main()
