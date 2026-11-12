"""Reject mismatched OFT/TCP recipes before expensive model initialization."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from omegaconf import OmegaConf

from examples.Robotwin.train_files.check_oft_setup import check_setup, check_training_config
from examples.Robotwin.train_files.prepare_clean_tcp_dataset import RAW_LAST_GOAL_POLICY


def training_config():
    return OmegaConf.create({
        "framework": {
            "name": "QwenOFT",
            "action_model": {"action_dim": 14, "action_horizon": 32,
                             "future_action_window_size": 31, "past_action_window_size": 0,
                             "use_time_budget": False},
            "conditioning": {"enabled": True, "freeze_vlm": True, "include_state": True,
                             "include_tcp": True, "state_dim": 14, "tcp_dim": 14},
        },
        "datasets": {"vla_data": {
            "data_root_dir": "/nonexistent/RoboTwin-Clean-TCP", "data_mix": "robotwin_wrap_32",
            "include_state": True, "include_tcp": True, "include_target_pose": False,
            "include_time_budget": False, "tcp_target_schema_version": 2,
            "expected_task_count": 50, "expected_episodes_per_task": 50,
            "train_start_frames": None, "drop_incomplete_action_chunks": False,
        }},
        "trainer": {
            "reload_modules": "qwen_vl_interface", "freeze_modules": "qwen_vl_interface",
            "random_init_action_model": True, "is_resume": False,
            "shortest_angular_joint_loss": True, "pretrained_checkpoint": "/nonexistent/source.pt",
        },
    })


class OFTSetupTests(unittest.TestCase):
    def test_all_four_state_tcp_switch_combinations_validate_without_io(self):
        for state in (False, True):
            for tcp in (False, True):
                with self.subTest(state=state, tcp=tcp):
                    config = training_config()
                    config.framework.conditioning.include_state = config.datasets.vla_data.include_state = state
                    config.framework.conditioning.include_tcp = config.datasets.vla_data.include_tcp = tcp
                    self.assertEqual(check_training_config(config), [])
                    with patch("examples.Robotwin.train_files.check_oft_setup._check_data", side_effect=AssertionError("No data I/O")), \
                            patch("examples.Robotwin.train_files.check_oft_setup._check_checkpoint", side_effect=AssertionError("No weights I/O")):
                        self.assertTrue(check_setup(config, config_only=True)["ready"])

    def test_wrong_pi_recipe_and_extra_model_loading_are_rejected(self):
        config = training_config()
        config.framework.name = "QwenPI_v4"
        config.trainer.reload_modules = "qwen_vl_interface,action_model"
        config.trainer.random_init_action_model = False
        config.framework.action_model.action_horizon = 16
        issues = check_training_config(config)
        self.assertTrue(any("framework.name must be QwenOFT" in issue for issue in issues))
        self.assertIn("trainer.reload_modules must be 'qwen_vl_interface'", issues)
        self.assertIn("trainer.random_init_action_model must be true", issues)
        self.assertIn("framework.action_model.action_horizon must be 32", issues)

    def test_non_boolean_mismatched_and_time_prefix_flags_fail(self):
        config = training_config()
        config.framework.conditioning.include_state = False
        config.datasets.vla_data.include_tcp = "False"
        config.datasets.vla_data.include_time_budget = True
        config.datasets.vla_data.train_start_frames = 5
        issues = check_training_config(config)
        self.assertIn("Model and dataset include_state switches disagree", issues)
        self.assertIn("datasets.vla_data.include_tcp must be a boolean", issues)
        self.assertIn("datasets.vla_data.include_time_budget must be false", issues)
        self.assertIn("Full-frame training requires train_start_frames=null", issues)

    def test_rawlast_policy_cannot_pass_even_after_renaming_directory(self):
        with tempfile.TemporaryDirectory(prefix="vlact-oft-setup-") as folder:
            root = Path(folder)
            data_root = root / "RoboTwin-Clean-TCP"
            data_root.mkdir()
            (data_root / "preparation_manifest.json").write_text(json.dumps({
                "status": "passed", "target_sampling": RAW_LAST_GOAL_POLICY,
                "tasks": [], "task_count": 0, "episode_count": 0, "public_frame_count": 0,
            }))
            checkpoint = root / "source/checkpoints/source.pt"
            checkpoint.parent.mkdir(parents=True)
            with zipfile.ZipFile(checkpoint, "w") as archive:
                archive.writestr("source/data.pkl", b"fixture header only")
                archive.writestr("source/version", b"3")
                archive.writestr("source/data/0", b"0")
            OmegaConf.save(OmegaConf.create({"framework": {
                "name": "QwenOFT", "action_model": {"action_dim": 14,
                                                       "future_action_window_size": 31, "past_action_window_size": 0},
            }}), checkpoint.parent.parent / "config.yaml")
            (checkpoint.parent.parent / "dataset_statistics.json").write_text(json.dumps({"new_embodiment": {}}))
            config = training_config()
            config.datasets.vla_data.data_root_dir = str(data_root)
            config.trainer.pretrained_checkpoint = str(checkpoint)
            report = check_setup(config)
            self.assertFalse(report["ready"])
            self.assertTrue(any("RawLast is not allowed" in issue for issue in report["issues"]))
            self.assertEqual(report["data"]["target_schema_version"], 3)
            self.assertFalse(report["checkpoint"]["full_weights_loaded"])


if __name__ == "__main__":
    unittest.main()
