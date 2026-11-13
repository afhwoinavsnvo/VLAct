"""Prevent mixing current TCP poses with legacy joint-state data recipes."""

from pathlib import Path
import unittest

from omegaconf import OmegaConf

from examples.Robotwin.train_files.check_oft_setup import check_training_config


class OFTTCPPoseStateSetupTests(unittest.TestCase):
    def config(self):
        root = Path(__file__).resolve().parents[1]
        return OmegaConf.load(root / 'examples/Robotwin/train_files/starvla_oft_tcp_pose_state.yaml')

    def test_pose_recipe_passes_without_reusing_joint_state_statistics(self):
        self.assertEqual(check_training_config(self.config()), [])

    def test_legacy_joint_mix_cannot_be_used_for_pose_state(self):
        cfg = self.config()
        cfg.datasets.vla_data.data_mix = 'robotwin_wrap_32'
        self.assertTrue(any('data_mix' in issue for issue in check_training_config(cfg)))

    def test_legacy_joint_statistics_cannot_be_used_for_pose_state(self):
        cfg = self.config()
        cfg.datasets.vla_data.pretrained_stats_path = '/old/joint/statistics.json'
        self.assertTrue(any('joint-state statistics' in issue for issue in check_training_config(cfg)))

    def test_model_representation_must_match_the_dataset(self):
        cfg = self.config()
        cfg.framework.conditioning.state_representation = 'joint'
        self.assertTrue(any('state_representation disagree' in issue for issue in check_training_config(cfg)))


if __name__ == '__main__':
    unittest.main()
