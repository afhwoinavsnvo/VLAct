"""Opt-in current TCP pose state must not pass through joint-state transforms."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset

from examples.Robotwin.train_files.prepare_clean_tcp_dataset import GOAL_POLICY
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag, ROBOT_TYPE_TO_EMBODIMENT_TAG
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor
from starVLA.dataloader.tcp_goal_dataset import TCPGoalDataset, TCP_POSE_STATE_LAYOUT, TCP_POSE_STATE_MIX


POSE_TYPE = "robotwin_tcp_state_wrap32"


def pose_metadata():
    return {
        "schema_version": 1, "state_representation": "tcp_pose", "feature": "observation.state",
        "source_feature": "observation.tcp_pose", "backup_joint_feature": "observation.joint_state",
        "current_frame": True, "normalization_applied": False,
        "dtype": "float32", "shape": [14], "coordinate_frame": "world",
        "position_unit": "meters", "quaternion_order": "wxyz", "layout": TCP_POSE_STATE_LAYOUT,
        "tcp_definition": "p_TCP=p_EE+R(q_wxyz)@[0.12,0,0]; q_TCP=q_EE",
    }


def statistical_values(width):
    return {"min": [-10.] * width, "max": [10.] * width, "mean": [0.] * width,
            "std": [1.] * width, "q01": [-9.] * width, "q99": [9.] * width}


def transform_metadata(config):
    keys = {"state": config.state_keys, "action": config.action_keys}
    stats, modalities = {}, {"video": {}}
    for modality, names in keys.items():
        stats[modality], modalities[modality] = {}, {}
        for key in names:
            width = 14 if key == "state.tcp_pose" else 1 if "gripper" in key else 6
            name = key.split(".", 1)[1]
            stats[modality][name] = statistical_values(width)
            modalities[modality][name] = {"absolute": True, "shape": [width], "continuous": True}
    return DatasetMetadata.model_validate({"statistics": stats, "modalities": modalities,
                                           "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT})


def numeric_transform(config, *, pose=False):
    full = config.transform({"state_representation": "tcp_pose"} if pose else None)
    transform = ComposedModalityTransform(transforms=[
        item for item in full.transforms if not any(key.startswith("video.") for key in item.apply_to)
    ])
    transform.set_metadata(transform_metadata(config))
    return transform


def packed(config, data, *, pose=False):
    sample_data = {**data, "annotation.human.action.task_description": ["same instruction"]}
    for key in config.video_keys:
        sample_data[key] = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    context = SimpleNamespace(
        data_cfg={"include_state": True, **({"state_representation": "tcp_pose"} if pose else {})},
        modality_keys={"state": config.state_keys, "action": config.action_keys,
                       "video": config.video_keys, "language": config.language_keys},
        robot_type="new_embodiment",
    )
    return LeRobotSingleDataset._pack_sample(context, sample_data)


class PoseFixtureDataset(Dataset):
    def __init__(self, root, sequence):
        self.root, self.sequence = root, sequence
        self.data_cfg = {"data_mix": TCP_POSE_STATE_MIX, "state_representation": "tcp_pose"}
        self.override = None
        self.include_state = True

    def __len__(self):
        return len(self.sequence) - 1

    def __getitem__(self, index):
        sample = {"image": [np.zeros((2, 2, 3), dtype=np.uint8)], "lang": "same instruction",
                "action": np.zeros((32, 14), dtype=np.float16),
                "subset_path": str(self.root / "fixture_task"), "dataset_name": "fixture_task",
                "trajectory_id": 0, "step_index": index}
        if self.include_state:
            sample["state"] = self.sequence[index:index + 1].copy() if self.override is None else self.override
        return sample

    def save_dataset_statistics(self, path, format="json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"pose_statistics_from_new_dataset": True}))


def make_fixture(root):
    task = root / "fixture_task"
    (task / "meta").mkdir(parents=True)
    state = pose_metadata()
    features = {name: {"dtype": "float32", "shape": [14]} for name in (
        "observation.state", "observation.tcp_pose", "observation.joint_state",
    )}
    # Converted pose-state datasets use flat names; original TCP features may
    # use a one-element outer list for the same ordered fourteen dimensions.
    features["observation.state"]["names"] = list(TCP_POSE_STATE_LAYOUT)
    (task / "meta/info.json").write_text(json.dumps({"chunks_size": 1000, "features": features}))
    (task / "meta/modality.json").write_text(json.dumps({"state": {
        "tcp_pose": {"start": 0, "end": 14, "original_key": "observation.state"},
    }}))
    (task / "meta/state_pose.json").write_text(json.dumps(state))
    tcp = {key: state[key] for key in ("coordinate_frame", "position_unit", "quaternion_order", "layout", "tcp_definition")}
    tcp.update(target_sampling=GOAL_POLICY,
               sidecar_template="tcp/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz")
    (task / "meta/tcp_pose.json").write_text(json.dumps(tcp))
    (root / "preparation_manifest.json").write_text(json.dumps({
        "status": "passed", "target_sampling": GOAL_POLICY,
        "state_representation": "tcp_pose", "state_contract": state,
        "tasks": [{"task": "fixture_task", "episodes": [{"public_episode_id": 0, "public_length": 120}]}],
    }))
    (root / "tcp_statistics.json").write_text(json.dumps({"raw_pose_statistics": True}))
    sequence = np.zeros((121, 14), dtype=np.float32)
    sequence[:, 0] = np.arange(121) * np.float32(.00013) + np.float32(10.1234567)
    sequence[:, 7] = np.arange(121) * np.float32(.00017) - np.float32(3.1234567)
    sequence[:, 3:7] = [.5, -.5, .5, -.5]
    sequence[:, 10:14] = [.5, .5, -.5, -.5]
    sidecar = task / "tcp/chunk-000/episode_000000.npz"
    sidecar.parent.mkdir(parents=True)
    metadata = {"task": "fixture_task", "public_episode_id": 0, "public_length": 120,
                "raw_length": 121, "target_sampling": GOAL_POLICY}
    np.savez_compressed(sidecar, tcp_pose_world_wxyz=sequence, metadata_json=np.array(json.dumps(metadata)))
    return PoseFixtureDataset(root, sequence)


class TCPPoseStateLoaderTests(unittest.TestCase):
    def test_registered_mix_preserves_fifty_tasks_weights_action_horizon_and_embodiment(self):
        old = DATASET_NAMED_MIXTURES["robotwin_wrap_32"]
        new = DATASET_NAMED_MIXTURES[TCP_POSE_STATE_MIX]
        self.assertEqual(len(new), 50)
        self.assertEqual([(name, weight) for name, weight, _ in old],
                         [(name, weight) for name, weight, _ in new])
        self.assertEqual({robot for _, _, robot in new}, {POSE_TYPE})
        config = ROBOT_TYPE_CONFIG_MAP[POSE_TYPE]
        self.assertEqual(config.state_keys, ["state.tcp_pose"])
        self.assertEqual(config.modality_config()["state"].delta_indices, [0])
        self.assertEqual(config.action_indices, list(range(32)))
        self.assertEqual(config.action_keys, ROBOT_TYPE_CONFIG_MAP["robotwin_wrap32"].action_keys)
        self.assertEqual(ROBOT_TYPE_TO_EMBODIMENT_TAG[POSE_TYPE], EmbodimentTag.NEW_EMBODIMENT)

    def test_new_transform_requires_explicit_pose_mode(self):
        config = ROBOT_TYPE_CONFIG_MAP[POSE_TYPE]
        for data_cfg in (None, {}, {"state_representation": "joint"}):
            with self.subTest(config=data_cfg), self.assertRaisesRegex(ValueError, "explicit"):
                config.transform(data_cfg)

    def test_pose_state_is_raw_float32_and_actions_exactly_match_legacy_wrap(self):
        old, new = ROBOT_TYPE_CONFIG_MAP["robotwin_wrap32"], ROBOT_TYPE_CONFIG_MAP[POSE_TYPE]
        rng = np.random.default_rng(51)
        action = {key: rng.uniform(-14, 14, (32, 1 if "gripper" in key else 6)).astype(np.float32)
                  for key in old.action_keys}
        action["action.left_gripper"] = np.tile([[.48], [.5]], (16, 1)).astype(np.float32)
        action["action.right_gripper"] = np.tile([[.7], [.1]], (16, 1)).astype(np.float32)
        old_state = {key: rng.uniform(-2, 2, (1, 1 if "gripper" in key else 6)).astype(np.float32)
                     for key in old.state_keys}
        pose = np.asarray([[10.1234567, -.3, .7, .5, -.5, .5, -.5,
                            -3.1234567, .2, -.9, .5, .5, -.5, -.5]], dtype=np.float32)
        old_output = numeric_transform(old)({**deepcopy(action), **old_state})
        new_transform = numeric_transform(new, pose=True)
        state_transforms = [item for item in new_transform.transforms if "state.tcp_pose" in item.apply_to]
        self.assertEqual(len(state_transforms), 1)
        self.assertIsInstance(state_transforms[0], StateActionToTensor)
        new_output = new_transform({**deepcopy(action), "state.tcp_pose": pose.copy()})
        np.testing.assert_array_equal(new_output["state.tcp_pose"].numpy(), pose)
        for key in old.action_keys:
            torch.testing.assert_close(new_output[key], old_output[key], rtol=0, atol=0)
        old_sample, new_sample = packed(old, old_output), packed(new, new_output, pose=True)
        self.assertEqual(old_sample["state"].dtype, np.float16)
        self.assertEqual(new_sample["state"].dtype, np.float32)
        np.testing.assert_array_equal(new_sample["state"], pose)
        np.testing.assert_array_equal(new_sample["action"], old_sample["action"])

    def test_current_state_and_future_target_align_independently_and_save_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            base = make_fixture(Path(folder) / "data")
            wrapper = TCPGoalDataset(base, base.root, seed=42, expected_target_schema_version=2)
            sample = wrapper[7]
            np.testing.assert_array_equal(sample["state"][0], base.sequence[7])
            self.assertTrue(71 <= sample["goal_frame"] < 103)
            np.testing.assert_array_equal(sample["target_tcp_pose"], base.sequence[sample["goal_frame"]])
            self.assertFalse(np.array_equal(sample["state"][0], sample["target_tcp_pose"]))
            saved = Path(folder) / "run/dataset_statistics.json"
            wrapper.save_dataset_statistics(saved)
            contract = json.loads((saved.parent / "target_tcp_contract.json").read_text())
            self.assertEqual(contract["state_representation"], "tcp_pose")
            self.assertEqual(contract["state_contract"], pose_metadata())
            self.assertIn("raw float32", contract["state"])

    def test_wrong_dtype_nonfinite_or_future_state_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = make_fixture(Path(folder) / "data")
            wrapper = TCPGoalDataset(base, base.root, seed=42)
            invalid = [base.sequence[7:8].astype(np.float16), base.sequence[8:9].copy(),
                       np.full((1, 14), np.nan, dtype=np.float32)]
            for state in invalid:
                base.override = state
                with self.subTest(dtype=state.dtype), self.assertRaises(ValueError):
                    wrapper[7]

    def test_flat_and_nested_names_preserve_exact_pose_layout_and_reject_reordering(self):
        with tempfile.TemporaryDirectory() as folder:
            base = make_fixture(Path(folder) / "data")
            path = base.root / "fixture_task/meta/info.json"
            info = json.loads(path.read_text())
            for names in (list(TCP_POSE_STATE_LAYOUT), [list(TCP_POSE_STATE_LAYOUT)]):
                with self.subTest(names=names):
                    info["features"]["observation.state"]["names"] = names
                    path.write_text(json.dumps(info))
                    sample = TCPGoalDataset(base, base.root)[7]
                    np.testing.assert_array_equal(sample["state"][0], base.sequence[7])
            for names in (list(reversed(TCP_POSE_STATE_LAYOUT)),
                          [list(reversed(TCP_POSE_STATE_LAYOUT))],
                          TCP_POSE_STATE_LAYOUT[:-1], [TCP_POSE_STATE_LAYOUT, TCP_POSE_STATE_LAYOUT]):
                with self.subTest(names=names):
                    info["features"]["observation.state"]["names"] = names
                    path.write_text(json.dumps(info))
                    with self.assertRaisesRegex(ValueError, "feature names"):
                        TCPGoalDataset(base, base.root)[7]

    def test_pose_root_and_task_contracts_must_match_and_joint_loader_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = make_fixture(Path(folder) / "data")
            base.data_cfg = {}
            with self.assertRaisesRegex(ValueError, "explicit"):
                TCPGoalDataset(base, base.root)
            base.data_cfg = {"data_mix": TCP_POSE_STATE_MIX, "state_representation": "tcp_pose"}
            path = base.root / "fixture_task/meta/state_pose.json"
            state = json.loads(path.read_text())
            state["normalization_applied"] = True
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                TCPGoalDataset(base, base.root)[0]

    def test_factory_preserves_pose_state_and_both_include_switches(self):
        from starVLA.dataloader import build_dataloader

        with tempfile.TemporaryDirectory() as folder:
            base = make_fixture(Path(folder) / "data")
            cfg = OmegaConf.create({"seed": 42, "output_dir": str(Path(folder) / "run"), "datasets": {"vla_data": {
                **base.data_cfg, "data_root_dir": str(base.root), "include_state": True, "include_tcp": True,
                "tcp_target_schema_version": 2, "per_device_batch_size": 2, "num_workers": 0,
            }}})
            for state_on in (False, True):
                for tcp_on in (False, True):
                    with self.subTest(state=state_on, tcp=tcp_on):
                        cfg.datasets.vla_data.include_state = state_on
                        cfg.datasets.vla_data.include_tcp = tcp_on
                        base.include_state = state_on
                        with patch("starVLA.dataloader.lerobot_datasets.get_vla_dataset", return_value=base):
                            loader = build_dataloader(cfg, dataset_py="lerobot_datasets")
                        batch = next(iter(loader))
                        self.assertEqual("state" in batch[1], state_on)
                        self.assertEqual("target_tcp_pose" in batch[1], tcp_on)
                        if tcp_on:
                            self.assertEqual(loader.dataset.state_representation, "tcp_pose")
                        if state_on:
                            np.testing.assert_array_equal(batch[1]["state"][0], base.sequence[1])
                            self.assertEqual(batch[1]["state"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
