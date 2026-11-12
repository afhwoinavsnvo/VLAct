"""Preserve the user's TCP dataset contract while separating state/TCP flags."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset

from examples.Robotwin.train_files.prepare_clean_tcp_dataset import GOAL_POLICY, RAW_LAST_GOAL_POLICY
from starVLA.dataloader.tcp_goal_dataset import TCPGoalDataset, resolve_include_tcp


class FixtureBaseDataset(Dataset):
    def __init__(self, root, *, include_state=False, length=120):
        self.root = Path(root)
        self.include_state = include_state
        self.length = length
        self.epoch = 0
        self.calls = []
        self.image = np.zeros((2, 2, 3), dtype=np.uint8)
        self.state = np.linspace(-1, 1, 14, dtype=np.float32)[None, :]
        self.action = np.zeros((32, 14), dtype=np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.calls.append(index)
        sample = {
            "image": [self.image], "lang": "fixture instruction", "action": self.action,
            "subset_path": str(self.root / "fixture_task"), "dataset_name": "fixture_task",
            "trajectory_id": 0, "step_index": int(index),
        }
        if self.include_state:
            sample["state"] = self.state
        return sample

    def set_epoch(self, epoch):
        self.epoch = epoch

    def save_dataset_statistics(self, path, format="json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"base_statistics_unchanged": True}))


def make_tcp_fixture(root, *, policy=GOAL_POLICY, length=120):
    root = Path(root)
    task = root / "fixture_task"
    (task / "meta").mkdir(parents=True)
    sidecar = task / "tcp/chunk-000/episode_000000.npz"
    sidecar.parent.mkdir(parents=True)
    (task / "meta/info.json").write_text(json.dumps({
        "chunks_size": 1000,
        "features": {"observation.tcp_pose": {"dtype": "float32", "shape": [14]}},
    }))
    (task / "meta/tcp_pose.json").write_text(json.dumps({
        "target_sampling": policy,
        "sidecar_template": "tcp/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz",
    }))
    (root / "preparation_manifest.json").write_text(json.dumps({
        "schema_version": 1, "status": "passed", "target_sampling": policy,
        "tasks": [{"task": "fixture_task", "episodes": [{"public_episode_id": 0, "public_length": length}]}],
    }))
    (root / "tcp_statistics.json").write_text(json.dumps({"fixture_statistics": True}))
    sequence = np.zeros((length + 1, 14), dtype=np.float32)
    sequence[:, 0] = np.arange(length + 1) / 1000
    sequence[:, 7] = np.arange(length + 1) / 1000 + 1
    sequence[:, [3, 10]] = 1
    metadata = {
        "task": "fixture_task", "public_episode_id": 0, "public_length": length,
        "raw_length": length + 1, "target_sampling": policy,
    }
    np.savez_compressed(sidecar, tcp_pose_world_wxyz=sequence, metadata_json=np.array(json.dumps(metadata)))
    return root, sequence


class TCPGoalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vlact-tcp-data-")
        self.root, self.sequence = make_tcp_fixture(Path(self.temp.name) / "data")

    def tearDown(self):
        self.temp.cleanup()

    def wrapper(self, *, include_state=False, base=None):
        if base is None:
            base = FixtureBaseDataset(self.root, include_state=include_state)
        return TCPGoalDataset(base, self.root, seed=42, include_state=include_state, expected_target_schema_version=2)

    def test_tcp_only_does_not_require_state_or_add_time(self):
        wrapper = self.wrapper()
        sample = wrapper[0]
        self.assertNotIn("state", sample)
        self.assertNotIn("time_budget", sample)
        self.assertEqual(sample["target_tcp_pose"].shape, (14,))
        self.assertEqual(sample["target_tcp_pose"].dtype, np.dtype("float32"))
        self.assertTrue(64 <= sample["goal_frame"] < 96)
        np.testing.assert_array_equal(sample["target_tcp_pose"], self.sequence[sample["goal_frame"]])
        self.assertIs(sample["action"], wrapper.base.action)
        self.assertIs(sample["image"][0], wrapper.base.image)
        self.assertEqual(sample["lang"], "fixture instruction")

    def test_state_on_preserves_original_joint_state_and_validates_shape(self):
        wrapper = self.wrapper(include_state=True)
        self.assertIs(wrapper[0]["state"], wrapper.base.state)
        wrapper.base.include_state = False
        with self.assertRaisesRegex(ValueError, "state"):
            wrapper[0]
        wrapper.base.include_state = True
        wrapper.base.state = np.zeros(14, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shaped \\[1,14\\]"):
            wrapper[0]

    def test_full_anchor_range_and_clean_last_fallback_are_unchanged(self):
        wrapper = self.wrapper()
        self.assertEqual(len(wrapper), 120)
        samples = [wrapper[index] for index in range(len(wrapper))]
        self.assertEqual(wrapper.base.calls, list(range(120)))
        self.assertEqual([sample["step_index"] for sample in samples], list(range(120)))
        self.assertEqual(samples[24]["used_terminal_fallback"], False)
        self.assertEqual(samples[25]["used_terminal_fallback"], True)
        self.assertEqual(samples[-1]["goal_frame"], 119)
        np.testing.assert_array_equal(samples[-1]["target_tcp_pose"], self.sequence[119])
        self.assertTrue(all(sample["action"] is wrapper.base.action for sample in samples))

    def test_explicit_aliases_require_booleans_and_cannot_disagree(self):
        for config, expected in (
            ({}, False), ({"include_tcp": True}, True), ({"include_tcp": False}, False),
            ({"use_target_tcp": True}, True), ({"include_tcp": True, "use_target_tcp": True}, True),
        ):
            self.assertEqual(resolve_include_tcp(config), expected)
        for name in ("include_tcp", "use_target_tcp"):
            for value in ("False", "True", 0, 1, None):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(TypeError, "must be a boolean"):
                    resolve_include_tcp({name: value})
        with self.assertRaisesRegex(ValueError, "explicitly disagree"):
            resolve_include_tcp({"include_tcp": False, "use_target_tcp": True})

    def test_expected_schema_locks_clean_last_and_keeps_legacy_rawlast_support(self):
        raw_root, raw_sequence = make_tcp_fixture(Path(self.temp.name) / "rawlast", policy=RAW_LAST_GOAL_POLICY)
        base = FixtureBaseDataset(raw_root)
        with self.assertRaisesRegex(ValueError, "schema must be 2"):
            TCPGoalDataset(base, raw_root, include_state=False, expected_target_schema_version=2)
        legacy = TCPGoalDataset(base, raw_root, include_state=False)
        sample = legacy[119]
        self.assertEqual(sample["goal_frame"], 120)
        np.testing.assert_array_equal(sample["target_tcp_pose"], raw_sequence[120])

    def test_goal_rng_is_independent_reproducible_and_updates_with_epoch(self):
        np.random.seed(78)
        before = deepcopy(np.random.get_state())
        first = self.wrapper()
        second = self.wrapper()
        goals = [first[0]["goal_frame"] for _ in range(20)]
        self.assertEqual(goals, [second[0]["goal_frame"] for _ in range(20)])
        self.assertGreater(len(set(goals)), 1)
        first.set_epoch(1)
        self.assertEqual(first.base.epoch, 1)
        self.assertNotEqual(goals, [first[0]["goal_frame"] for _ in range(20)])
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_saved_contract_records_state_off_without_changing_source_statistics(self):
        wrapper = self.wrapper()
        original = (self.root / "tcp_statistics.json").read_bytes()
        path = Path(self.temp.name) / "run/dataset_statistics.json"
        wrapper.save_dataset_statistics(path)
        contract = json.loads((path.parent / "target_tcp_contract.json").read_text())
        self.assertFalse(contract["include_state"])
        self.assertFalse(contract["normalization_applied_by_wrapper"])
        self.assertEqual(contract["target_sampling"], GOAL_POLICY)
        self.assertEqual((self.root / "tcp_statistics.json").read_bytes(), original)
        self.assertEqual((path.parent / "tcp_pose_statistics.json").read_bytes(), original)


class TCPDataLoaderFactoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vlact-tcp-factory-")
        self.root, _ = make_tcp_fixture(Path(self.temp.name) / "data")
        self.config = OmegaConf.create({
            "seed": 42, "output_dir": str(Path(self.temp.name) / "run"),
            "datasets": {"vla_data": {
                "data_root_dir": str(self.root), "include_state": False, "include_tcp": True,
                "include_time_budget": False, "tcp_target_schema_version": 2,
                "per_device_batch_size": 2, "num_workers": 0, "pin_memory": False,
            }},
        })

    def tearDown(self):
        self.temp.cleanup()

    def test_tcp_only_factory_runs_without_distributed_initialization(self):
        from starVLA.dataloader import build_dataloader

        base = FixtureBaseDataset(self.root)
        with patch("starVLA.dataloader.lerobot_datasets.get_vla_dataset", return_value=base), \
                patch("starVLA.dataloader.dist.is_initialized", return_value=False), \
                patch("starVLA.dataloader.dist.get_rank", side_effect=AssertionError("No process group")):
            loader = build_dataloader(self.config, dataset_py="lerobot_datasets")
        self.assertIsInstance(loader.dataset, TCPGoalDataset)
        self.assertFalse(loader.dataset.include_state)
        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.persistent_workers)
        batch = next(iter(loader))
        self.assertEqual(len(batch), 2)
        self.assertNotIn("state", batch[0])
        self.assertIn("target_tcp_pose", batch[0])
        self.assertNotIn("time_budget", batch[0])
        self.assertTrue((Path(self.config.output_dir) / "target_tcp_contract.json").is_file())

    def test_tcp_off_keeps_base_and_honors_pin_memory(self):
        from starVLA.dataloader import build_dataloader

        self.config.datasets.vla_data.include_tcp = False
        self.config.datasets.vla_data.pin_memory = True
        base = FixtureBaseDataset(self.root)
        with patch("starVLA.dataloader.lerobot_datasets.get_vla_dataset", return_value=base):
            loader = build_dataloader(self.config, dataset_py="lerobot_datasets")
        self.assertIs(loader.dataset, base)
        self.assertTrue(loader.pin_memory)
        self.assertFalse(loader.persistent_workers)

    def test_invalid_switches_fail_before_building_base_dataset(self):
        from starVLA.dataloader import build_dataloader

        for key, value in (
            ("include_tcp", "False"), ("include_state", "False"),
            ("include_time_budget", True), ("include_time_budget", "False"),
            ("use_target_tcp", False),
        ):
            with self.subTest(key=key, value=value):
                config = OmegaConf.create(OmegaConf.to_container(self.config))
                config.datasets.vla_data[key] = value
                with patch("starVLA.dataloader.lerobot_datasets.get_vla_dataset") as make_base:
                    with self.assertRaises((ValueError, TypeError)):
                        build_dataloader(config, dataset_py="lerobot_datasets")
                    make_base.assert_not_called()


if __name__ == "__main__":
    unittest.main()
