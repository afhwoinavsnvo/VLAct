"""Explicit XYZW datasets preserve both current state and sampled TCP order."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from deployment.model_server.pose_conventions import pose_layout
from starVLA.dataloader.tcp_goal_dataset import TCPGoalDataset
from test_tcp_pose_state_loader import make_fixture


PERMUTATION = [0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 11, 12, 13, 10]
TARGET_ARRAY = "tcp_pose_world_xyzw"
EE_ARRAY = "ee_pose_world_wxyz"
ARRAY_ORDERS = {EE_ARRAY: "wxyz", TARGET_ARRAY: "xyzw"}
TCP_DEFINITION = "p_TCP=p_EE+R(q_xyzw)@[0.12,0,0]; q_TCP=q_EE"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data))


def replace_sidecar(base, edit):
    path = base.root / "fixture_task/tcp/chunk-000/episode_000000.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays["metadata_json"]))
    edit(arrays, metadata)
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **arrays)


def make_xyzw_fixture(root):
    base = make_fixture(root)
    wxyz = base.sequence.copy()
    # Distinct quaternion components detect both missing and double reorder.
    wxyz[:, 3:7] = np.asarray([1., 2., 3., 4.], dtype=np.float32) / np.float32(np.sqrt(30.))
    wxyz[:, 10:14] = np.asarray([-4., 3., -2., 1.], dtype=np.float32) / np.float32(np.sqrt(30.))
    xyzw = wxyz[:, PERMUTATION].copy()
    base.sequence = xyzw
    base.data_cfg["tcp_quaternion_order"] = "xyzw"
    meta = root / "fixture_task/meta"
    state = read_json(meta / "state_pose.json")
    state.update(schema_version=2, quaternion_order="xyzw", layout=pose_layout("xyzw"),
                 tcp_definition=TCP_DEFINITION, target_pose_array=TARGET_ARRAY)
    write_json(meta / "state_pose.json", state)
    tcp = read_json(meta / "tcp_pose.json")
    tcp.update(schema_version=2, quaternion_order="xyzw", layout=pose_layout("xyzw"),
               tcp_definition=TCP_DEFINITION, target_pose_array=TARGET_ARRAY,
               array_quaternion_orders=ARRAY_ORDERS, provenance_arrays=[EE_ARRAY])
    write_json(meta / "tcp_pose.json", tcp)
    info = read_json(meta / "info.json")
    info["features"]["observation.state"]["names"] = pose_layout("xyzw")
    info["features"]["observation.tcp_pose"]["names"] = [pose_layout("xyzw")]
    write_json(meta / "info.json", info)
    manifest = read_json(root / "preparation_manifest.json")
    manifest.update(quaternion_order="xyzw", target_pose_array=TARGET_ARRAY,
                    array_quaternion_orders=ARRAY_ORDERS, state_contract=state)
    write_json(root / "preparation_manifest.json", manifest)

    def edit(arrays, metadata):
        del arrays["tcp_pose_world_wxyz"]
        arrays[TARGET_ARRAY] = xyzw.copy()
        arrays[EE_ARRAY] = wxyz.astype(np.float64)
        metadata.update(quaternion_order="xyzw", layout=pose_layout("xyzw"), tcp_definition=TCP_DEFINITION,
                        target_pose_array=TARGET_ARRAY, array_quaternion_orders=ARRAY_ORDERS,
                        provenance_arrays=[EE_ARRAY])
    replace_sidecar(base, edit)
    return base, wxyz


class TCPXYZWLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vlact-xyzw-loader-")
        self.addCleanup(self.temp.cleanup)

    def fixture(self, name="data"):
        return make_xyzw_fixture(Path(self.temp.name) / name)

    def test_xyzw_state_and_target_are_consumed_without_another_conversion(self):
        base, wxyz = self.fixture()
        wrapper = TCPGoalDataset(base, base.root, seed=42, expected_target_schema_version=2)
        sample = wrapper[7]
        self.assertEqual(wrapper.tcp_quaternion_order, "xyzw")
        self.assertEqual(wrapper.target_pose_array, TARGET_ARRAY)
        np.testing.assert_array_equal(sample["state"][0], wxyz[7, PERMUTATION])
        np.testing.assert_array_equal(sample["target_tcp_pose"], wxyz[sample["goal_frame"], PERMUTATION])
        self.assertEqual(sample["state"].dtype, np.float32)
        self.assertEqual(sample["target_tcp_pose"].dtype, np.float32)
        self.assertFalse(np.array_equal(sample["state"][0], wxyz[7]))

    def test_old_wxyz_default_and_target_sampling_policy_are_unchanged(self):
        legacy = make_fixture(Path(self.temp.name) / "legacy")
        xyzw, _ = self.fixture("xyzw")
        old = TCPGoalDataset(legacy, legacy.root, seed=18)
        new = TCPGoalDataset(xyzw, xyzw.root, seed=18)
        self.assertEqual(old.tcp_quaternion_order, "wxyz")
        self.assertEqual(old.target_pose_array, "tcp_pose_world_wxyz")
        for step in (0, 7, 24, 25, 119):
            before, after = old[step], new[step]
            self.assertEqual(before["goal_frame"], after["goal_frame"])
            self.assertEqual(before["used_terminal_fallback"], after["used_terminal_fallback"])
            np.testing.assert_array_equal(before["state"][0], legacy.sequence[step])
            np.testing.assert_array_equal(before["target_tcp_pose"], legacy.sequence[before["goal_frame"]])

    def test_config_and_root_order_must_agree(self):
        base, _ = self.fixture()
        del base.data_cfg["tcp_quaternion_order"]
        with self.assertRaisesRegex(ValueError, "root manifest"):
            TCPGoalDataset(base, base.root)
        base.data_cfg["tcp_quaternion_order"] = "xyzw"
        path = base.root / "preparation_manifest.json"
        manifest = read_json(path)
        del manifest["quaternion_order"]
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "root manifest"):
            TCPGoalDataset(base, base.root)

    def test_new_contract_schema_order_layout_and_array_are_strict(self):
        for index, (key, value) in enumerate((
            ("schema_version", 1), ("quaternion_order", "wxyz"),
            ("layout", pose_layout("wxyz")), ("target_pose_array", "tcp_pose_world_wxyz"),
        )):
            base, _ = self.fixture(str(index))
            path = base.root / "fixture_task/meta/tcp_pose.json"
            contract = read_json(path)
            contract[key] = value
            write_json(path, contract)
            with self.subTest(key=key), self.assertRaises(ValueError):
                TCPGoalDataset(base, base.root)[0]

    def test_wxyz_state_cannot_be_silently_reinterpreted_as_xyzw(self):
        base, wxyz = self.fixture()
        base.override = wxyz[7:8].copy()
        with self.assertRaisesRegex(ValueError, "same-step"):
            TCPGoalDataset(base, base.root)[7]

    def test_target_array_name_has_no_legacy_fallback(self):
        base, _ = self.fixture()

        def edit(arrays, _metadata):
            arrays["tcp_pose_world_wxyz"] = arrays.pop(TARGET_ARRAY)
        replace_sidecar(base, edit)
        with self.assertRaisesRegex(ValueError, "Cannot load"):
            TCPGoalDataset(base, base.root)[0]

    def test_sidecar_order_and_ee_provenance_are_checked_separately(self):
        for index, key in enumerate(("quaternion_order", "layout", "target_pose_array", "array_quaternion_orders", "missing_ee")):
            base, _ = self.fixture(str(index))

            def edit(arrays, metadata):
                if key == "quaternion_order":
                    metadata[key] = "wxyz"
                elif key == "layout":
                    metadata[key] = pose_layout("wxyz")
                elif key == "target_pose_array":
                    metadata[key] = "tcp_pose_world_wxyz"
                elif key == "array_quaternion_orders":
                    metadata[key] = {EE_ARRAY: "xyzw", TARGET_ARRAY: "xyzw"}
                else:
                    del arrays[EE_ARRAY]
            replace_sidecar(base, edit)
            with self.subTest(key=key), self.assertRaises(ValueError):
                TCPGoalDataset(base, base.root)[0]

    def test_feature_names_reject_wrong_quaternion_order(self):
        base, _ = self.fixture()
        path = base.root / "fixture_task/meta/info.json"
        info = read_json(path)
        info["features"]["observation.tcp_pose"]["names"] = pose_layout("wxyz")
        write_json(path, info)
        with self.assertRaisesRegex(ValueError, "feature names"):
            TCPGoalDataset(base, base.root)[0]

    def test_saved_contract_records_xyzw_without_altering_ee_provenance(self):
        base, wxyz = self.fixture()
        saved = Path(self.temp.name) / "run/dataset_statistics.json"
        wrapper = TCPGoalDataset(base, base.root)
        wrapper.save_dataset_statistics(saved)
        contract = read_json(saved.parent / "target_tcp_contract.json")
        self.assertEqual(contract["schema_version"], 2)
        self.assertEqual(contract["quaternion_order"], "xyzw")
        self.assertEqual(contract["target_pose_array"], TARGET_ARRAY)
        self.assertEqual(contract["array_quaternion_orders"], ARRAY_ORDERS)
        self.assertEqual(contract["state_contract"]["layout"], pose_layout("xyzw"))
        self.assertIn("world/meters/xyzw", contract["state"])
        self.assertIn("keep xyzw quaternions", contract["statistics_note"])
        sidecar = base.root / "fixture_task/tcp/chunk-000/episode_000000.npz"
        with np.load(sidecar, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays[EE_ARRAY], wxyz.astype(np.float64))


if __name__ == "__main__":
    unittest.main()
