"""Independent XYZW audit fixtures; never import the production converter."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from examples.Robotwin.train_files import check_tcp_xyzw_correspondence as audit


_spec = importlib.util.spec_from_file_location("_wxyz_audit_fixture", Path(__file__).with_name("test_tcp_state_correspondence.py"))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
write_json, statistics = _fixture.write_json, _fixture.statistics


def xyzw(values):
    """Build expected data by arm/block slicing, without the checker's indices."""
    return np.concatenate([values[:, :3], values[:, 4:7], values[:, 3:4],
                           values[:, 7:10], values[:, 11:14], values[:, 10:11]], axis=1)


def reordered_stats(record):
    result = deepcopy(record)
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        if key in result:
            result[key] = xyzw(np.asarray(result[key])[None, :])[0].tolist()
    return result


def make_fixture(base):
    clean, _tcp, source = _fixture.make_fixture(base)
    target = base / "xyzw"
    target.mkdir()
    source_manifest = audit.read_json(source / "preparation_manifest.json")
    source_manifest.update(full_raw_frame_count=109, source_manifest_unchanged=True,
                           original_non_state_columns_preserved=True, training_ready_claimed=False)
    source_task = source / "fixture_task"
    contract = audit.read_json(source_task / "meta/tcp_pose.json")
    contract.update(schema_version=1, feature=audit.TCP, coordinate_frame="world", position_unit="meters",
                    quaternion_order="wxyz", layout=audit.common.LAYOUT, normalization_applied=False,
                    sidecar_arrays={audit.EE: "float64 [public_length + 1, 14]",
                                    audit.OLD_TCP: "float32 [public_length + 1, 14]"})
    write_json(source_task / "meta/tcp_pose.json", contract)
    write_json(source / "preparation_manifest.json", source_manifest)
    task = target / "fixture_task"
    (task / "meta").mkdir(parents=True)
    for path in (source_task / "meta").iterdir():
        shutil.copy2(path, task / "meta" / path.name)
    (task / "videos").symlink_to(clean / "fixture_task/videos", target_is_directory=True)
    target_manifest = deepcopy(source_manifest)
    target_manifest.update(source_root=str(source), source_pose_state_root=str(source), output_root=str(target),
                           source_pose_state_manifest_sha256=audit.sha256(source / "preparation_manifest.json"),
                           quaternion_order="xyzw", target_pose_array=audit.NEW_TCP,
                           array_quaternion_orders=audit.ARRAY_ORDERS, original_non_state_columns_preserved=False,
                           original_non_pose_columns_preserved=True, changed_columns=audit.POSE_COLUMNS,
                           orientation_migration={"schema_version": 1, "source_order": "wxyz", "target_order": "xyzw",
                               "permutation": [0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 11, 12, 13, 10],
                               "geometry_recomputed": False, "quaternion_normalization_applied": False,
                               "quaternion_sign_changed": False,
                               "source_tcp_definition": source_manifest["state_contract"]["tcp_definition"]})
    for name in ("state_pose.json", "tcp_pose.json"):
        value = audit.read_json(source_task / "meta" / name)
        value.update(schema_version=2, quaternion_order="xyzw", layout=audit.LAYOUT,
                     tcp_definition=audit.DEFINITION, target_pose_array=audit.NEW_TCP)
        if name == "tcp_pose.json":
            value["sidecar_arrays"][audit.NEW_TCP] = value["sidecar_arrays"].pop(audit.OLD_TCP)
            value.update(array_quaternion_orders=audit.ARRAY_ORDERS, provenance_arrays=[audit.EE])
        else:
            target_manifest["state_contract"] = value
        write_json(task / "meta" / name, value)
    info = audit.read_json(source_task / "meta/info.json")
    for column in audit.POSE_COLUMNS:
        info["features"][column]["names"] = audit.LAYOUT
    write_json(task / "meta/info.json", info)
    all_full_poses = []
    for record in target_manifest["tasks"][0]["episodes"]:
        eid = record["public_episode_id"]
        relative = Path(f"data/chunk-000/episode_{eid:06d}.parquet")
        old_path, new_path = source_task / relative, task / relative
        table = pq.read_table(old_path)
        for column in audit.POSE_COLUMNS:
            values = xyzw(audit.common.pose_array(table, column, "fixture"))
            array = _fixture.array_column(values)
            table = table.set_column(table.column_names.index(column), table.schema.field(column), array)
        metadata = dict(table.schema.metadata)
        metadata[b"tcp_pose_contract"] = json.dumps({"schema_version": 2, "features": audit.POSE_COLUMNS,
            "quaternion_order": "xyzw", "layout": audit.LAYOUT, "coordinate_frame": "world",
            "position_unit": "meters", "normalization_applied": False}).encode()
        table = table.replace_schema_metadata(metadata)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, new_path)
        relative_sidecar = Path(f"tcp/chunk-000/episode_{eid:06d}.npz")
        old_sidecar, new_sidecar = source_task / relative_sidecar, task / relative_sidecar
        with np.load(old_sidecar, allow_pickle=False) as arrays:
            ee, old_values = arrays[audit.EE], arrays[audit.OLD_TCP]
            details = json.loads(str(arrays["metadata_json"]))
        all_full_poses.append(old_values)
        details.update(quaternion_order="xyzw", layout=audit.LAYOUT, tcp_definition=audit.DEFINITION,
                       target_pose_array=audit.NEW_TCP, array_quaternion_orders=audit.ARRAY_ORDERS,
                       provenance_arrays=[audit.EE], source_pose_state_sidecar=str(old_sidecar),
                       source_pose_state_sidecar_sha256=audit.sha256(old_sidecar))
        new_sidecar.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(new_sidecar, **{audit.EE: ee, audit.NEW_TCP: xyzw(old_values),
                                          "metadata_json": np.array(json.dumps(details))})
        record.update(parquet=str(new_path), parquet_sha256=audit.sha256(new_path),
                      sidecar=str(new_sidecar), sidecar_sha256=audit.sha256(new_sidecar),
                      source_pose_state_parquet=str(old_path), source_pose_state_parquet_sha256=audit.sha256(old_path),
                      source_pose_state_sidecar=str(old_sidecar), source_pose_state_sidecar_sha256=audit.sha256(old_sidecar),
                      original_non_state_columns_and_types_preserved=False,
                      original_non_pose_columns_and_types_preserved=True,
                      changed_columns=audit.POSE_COLUMNS, quaternion_order="xyzw")
    target_manifest["tasks"][0]["quaternion_order"] = "xyzw"
    for filename in ("stats_gr00t.json", "episodes_stats.jsonl"):
        path = task / "meta" / filename
        records = [audit.read_json(path)] if filename.endswith(".json") else [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:
            values = record if filename.endswith(".json") else record["stats"]
            for column in audit.POSE_COLUMNS:
                values[column] = reordered_stats(values[column])
        if filename.endswith(".json"):
            write_json(path, records[0])
        else:
            _fixture.write_jsonl(path, records)
    diagnostics = {"scope": "Full RAW diagnostics", "frame_count": 109, "layout": audit.common.LAYOUT,
                   "normalization_applied": False, "stats": statistics(np.concatenate(all_full_poses))}
    write_json(source / "tcp_statistics.json", diagnostics)
    diagnostics.update(layout=audit.LAYOUT, quaternion_order="xyzw", target_pose_array=audit.NEW_TCP,
                       stats=reordered_stats(diagnostics["stats"]))
    write_json(target / "tcp_statistics.json", diagnostics)
    write_json(target / "preparation_manifest.json", target_manifest)
    return clean, source, target


class TCPXYZWCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vlact-xyzw-audit-")
        self.addCleanup(self.temporary.cleanup)
        self.roots = make_fixture(Path(self.temporary.name))
        self.target = self.roots[2] / "fixture_task"

    def validate(self):
        return audit.validate_correspondence(*self.roots, expected_tasks=1,
                                             expected_episodes_per_task=2, expected_frames=107)

    def test_full_row_roundtrip_geometry_statistics_and_clean_correspondence(self):
        report = self.validate()
        self.assertEqual((report["row_count"], report["full_raw_frame_count"], report["video_count"]), (107, 109, 6))
        self.assertEqual((report["full_window_anchors"], report["clean_last_fallback_anchors"]), (5, 102))
        self.assertLessEqual(report["max_geometry_error_m"], 1e-7)

    def test_literal_inverse_and_asymmetric_components_are_independent(self):
        values = np.arange(28, dtype=np.float32).reshape(2, 14)
        expected = xyzw(values)
        audit.check_permutation(values, expected, "asymmetric")
        np.testing.assert_array_equal(expected[:, audit.INVERSE], values)
        with self.assertRaisesRegex(ValueError, "exact WXYZ-to-XYZW"):
            audit.check_permutation(values, xyzw(expected), "double conversion")

    def test_current_only_or_future_only_migration_is_rejected(self):
        path = self.target / "data/chunk-000/episode_000000.parquet"
        original = path.read_bytes()
        old = pq.read_table(self.roots[1] / "fixture_task/data/chunk-000/episode_000000.parquet")
        for column in audit.POSE_COLUMNS:
            with self.subTest(unconverted=column):
                table = pq.read_table(path)
                table = table.set_column(table.column_names.index(column), table.schema.field(column), old[column])
                pq.write_table(table, path)
                with self.assertRaisesRegex(ValueError, "exact WXYZ-to-XYZW"):
                    self.validate()
                path.write_bytes(original)

    def test_joint_action_and_backup_stay_unchanged(self):
        path = self.target / "data/chunk-000/episode_000000.parquet"
        table = pq.read_table(path)
        values = audit.common.pose_array(table, "action", "fixture").copy()
        values[-1, 13] += 1
        table = table.set_column(table.column_names.index("action"), table.schema.field("action"), _fixture.array_column(values))
        pq.write_table(table, path)
        with self.assertRaisesRegex(ValueError, "row values or nulls changed for action"):
            self.validate()

    def test_statistics_require_exact_permutation(self):
        path = self.target / "meta/stats_gr00t.json"
        stats = audit.read_json(path)
        stats[audit.STATE]["mean"][3] += 1
        write_json(path, stats)
        with self.assertRaisesRegex(ValueError, "mean is not an exact"):
            self.validate()

    def test_original_ee_provenance_cannot_change_even_if_hashes_are_updated(self):
        path = self.target / "tcp/chunk-000/episode_000000.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        arrays[audit.EE][-1, 0] += .1
        np.savez_compressed(path, **arrays)
        manifest_path = self.roots[2] / "preparation_manifest.json"
        manifest = audit.read_json(manifest_path)
        manifest["tasks"][0]["episodes"][0]["sidecar_sha256"] = audit.sha256(path)
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "EE provenance"):
            self.validate()

    def test_raw_terminal_future_frame_must_also_be_permuted(self):
        path = self.target / "tcp/chunk-000/episode_000000.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        arrays[audit.NEW_TCP][-1, 3:7] = arrays[audit.EE][-1, 3:7]
        np.savez_compressed(path, **arrays)
        manifest_path = self.roots[2] / "preparation_manifest.json"
        manifest = audit.read_json(manifest_path)
        manifest["tasks"][0]["episodes"][0]["sidecar_sha256"] = audit.sha256(path)
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "full future TCP sequence: pose is not"):
            self.validate()

    def test_nonpose_metadata_and_video_sharing_are_strict(self):
        (self.target / "videos").unlink()
        shutil.copytree(self.roots[0] / "fixture_task/videos", self.target / "videos")
        with self.assertRaisesRegex(ValueError, "video no longer shares"):
            self.validate()

    def test_source_manifest_sha_is_required(self):
        path = self.roots[2] / "preparation_manifest.json"
        manifest = audit.read_json(path)
        manifest["source_pose_state_manifest_sha256"] = "wrong"
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "Source WXYZ manifest SHA"):
            self.validate()

    def test_default_requires_full_release_and_cli_failed_report_is_safe(self):
        with self.assertRaisesRegex(ValueError, "Task IDs/counts"):
            audit.validate_correspondence(*self.roots)
        report = Path(self.temporary.name) / "audit.json"
        arguments = ["--clean-root", str(self.roots[0]), "--source-pose-state-root", str(self.roots[1]),
                     "--xyzw-root", str(self.roots[2]), "--report", str(report)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(arguments), 1)
        self.assertEqual(audit.read_json(report)["status"], "failed")
        arguments[-1] = str(self.roots[2] / "forbidden.json")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(audit.main(arguments), 1)
        self.assertFalse((self.roots[2] / "forbidden.json").exists())


if __name__ == "__main__":
    unittest.main()
