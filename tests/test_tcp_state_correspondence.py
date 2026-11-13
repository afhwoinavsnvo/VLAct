"""Exercise the independent full-row dataset audit using tiny private fixtures."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from examples.Robotwin.train_files import check_tcp_state_correspondence as audit


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def statistics(values, *, episode=False):
    result = {"mean": values.mean(axis=0, dtype=np.float64).tolist(),
              "std": values.std(axis=0, dtype=np.float64).tolist(),
              "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}
    if episode:
        result["count"] = [len(values)]
    else:
        result.update(q01=np.quantile(values, .01, axis=0).tolist(),
                      q99=np.quantile(values, .99, axis=0).tolist())
    return result


def array_column(values):
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), 14)


def make_fixture(base):
    roots = [base / name for name in ("clean", "tcp", "pose")]
    task_name = "fixture_task"
    tasks = [root / task_name for root in roots]
    for task in tasks:
        (task / "meta").mkdir(parents=True)
        (task / "data/chunk-000").mkdir(parents=True)
    clean, source, target = tasks
    policy = {"schema_version": 2, "start_offset": 64, "end_offset_exclusive": 96,
              "full_window_required": True, "eligible_frame_scope": "clean_observation_frames",
              "fallback_frame": "n_clean - 1 = len(full_tcp_sequence) - 2"}
    tcp_contract = {"target_sampling": policy, "tcp_definition": "EE @ Tx(0.12 meters)",
                    "sidecar_template": "tcp/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz"}
    state_contract = {"schema_version": 1, "state_representation": "tcp_pose", "feature": audit.STATE,
                      "backup_joint_feature": audit.BACKUP, "current_frame": True,
                      "normalization_applied": False, "coordinate_frame": "world", "position_unit": "meters",
                      "quaternion_order": "wxyz", "layout": audit.LAYOUT,
                      "tcp_definition": tcp_contract["tcp_definition"], "dtype": "float32", "shape": [14]}
    feature = {"dtype": "float32", "shape": [14], "names": [[f"joint_{i}" for i in range(14)]]}
    pose_feature = {"dtype": "float32", "shape": [14], "names": [audit.LAYOUT]}
    info = {"chunks_size": 1000, "total_episodes": 2, "total_frames": 107, "total_tasks": 2, "fps": 15,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {audit.STATE: feature, "action": feature,
                         **{f"observation.images.{camera}": {"dtype": "video"}
                            for camera in ("high", "left", "right")}}}
    modality = {"state": {"left_joints": {"start": 0, "end": 6, "original_key": audit.STATE}},
                "action": {"left_joints": {"start": 0, "end": 6, "original_key": "action"}},
                "video": {"high": {"original_key": "observation.images.high"}},
                "annotation": {"task": {"original_key": "task_index"}}}
    episode_records, target_records = [], []
    episodic = [[], [], []]
    joint_values, action_values, tcp_values = [], [], []
    offset = 0
    for episode, length in enumerate((100, 7)):
        ee = np.zeros((length + 1, 14), dtype=np.float64)
        angle = np.arange(length + 1) * .003
        for start in (0, 7):
            ee[:, start] = np.arange(length + 1) * .001 + start / 10
            ee[:, start + 2] = .3
            ee[:, start + 3] = np.cos(angle / 2)
            ee[:, start + 6] = np.sin(angle / 2)
        tcp = ee.copy()
        for start in (0, 7):
            tcp[:, start] += .12 * np.cos(angle)
            tcp[:, start + 1] += .12 * np.sin(angle)
        tcp = tcp.astype(np.float32)
        joints = (np.arange(length * 14).reshape(length, 14) / 100).astype(np.float32)
        actions = (joints + .25).astype(np.float32)
        table = pa.table({audit.STATE: array_column(joints), "action": array_column(actions),
                          "timestamp": pa.array(np.arange(length) / 15, type=pa.float32()),
                          "frame_index": pa.array(np.arange(length), type=pa.int64()),
                          "episode_index": pa.array(np.full(length, episode), type=pa.int64()),
                          "index": pa.array(np.arange(offset, offset + length), type=pa.int64()),
                          "task_index": pa.array(np.full(length, episode), type=pa.int64())})
        table = table.replace_schema_metadata({b"huggingface": json.dumps({"info": {"features": {
            audit.STATE: {"dtype": "float32"}, "action": {"dtype": "float32"}}}}).encode()})
        tcp_table = table.append_column(pa.field(audit.TCP, pa.list_(pa.float32(), 14), nullable=False), array_column(tcp[:-1]))
        metadata = json.loads(table.schema.metadata[b"huggingface"])
        metadata["info"]["features"][audit.TCP] = {"dtype": "float32"}
        tcp_table = tcp_table.replace_schema_metadata({b"huggingface": json.dumps(metadata).encode()})
        pose_table = tcp_table.set_column(0, tcp_table.schema.field(audit.TCP).with_name(audit.STATE), tcp_table[audit.TCP])
        pose_table = pose_table.append_column(table.schema.field(audit.STATE).with_name(audit.BACKUP), table[audit.STATE])
        metadata["info"]["features"][audit.BACKUP] = metadata["info"]["features"][audit.STATE]
        pose_table = pose_table.replace_schema_metadata({b"huggingface": json.dumps(metadata).encode()})
        relative = Path(f"data/chunk-000/episode_{episode:06d}.parquet")
        paths = [task / relative for task in tasks]
        for path, value in zip(paths, (table, tcp_table, pose_table)):
            pq.write_table(value, path)
        sidecar = source / f"tcp/chunk-000/episode_{episode:06d}.npz"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar_metadata = {"task": task_name, "public_episode_id": episode, "public_length": length,
                            "raw_length": length + 1, "coordinate_frame": "world", "position_unit": "meters",
                            "quaternion_order": "wxyz", "layout": audit.LAYOUT, "target_sampling": policy}
        np.savez_compressed(sidecar, ee_pose_world_wxyz=ee, tcp_pose_world_wxyz=tcp,
                            metadata_json=np.array(json.dumps(sidecar_metadata)))
        target_sidecar = target / sidecar.relative_to(source)
        target_sidecar.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sidecar, target_sidecar)
        record = {"public_episode_id": episode, "public_length": length, "raw_episode_id": 37 + episode,
                  "raw_length": length + 1, "parquet": str(paths[1]), "parquet_sha256": audit.sha256(paths[1]),
                  "sidecar": str(sidecar), "sidecar_sha256": audit.sha256(sidecar),
                  "source_parquet_sha256": audit.sha256(paths[0]), "source_hdf5_sha256": "fixture-source-hash"}
        episode_records.append(record)
        target_records.append({**record, "parquet": str(paths[2]), "parquet_sha256": audit.sha256(paths[2]),
                               "source_tcp_parquet": str(paths[1]), "source_tcp_parquet_sha256": record["parquet_sha256"],
                               "sidecar": str(target_sidecar), "original_columns_and_types_preserved": False,
                               "original_non_state_columns_and_types_preserved": True,
                               "clean_joint_state_backed_up": True, "current_tcp_state_frame_aligned": True})
        base_stats = {audit.STATE: statistics(joints, episode=True), "action": statistics(actions, episode=True)}
        tcp_stats = {**base_stats, audit.TCP: statistics(tcp[:-1], episode=True)}
        pose_stats = {**tcp_stats, audit.STATE: tcp_stats[audit.TCP], audit.BACKUP: base_stats[audit.STATE]}
        for destination, value in zip(episodic, (base_stats, tcp_stats, pose_stats)):
            destination.append({"episode_index": episode, "stats": value})
        for camera in ("high", "left", "right"):
            video = clean / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"fixture video {camera} {episode}".encode())
        joint_values.append(joints)
        action_values.append(actions)
        tcp_values.append(tcp[:-1])
        offset += length
    base_stats = {audit.STATE: statistics(np.concatenate(joint_values)), "action": statistics(np.concatenate(action_values))}
    tcp_stats = {**base_stats, audit.TCP: statistics(np.concatenate(tcp_values))}
    pose_stats = {**tcp_stats, audit.STATE: tcp_stats[audit.TCP], audit.BACKUP: base_stats[audit.STATE]}
    source_info = deepcopy(info)
    source_info["features"][audit.TCP] = pose_feature
    target_info = deepcopy(source_info)
    target_info["features"][audit.STATE] = {**pose_feature, "names": audit.LAYOUT}
    target_info["features"][audit.BACKUP] = feature
    target_modality = deepcopy(modality)
    target_modality["state"] = {"tcp_pose": {"start": 0, "end": 14, "original_key": audit.STATE}}
    for index, task in enumerate(tasks):
        write_json(task / "meta/info.json", (info, source_info, target_info)[index])
        write_json(task / "meta/modality.json", (modality, modality, target_modality)[index])
        write_json(task / "meta/stats_gr00t.json", (base_stats, tcp_stats, pose_stats)[index])
        write_jsonl(task / "meta/episodes_stats.jsonl", episodic[index])
        write_jsonl(task / "meta/episodes.jsonl", [{"episode_index": ep, "length": n, "tasks": [lang]}
                                                  for ep, n, lang in ((0, 100, "lift bottle"), (1, 7, "place bottle"))])
        write_jsonl(task / "meta/tasks.jsonl", [{"task_index": 0, "task": "lift bottle"},
                                               {"task_index": 1, "task": "place bottle"}])
        if index:
            (task / "videos").symlink_to(clean / "videos", target_is_directory=True)
            write_json(task / "meta/tcp_pose.json", tcp_contract)
    write_json(target / "meta/state_pose.json", state_contract)
    for root, records in zip(roots[1:], (episode_records, target_records)):
        manifest = {"status": "passed", "task_count": 1, "episode_count": 2, "public_frame_count": 107,
                    "target_sampling": policy, "tasks": [{"task": task_name, "episode_count": 2,
                        "public_frames": 107, "full_raw_frames": 109, "episodes": records}]}
        if root == roots[2]:
            manifest.update(state_representation="tcp_pose", state_contract=state_contract, source_modified=False,
                            source_tcp_root=str(roots[1]), clean_root=str(roots[0]),
                            source_tcp_manifest_sha256=audit.sha256(roots[1] / "preparation_manifest.json"))
            manifest["tasks"][0]["state_representation"] = "tcp_pose"
        write_json(root / "preparation_manifest.json", manifest)
        write_json(root / "tcp_statistics.json", {"fixture": "unchanged future diagnostics"})
    return roots


class TCPStateCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vlact-tcp-state-audit-")
        self.addCleanup(self.temporary.cleanup)
        self.roots = make_fixture(Path(self.temporary.name))
        self.target = self.roots[2] / "fixture_task"

    def validate(self):
        return audit.validate_correspondence(*self.roots, expected_tasks=1,
                                             expected_episodes_per_task=2, expected_frames=107)

    def mutate_column(self, name):
        path = self.target / "data/chunk-000/episode_000000.parquet"
        table = pq.read_table(path)
        values = audit.pose_array(table, name, "fixture").copy()
        values[-1, 0] += 1
        table = table.set_column(table.column_names.index(name), table.schema.field(name), array_column(values))
        pq.write_table(table, path)

    def test_all_rows_and_terminal_geometry_match_without_decoding_video(self):
        report = self.validate()
        self.assertEqual(report["status"], "passed")
        self.assertEqual((report["row_count"], report["episode_count"], report["video_count"]), (107, 2, 6))
        self.assertEqual((report["full_window_anchors"], report["clean_last_fallback_anchors"]), (5, 102))
        self.assertLessEqual(report["max_geometry_error_m"], 1e-7)

    def test_checks_late_current_pose_rows_and_joint_backup(self):
        for name, message in ((audit.STATE, "same-frame"), (audit.BACKUP, "backup joints")):
            with self.subTest(column=name):
                path = self.target / "data/chunk-000/episode_000000.parquet"
                original = path.read_bytes()
                self.mutate_column(name)
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()
                path.write_bytes(original)

    def test_action_targets_cannot_change(self):
        self.mutate_column("action")
        with self.assertRaisesRegex(ValueError, "row values or nulls changed for action"):
            self.validate()

    def test_language_metadata_cannot_change(self):
        path = self.target / "meta/tasks.jsonl"
        path.write_text(path.read_text().replace("lift bottle", "wrong instruction"))
        with self.assertRaisesRegex(ValueError, "metadata file changed: tasks.jsonl"):
            self.validate()

    def test_source_notebook_backups_need_not_be_copied(self):
        source_meta = self.roots[1] / "fixture_task/meta"
        for original, backup in (("info.json", "info-checkpoint.json"),
                                 ("modality.json", "modality-checkpoint.json")):
            path = source_meta / ".ipynb_checkpoints" / backup
            path.parent.mkdir(exist_ok=True)
            shutil.copy2(source_meta / original, path)
        self.assertEqual(self.validate()["status"], "passed")
        # A canonical metadata change remains an error even with ignored backups.
        info_path = self.target / "meta/info.json"
        info = audit.read_json(info_path)
        info["fps"] = 30
        write_json(info_path, info)
        with self.assertRaisesRegex(ValueError, "target info changed beyond state/backup"):
            self.validate()

    def test_equal_video_bytes_are_insufficient_when_not_the_same_source_file(self):
        (self.target / "videos").unlink()
        shutil.copytree(self.roots[0] / "fixture_task/videos", self.target / "videos")
        with self.assertRaisesRegex(ValueError, "video not shared"):
            self.validate()

    def test_future_sidecar_bytes_cannot_change(self):
        path = self.target / "tcp/chunk-000/episode_000000.npz"
        with path.open("ab") as stream:
            stream.write(b"unexpected bytes")
        with self.assertRaisesRegex(ValueError, "sidecar bytes/SHA changed"):
            self.validate()

    def test_nonstate_statistics_cannot_change(self):
        path = self.target / "meta/stats_gr00t.json"
        value = audit.read_json(path)
        value["action"]["mean"][0] += 1
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "non-state aggregate statistics"):
            self.validate()

    def test_recomputes_pose_statistics_instead_of_only_comparing_metadata(self):
        for root in self.roots[1:]:
            path = root / "fixture_task/meta/stats_gr00t.json"
            value = audit.read_json(path)
            value[audit.TCP]["mean"][0] += 1
            if root == self.roots[2]:
                value[audit.STATE]["mean"][0] += 1
            write_json(path, value)
        with self.assertRaisesRegex(ValueError, "incorrect mean statistics"):
            self.validate()

    def test_default_api_rejects_partial_release(self):
        with self.assertRaisesRegex(ValueError, "task sets/counts"):
            audit.validate_correspondence(*self.roots)

    def test_cli_writes_failed_report_and_returns_one(self):
        report = Path(self.temporary.name) / "audit.json"
        code = audit.main(["--clean-root", str(self.roots[0]), "--source-tcp-root", str(self.roots[1]),
                           "--pose-state-root", str(self.roots[2]), "--report", str(report)])
        self.assertEqual(code, 1)
        self.assertEqual(audit.read_json(report)["status"], "failed")

    def test_cli_refuses_report_inside_dataset(self):
        report = self.roots[2] / "audit.json"
        code = audit.main(["--clean-root", str(self.roots[0]), "--source-tcp-root", str(self.roots[1]),
                           "--pose-state-root", str(self.roots[2]), "--report", str(report)])
        self.assertEqual(code, 1)
        self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
