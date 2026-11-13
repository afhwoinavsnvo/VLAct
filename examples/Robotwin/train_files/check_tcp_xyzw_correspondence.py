#!/usr/bin/env python3
"""Audit every XYZW pose against its WXYZ source and original Clean episode.

This checker deliberately does not import the migration code or its permutation.
Videos are checked by file identity, without decoding. Only --report is written.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq

# Support both direct script execution and repository package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.Robotwin.train_files import check_tcp_state_correspondence as common


STATE, TCP, JOINT = "observation.state", "observation.tcp_pose", "observation.joint_state"
OLD_TCP, NEW_TCP, EE = "tcp_pose_world_wxyz", "tcp_pose_world_xyzw", "ee_pose_world_wxyz"
P = [0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 11, 12, 13, 10]
INVERSE = [0, 1, 2, 6, 3, 4, 5, 7, 8, 9, 13, 10, 11, 12]
LAYOUT = [f"{arm}_{part}" for arm in ("left", "right")
          for part in ("x", "y", "z", "qx", "qy", "qz", "qw")]
ARRAY_ORDERS = {EE: "wxyz", NEW_TCP: "xyzw"}
DEFINITION = "p_TCP=p_EE+R(q_xyzw)@[0.12,0,0]; q_TCP=q_EE"
POSE_COLUMNS = [STATE, TCP]
require, read_json, sha256 = common.require, common.read_json, common.sha256
without = common.without_keys


def equal_bits(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def check_permutation(source, target, context):
    require(source.dtype == target.dtype == np.float32 and source.shape == target.shape
            and source.ndim == 2 and source.shape[1] == 14
            and np.isfinite(source).all() and np.isfinite(target).all(), f"{context}: invalid float32 pose arrays")
    require(equal_bits(target, source[:, P]), f"{context}: pose is not the exact WXYZ-to-XYZW permutation")
    require(equal_bits(target[:, INVERSE], source), f"{context}: permutation is not exactly reversible")
    require(equal_bits(target[:, [0, 1, 2, 7, 8, 9]], source[:, [0, 1, 2, 7, 8, 9]]),
            f"{context}: physical position changed")


def check_statistics(source, target, context):
    require(set(source) == set(target), f"{context}: statistic keys changed")
    for key, value in source.items():
        if key in ("mean", "std", "min", "max", "q01", "q99"):
            old, new = np.asarray(value), np.asarray(target[key])
            require(old.shape == new.shape == (14,) and np.isfinite(old).all()
                    and np.isfinite(new).all() and equal_bits(new, old[P]),
                    f"{context}: {key} is not an exact component permutation")
        else:
            require(target[key] == value, f"{context}: non-component statistic {key} changed")


def check_stats_mapping(source, target, context):
    require(without(source, set(POSE_COLUMNS)) == without(target, set(POSE_COLUMNS)),
            f"{context}: action/joint backup/other statistics changed")
    for column in POSE_COLUMNS:
        check_statistics(source[column], target[column], f"{context}/{column}")


def check_contract(source, target, *, state, context):
    changed = {"schema_version", "quaternion_order", "layout", "tcp_definition", "target_pose_array"}
    if not state:
        changed |= {"sidecar_arrays", "array_quaternion_orders", "provenance_arrays"}
    require(source.get("schema_version") == 1 and source.get("quaternion_order") == "wxyz"
            and source.get("layout") == common.LAYOUT,
            f"{context}: source must explicitly declare WXYZ")
    require(without(source, changed) == without(target, changed), f"{context}: non-orientation contract changed")
    for key, value in {"schema_version": 2, "quaternion_order": "xyzw", "layout": LAYOUT,
                       "coordinate_frame": "world", "position_unit": "meters",
                       "tcp_definition": DEFINITION, "target_pose_array": NEW_TCP}.items():
        require(type(target.get(key)) is type(value) and target[key] == value, f"{context}: invalid {key}")
    require(target.get("normalization_applied") is False, f"{context}: pose must remain raw")
    if state:
        require(target.get("current_frame") is True and target.get("feature") == STATE
                and target.get("backup_joint_feature") == JOINT, f"{context}: current-state contract changed")
    else:
        require(target.get("array_quaternion_orders") == ARRAY_ORDERS
                and target.get("provenance_arrays") == [EE], f"{context}: EE/TCP array order declarations differ")
        arrays = deepcopy(source["sidecar_arrays"])
        arrays[NEW_TCP] = arrays.pop(OLD_TCP)
        require(target.get("sidecar_arrays") == arrays, f"{context}: sidecar array schema changed beyond TCP key")


def check_geometry(ee, tcp, context):
    require(ee.dtype == np.float64 and ee.shape == tcp.shape and np.isfinite(ee).all(), f"{context}: invalid EE provenance")
    maximum = 0.0
    for start in (0, 7):
        original_q = ee[:, start + 3:start + 7]
        norm = np.linalg.norm(original_q, axis=1)
        require(np.allclose(norm, 1, rtol=0, atol=1e-5), f"{context}: source EE quaternion is not unit")
        xyzw = tcp[:, start + 3:start + 7]
        require(equal_bits(xyzw, original_q[:, [1, 2, 3, 0]].astype(np.float32)),
                f"{context}: XYZW quaternion does not preserve physical EE orientation/sign")
        require(np.allclose(np.linalg.norm(xyzw, axis=1), 1, rtol=0, atol=1e-5), f"{context}: nonunit XYZW quaternion")
        q = original_q / norm[:, None]
        axis = np.broadcast_to([0.12, 0., 0.], (len(tcp), 3))
        cross = 2 * np.cross(q[:, 1:], axis)
        rotated = axis + q[:, :1] * cross + np.cross(q[:, 1:], cross)
        error = float(np.max(np.abs(tcp[:, start:start + 3] - ee[:, start:start + 3] - rotated)))
        require(error <= 1e-7, f"{context}: physical TCP geometry changed ({error} m)")
        maximum = max(maximum, error)
    return maximum


def check_task_metadata(source, target, context):
    infos = [read_json(root / "meta/info.json") for root in (source, target)]
    restored = deepcopy(infos[1])
    for column in POSE_COLUMNS:
        feature = restored["features"][column]
        require(feature.get("names") == LAYOUT and feature.get("dtype") == "float32" and feature.get("shape") == [14],
                f"{context}: {column} feature must declare raw XYZW float32[14]")
        feature["names"] = infos[0]["features"][column]["names"]
    require(restored == infos[0], f"{context}: non-orientation info metadata changed")
    files = [common.metadata_files(root / "meta") for root in (source, target)]
    require(set(files[0]) == set(files[1]), f"{context}: metadata file inventory changed")
    changed = {"info.json", "state_pose.json", "tcp_pose.json", "stats_gr00t.json", "episodes_stats.jsonl"}
    for name in set(files[0]) - changed:
        require(sha256(files[0][name]) == sha256(files[1][name]), f"{context}: non-pose metadata changed: {name}")
    contracts = {name: [read_json(root / "meta" / name) for root in (source, target)]
                 for name in ("tcp_pose.json", "state_pose.json")}
    for name, pair in contracts.items():
        check_contract(*pair, state=name == "state_pose.json", context=f"{context}/{name}")
    return infos, contracts


def validate_correspondence(clean_root, source_pose_state_root, xyzw_root, *, expected_tasks=50,
                            expected_episodes_per_task=50, expected_frames=549787):
    started = time.perf_counter()
    roots = [Path(root).expanduser().resolve(strict=True) for root in (clean_root, source_pose_state_root, xyzw_root)]
    require(len(set(roots)) == 3, "Dataset roots must be distinct")
    clean_root, source_root, target_root = roots
    manifests = [read_json(root / "preparation_manifest.json") for root in (source_root, target_root)]
    old, new = manifests
    require(old.get("status") == new.get("status") == "passed", "Both dataset manifests must have passed")
    require(new.get("source_pose_state_manifest_sha256") == sha256(source_root / "preparation_manifest.json"),
            "Source WXYZ manifest SHA does not match provenance")
    require(Path(new.get("source_pose_state_root", "")).resolve() == source_root
            and Path(new.get("clean_root", "")).resolve() == clean_root, "Manifest source/Clean root mismatch")
    require(new.get("quaternion_order") == "xyzw" and new.get("target_pose_array") == NEW_TCP
            and new.get("array_quaternion_orders") == ARRAY_ORDERS, "Manifest canonical XYZW array/order mismatch")
    require(new.get("source_modified") is False and new.get("source_manifest_unchanged") is True
            and new.get("original_non_state_columns_preserved") is False
            and new.get("original_non_pose_columns_preserved") is True
            and new.get("changed_columns") == POSE_COLUMNS, "Manifest preservation flags are incorrect")
    migration = new.get("orientation_migration", {})
    require(migration.get("source_order") == "wxyz" and migration.get("target_order") == "xyzw"
            and migration.get("schema_version") == 1 and migration.get("permutation") == P
            and migration.get("source_tcp_definition") == old["state_contract"]["tcp_definition"]
            and all(migration.get(key) is False for key in (
                "geometry_recomputed", "quaternion_normalization_applied", "quaternion_sign_changed")),
            "Orientation migration must declare only the exact reversible component permutation")
    changed_manifest = {"created_at_utc", "source_root", "source_pose_state_root", "output_root",
                        "source_pose_state_manifest_sha256", "state_contract", "quaternion_order", "target_pose_array",
                        "array_quaternion_orders", "original_non_state_columns_preserved", "original_non_pose_columns_preserved",
                        "changed_columns", "orientation_migration", "tasks", "elapsed_seconds"}
    require(without(old, changed_manifest) == without(new, changed_manifest), "Non-orientation manifest provenance changed")
    policy = old["target_sampling"]
    require(new["target_sampling"] == policy and policy.get("schema_version") == 2
            and policy.get("start_offset") == 64 and policy.get("end_offset_exclusive") == 96
            and policy.get("full_window_required") is True
            and policy.get("fallback_frame") == "n_clean - 1 = len(full_tcp_sequence) - 2",
            "Future goal policy must remain unchanged Clean-last schema 2")
    task_maps = [{row["task"]: row for row in manifest["tasks"]} for manifest in manifests]
    require(all(len(mapping) == len(manifest["tasks"]) == expected_tasks for mapping, manifest in zip(task_maps, manifests))
            and set(task_maps[0]) == set(task_maps[1]), "Task IDs/counts differ")
    for root in roots:
        require({path.name for path in root.iterdir() if path.is_dir() and (path / "data").is_dir()} == set(task_maps[0]),
                f"{root.name}: task folder inventory differs")
    report = {"status": "passed", "task_count": 0, "episode_count": 0, "row_count": 0, "video_count": 0,
              "sidecar_count": 0, "full_raw_frame_count": 0, "full_window_anchors": 0,
              "clean_last_fallback_anchors": 0, "max_geometry_error_m": 0., "permutation": P,
              "inverse_permutation": INVERSE, "source_manifest_sha256": sha256(source_root / "preparation_manifest.json"),
              "target_manifest_sha256": sha256(target_root / "preparation_manifest.json"), "tasks": []}
    full_poses = []
    for name in sorted(task_maps[0]):
        require(Path(name).name == name and name not in (".", ".."), f"Invalid task name {name}")
        clean, source, target = (root / name for root in roots)
        infos, contracts = check_task_metadata(source, target, name)
        require(contracts["state_pose.json"][1] == new["state_contract"], f"{name}: root/task current-state contracts disagree")
        require(contracts["tcp_pose.json"][1]["target_sampling"] == policy, f"{name}: task goal policy changed")
        clean_info = read_json(clean / "meta/info.json")
        original_info = deepcopy(infos[0])
        require(original_info["features"].pop(JOINT) == clean_info["features"][STATE],
                f"{name}: original joint-backup feature metadata changed")
        original_info["features"].pop(TCP)
        original_info["features"][STATE] = clean_info["features"][STATE]
        require(original_info == clean_info, f"{name}: non-pose info differs from original Clean")
        clean_modality = read_json(clean / "meta/modality.json")
        source_modality = read_json(source / "meta/modality.json")
        require(without(clean_modality, {"state"}) == without(source_modality, {"state"})
                and source_modality.get("state") == {"tcp_pose": {"start": 0, "end": 14, "original_key": STATE}},
                f"{name}: Clean action/video/language modalities changed")
        for filename in ("tasks.jsonl", "episodes.jsonl"):
            require(sha256(clean / "meta" / filename) == sha256(source / "meta" / filename), f"{name}: Clean language metadata changed")
        episodes = common.read_jsonl(clean / "meta/episodes.jsonl", "episode_index")
        language = common.read_jsonl(clean / "meta/tasks.jsonl", "task_index")
        tasks = [mapping[name] for mapping in task_maps]
        require(tasks[1].get("quaternion_order") == "xyzw"
                and without(tasks[0], {"episodes"}) == without(tasks[1], {"episodes", "quaternion_order"}),
                f"{name}: task provenance changed")
        mappings = [{row["public_episode_id"]: row for row in task["episodes"]} for task in tasks]
        require(all(len(mapping) == len(task["episodes"]) == expected_episodes_per_task for mapping, task in zip(mappings, tasks))
                and set(mappings[0]) == set(mappings[1]) == set(episodes), f"{name}: episode IDs/counts differ")
        aggregate = [read_json(root / "meta/stats_gr00t.json") for root in (source, target)]
        check_stats_mapping(*aggregate, name)
        episodic = [common.read_jsonl(root / "meta/episodes_stats.jsonl", "episode_index") for root in (source, target)]
        require(all(set(mapping) == set(episodes) for mapping in episodic), f"{name}: episode statistics IDs differ")
        parquet_files, sidecar_files, current_poses = set(), set(), []
        task_frames = 0
        for eid in sorted(mappings[0]):
            records = [mapping[eid] for mapping in mappings]
            context = f"{name}/episode_{eid:06d}"
            changed_record = {"parquet", "parquet_sha256", "sidecar", "sidecar_sha256",
                              "source_pose_state_parquet", "source_pose_state_parquet_sha256",
                              "source_pose_state_sidecar", "source_pose_state_sidecar_sha256",
                              "original_non_state_columns_and_types_preserved", "original_non_pose_columns_and_types_preserved",
                              "changed_columns", "quaternion_order"}
            require(without(records[0], changed_record) == without(records[1], changed_record), f"{context}: Clean/RAW mapping changed")
            require(records[1].get("original_non_state_columns_and_types_preserved") is False
                    and records[1].get("original_non_pose_columns_and_types_preserved") is True
                    and records[1].get("changed_columns") == POSE_COLUMNS and records[1].get("quaternion_order") == "xyzw",
                    f"{context}: incorrect per-episode migration flags")
            length = records[0]["public_length"]
            require(type(length) is int and length > 0 and records[0]["raw_length"] == length + 1, f"{context}: bad lengths")
            relative = Path(clean_info["data_path"].format(episode_chunk=eid // clean_info["chunks_size"], episode_index=eid))
            require(not relative.is_absolute() and ".." not in relative.parts, f"{context}: unsafe data path")
            parquet_files.add(relative)
            paths = [root / relative for root in (clean, source, target)]
            tables = [pq.read_table(path) for path in paths]
            require(all(table.num_rows == length for table in tables), f"{context}: row count changed")
            require(tables[1].column_names == tables[2].column_names
                    and tables[1].schema.remove_metadata().equals(tables[2].schema.remove_metadata()), f"{context}: Parquet fields/dtypes/nullability changed")
            common.same_columns(tables[0], tables[1], [key for key in tables[0].column_names if key != STATE], context + " Clean/WXYZ")
            common.same_columns(tables[1], tables[2], [key for key in tables[1].column_names if key not in POSE_COLUMNS], context + " non-pose")
            require(equal_bits(common.pose_array(tables[0], STATE, context), common.pose_array(tables[1], JOINT, context)), f"{context}: backup differs from Clean joints")
            old_values, new_values = [], []
            for column in POSE_COLUMNS:
                old_values.append(common.pose_array(tables[1], column, context))
                new_values.append(common.pose_array(tables[2], column, context))
                check_permutation(old_values[-1], new_values[-1], f"{context}/{column}")
            require(equal_bits(old_values[0], old_values[1]) and equal_bits(new_values[0], new_values[1]), f"{context}: current state/TCP columns disagree")
            old_metadata, new_metadata = dict(tables[1].schema.metadata or {}), dict(tables[2].schema.metadata or {})
            arrow_contract = json.loads(new_metadata.pop(b"tcp_pose_contract", b"{}"))
            require(new_metadata == old_metadata and arrow_contract == {
                "schema_version": 2, "features": POSE_COLUMNS, "quaternion_order": "xyzw", "layout": LAYOUT,
                "coordinate_frame": "world", "position_unit": "meters", "normalization_applied": False}, f"{context}: Arrow metadata changed beyond declared XYZW contract")
            hashes = [sha256(path) for path in paths]
            require(hashes[0] == records[0]["source_parquet_sha256"] and hashes[1] == records[0]["parquet_sha256"]
                    == records[1]["source_pose_state_parquet_sha256"] and hashes[2] == records[1]["parquet_sha256"], f"{context}: Parquet SHA provenance mismatch")
            require(Path(records[1]["parquet"]).resolve() == paths[2].resolve()
                    and Path(records[1]["source_pose_state_parquet"]).resolve() == paths[1].resolve(), f"{context}: Parquet provenance path mismatch")
            require(np.array_equal(tables[0]["frame_index"].to_numpy(), np.arange(length))
                    and np.array_equal(tables[0]["index"].to_numpy(), np.arange(task_frames, task_frames + length))
                    and np.array_equal(tables[0]["episode_index"].to_numpy(), np.full(length, eid)), f"{context}: missing/reordered frame identity")
            indices = tables[0]["task_index"].to_numpy()
            require(all(index in language for index in indices) and {language[index]["task"] for index in indices} == set(episodes[eid]["tasks"])
                    and episodes[eid]["length"] == length, f"{context}: row language/episode mapping mismatch")
            require(all(tables[0][key].null_count == 0 for key in common.IDS)
                    and np.isfinite(tables[0]["timestamp"].to_numpy()).all(), f"{context}: invalid row IDs/timestamps")
            sidecar_relative = Path(contracts["tcp_pose.json"][0]["sidecar_template"].format(episode_chunk=eid // infos[0]["chunks_size"], episode_index=eid))
            require(not sidecar_relative.is_absolute() and ".." not in sidecar_relative.parts, f"{context}: unsafe sidecar path")
            sidecar_files.add(sidecar_relative)
            sidecars = [root / sidecar_relative for root in (source, target)]
            sidecar_hashes = [sha256(path) for path in sidecars]
            require(sidecar_hashes[0] == records[0]["sidecar_sha256"] == records[1]["source_pose_state_sidecar_sha256"]
                    and sidecar_hashes[1] == records[1]["sidecar_sha256"] and sidecar_hashes[0] != sidecar_hashes[1], f"{context}: sidecar SHA provenance mismatch")
            require(Path(records[1]["sidecar"]).resolve() == sidecars[1].resolve()
                    and Path(records[1]["source_pose_state_sidecar"]).resolve() == sidecars[0].resolve(), f"{context}: sidecar provenance path mismatch")
            with np.load(sidecars[0], allow_pickle=False) as data:
                old_arrays = {key: data[key] for key in data.files}
            with np.load(sidecars[1], allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            require(set(old_arrays) == {EE, OLD_TCP, "metadata_json"} and set(arrays) == {EE, NEW_TCP, "metadata_json"}, f"{context}: noncanonical sidecar keys")
            require(equal_bits(arrays[EE], old_arrays[EE]) and arrays[NEW_TCP].shape == (length + 1, 14), f"{context}: EE provenance or terminal length changed")
            check_permutation(old_arrays[OLD_TCP], arrays[NEW_TCP], context + " full future TCP sequence")
            require(equal_bits(old_arrays[OLD_TCP][:-1], old_values[0]) and equal_bits(arrays[NEW_TCP][:-1], new_values[0]), f"{context}: current pose is shifted against future sequence")
            old_details, details = (json.loads(str(value["metadata_json"])) for value in (old_arrays, arrays))
            sidecar_changes = {"quaternion_order", "layout", "tcp_definition", "target_pose_array", "array_quaternion_orders",
                               "provenance_arrays", "source_pose_state_sidecar", "source_pose_state_sidecar_sha256"}
            require(without(old_details, sidecar_changes) == without(details, sidecar_changes), f"{context}: sidecar source identity/RAW mapping changed")
            require(details.get("quaternion_order") == "xyzw" and details.get("layout") == LAYOUT
                    and details.get("tcp_definition") == DEFINITION and details.get("target_pose_array") == NEW_TCP
                    and details.get("array_quaternion_orders") == ARRAY_ORDERS and details.get("provenance_arrays") == [EE]
                    and details.get("target_sampling") == policy, f"{context}: sidecar XYZW/current/future contract mismatch")
            require(details.get("source_pose_state_sidecar_sha256") == sidecar_hashes[0]
                    and Path(details.get("source_pose_state_sidecar", "")).resolve() == sidecars[0].resolve(), f"{context}: sidecar metadata source SHA/path mismatch")
            for key, expected in {"task": name, "public_episode_id": eid, "public_length": length, "raw_length": length + 1}.items():
                require(details.get(key) == expected, f"{context}: sidecar {key} mismatch")
            report["max_geometry_error_m"] = max(report["max_geometry_error_m"], check_geometry(arrays[EE], arrays[NEW_TCP], context))
            old_stats, stats = [mapping[eid] for mapping in episodic]
            require(without(old_stats, {"stats"}) == without(stats, {"stats"}), f"{context}: episode statistics identity changed")
            check_stats_mapping(old_stats["stats"], stats["stats"], context)
            for column in POSE_COLUMNS:
                common.check_statistics(stats["stats"][column], new_values[0], f"{context}/{column}")
            cameras = [key for key, feature in clean_info["features"].items() if feature.get("dtype") == "video"]
            require(len(cameras) == 3, f"{context}: expected three video streams")
            for key in cameras:
                relative_video = clean_info["video_path"].format(episode_chunk=eid // clean_info["chunks_size"], episode_index=eid, video_key=key)
                videos = [root / relative_video for root in (clean, source, target)]
                require(videos[0].is_file() and all(path.is_file() and path.samefile(videos[0]) for path in videos[1:]), f"{context}: video no longer shares the Clean source")
            task_frames += length
            full_poses.append(arrays[NEW_TCP])
            current_poses.append(new_values[0])
            report["episode_count"] += 1
            report["sidecar_count"] += 1
            report["video_count"] += len(cameras)
            report["full_raw_frame_count"] += length + 1
            report["full_window_anchors"] += max(0, length - 95)
            report["clean_last_fallback_anchors"] += length - max(0, length - 95)
        for root in (clean, source, target):
            require({path.relative_to(root) for path in (root / "data").rglob("*.parquet")} == parquet_files, f"{name}: extra/missing Parquet episodes")
        for root in (source, target):
            require({path.relative_to(root) for path in (root / "tcp").rglob("*.npz")} == sidecar_files, f"{name}: extra/missing future sidecars")
        require(task_frames == tasks[0]["public_frames"] == clean_info["total_frames"]
                and tasks[0]["episode_count"] == clean_info["total_episodes"] == len(mappings[0]), f"{name}: task/frame totals mismatch")
        for column in POSE_COLUMNS:
            common.check_statistics(aggregate[1][column], np.concatenate(current_poses), f"{name}/{column}")
        report["task_count"] += 1
        report["row_count"] += task_frames
        report["tasks"].append({"task": name, "rows": task_frames, "episodes": len(mappings[0])})
    diagnostics = [read_json(root / "tcp_statistics.json") for root in (source_root, target_root)]
    require(without(diagnostics[0], {"layout", "quaternion_order", "target_pose_array", "stats"})
            == without(diagnostics[1], {"layout", "quaternion_order", "target_pose_array", "stats"}), "Non-pose TCP diagnostic metadata changed")
    require(diagnostics[1].get("layout") == LAYOUT and diagnostics[1].get("quaternion_order") == "xyzw"
            and diagnostics[1].get("target_pose_array") == NEW_TCP, "Global TCP diagnostics must declare XYZW")
    check_statistics(diagnostics[0]["stats"], diagnostics[1]["stats"], "Global full RAW TCP diagnostics")
    common.check_statistics(diagnostics[1]["stats"], np.concatenate(full_poses), "Global full RAW TCP diagnostics")
    expected_episodes = expected_tasks * expected_episodes_per_task
    require(report["row_count"] == expected_frames and report["episode_count"] == expected_episodes, "Complete release totals mismatch")
    for manifest in manifests:
        for key, expected in {"task_count": expected_tasks, "episode_count": expected_episodes,
                              "public_frame_count": expected_frames, "full_raw_frame_count": expected_frames + expected_episodes}.items():
            require(manifest.get(key) == expected, f"Manifest {key} mismatch")
    require(diagnostics[1]["frame_count"] == report["full_raw_frame_count"], "Full RAW diagnostic statistics count mismatch")
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean"))
    parser.add_argument("--source-pose-state-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean-TCP-PoseState"))
    parser.add_argument("--xyzw-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean-TCP-PoseState-XYZW"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    roots = (args.clean_root, args.source_pose_state_root, args.xyzw_root)
    report_path = args.report.expanduser().resolve()
    if any(report_path.is_relative_to(root.expanduser().resolve()) for root in roots):
        print("ERROR: --report must be outside dataset roots", file=sys.stderr)
        return 1
    started = time.perf_counter()
    try:
        report = validate_correspondence(*roots)
    except Exception as error:
        report = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                  "elapsed_seconds": round(time.perf_counter() - started, 3)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(without(report, {"tasks"}), allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
