#!/usr/bin/env python3
"""Independently compare every Clean, TCP, and TCP-pose-state observation.

Only the explicitly requested report is written. Dataset files are never
changed, and shared videos are checked by file identity without decoding.
The command requires the complete 50-task / 2,500-episode / 549,787-row release.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STATE = "observation.state"
TCP = "observation.tcp_pose"
BACKUP = "observation.joint_state"
LAYOUT = [f"{arm}_{part}" for arm in ("left", "right")
          for part in ("x", "y", "z", "qw", "qx", "qy", "qz")]
IDS = ("frame_index", "episode_index", "index", "task_index", "timestamp")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path, key):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate {key} in {path}")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def without_keys(value, keys):
    return {key: item for key, item in value.items() if key not in keys}


def same_columns(left, right, names, context):
    require(left.num_rows == right.num_rows, f"{context}: row count changed")
    for name in names:
        require(name in right.column_names, f"{context}: missing column {name}")
        require(left.schema.field(name).equals(right.schema.field(name), check_metadata=True),
                f"{context}: dtype/nullability/field metadata changed for {name}")
        require(left[name].null_count == right[name].null_count and left[name].equals(right[name]),
                f"{context}: row values or nulls changed for {name}")


def pose_array(table, name, context):
    field = table.schema.field(name)
    require(pa.types.is_fixed_size_list(field.type) and field.type.list_size == 14
            and field.type.value_type == pa.float32(), f"{context}: {name} must be fixed float32[14]")
    column = table[name].combine_chunks()
    require(column.null_count == 0 and column.values.null_count == 0, f"{context}: nulls in {name}")
    values = column.values.to_numpy(zero_copy_only=False).reshape(-1, 14)
    require(np.isfinite(values).all(), f"{context}: nonfinite {name}")
    return values


def cleaned_arrow_metadata(table, excluded_features):
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        huggingface = json.loads(metadata[b"huggingface"])
        features = huggingface.get("info", {}).get("features", {})
        for name in excluded_features:
            features.pop(name, None)
        metadata[b"huggingface"] = huggingface
    return metadata


def check_statistics(record, values, context):
    expected = {"mean": values.mean(axis=0, dtype=np.float64),
                "std": values.std(axis=0, dtype=np.float64),
                "min": values.min(axis=0), "max": values.max(axis=0)}
    for key in ("q01", "q99"):
        if key in record:
            expected[key] = np.quantile(values, 0.01 if key == "q01" else 0.99, axis=0)
    for key, target in expected.items():
        actual = np.asarray(record.get(key, []))
        require(actual.shape == (14,) and np.isfinite(actual).all()
                and np.allclose(actual, target, rtol=1e-6, atol=1e-7), f"{context}: incorrect {key} statistics")
    if "count" in record:
        require(record["count"] == [len(values)], f"{context}: incorrect statistics count")


def check_state_contract(contract, tcp_contract, context):
    expected = {"schema_version": 1, "state_representation": "tcp_pose", "feature": STATE,
                "backup_joint_feature": BACKUP, "current_frame": True, "normalization_applied": False,
                "coordinate_frame": "world", "position_unit": "meters", "quaternion_order": "wxyz",
                "layout": LAYOUT, "tcp_definition": tcp_contract["tcp_definition"]}
    for key, value in expected.items():
        require(key in contract and type(contract[key]) is type(value) and contract[key] == value,
                f"{context}: invalid state contract {key}")
    if "dtype" in contract:
        require(contract["dtype"] == "float32", f"{context}: state dtype must be float32")
    if "shape" in contract:
        require(contract["shape"] == [14], f"{context}: state shape must be [14]")


def check_geometry(path, task, episode, length, current_tcp, policy):
    with np.load(path, allow_pickle=False) as sidecar:
        ee = sidecar["ee_pose_world_wxyz"]
        tcp = sidecar["tcp_pose_world_wxyz"]
        metadata = json.loads(str(sidecar["metadata_json"]))
    context = f"{task}/episode_{episode:06d} sidecar"
    require(ee.dtype == np.float64 and ee.shape == (length + 1, 14) and np.isfinite(ee).all(),
            f"{context}: invalid EE array")
    require(tcp.dtype == np.float32 and tcp.shape == (length + 1, 14) and np.isfinite(tcp).all(),
            f"{context}: invalid TCP array")
    require(np.array_equal(current_tcp, tcp[:-1]), f"{context}: current TCP is shifted or differs from sidecar")
    for key, expected in {"task": task, "public_episode_id": episode, "public_length": length,
                          "raw_length": length + 1, "coordinate_frame": "world", "position_unit": "meters",
                          "quaternion_order": "wxyz", "layout": LAYOUT, "target_sampling": policy}.items():
        require(metadata.get(key) == expected, f"{context}: incorrect metadata {key}")
    maximum_error = 0.0
    # Use quaternion/vector rotation, independently of the converter's explicit
    # rotation-matrix column formula. Check both arms and the RAW-only terminal.
    for start in (0, 7):
        quaternion = ee[:, start + 3:start + 7]
        norm = np.linalg.norm(quaternion, axis=1)
        require(np.allclose(norm, 1, rtol=0, atol=1e-5), f"{context}: EE quaternion is not unit wxyz")
        q = quaternion / norm[:, None]
        vector = np.broadcast_to(np.array([0.12, 0., 0.]), (length + 1, 3))
        cross = 2 * np.cross(q[:, 1:], vector)
        rotated = vector + q[:, :1] * cross + np.cross(q[:, 1:], cross)
        expected_position = ee[:, start:start + 3] + rotated
        error = float(np.max(np.abs(tcp[:, start:start + 3] - expected_position)))
        maximum_error = max(maximum_error, error)
        require(error <= 1e-7, f"{context}: EE-to-TCP geometry error {error} exceeds 1e-7 m")
        require(np.array_equal(tcp[:, start + 3:start + 7], quaternion.astype(np.float32)),
                f"{context}: TCP quaternion/order/sign changed")
        require(np.allclose(np.linalg.norm(tcp[:, start + 3:start + 7], axis=1), 1, rtol=0, atol=1e-5),
                f"{context}: TCP quaternion is not unit")
    return maximum_error


def metadata_files(directory):
    return {str(path.relative_to(directory)): path for path in directory.rglob("*")
            if path.is_file() and path.suffix != ".pkl"
            and ".ipynb_checkpoints" not in path.relative_to(directory).parts}


def check_task_metadata(clean, source, target):
    context = clean.name
    infos = [read_json(root / "meta/info.json") for root in (clean, source, target)]
    original_info, source_info, target_info = (deepcopy(info) for info in infos)
    source_pose_feature = source_info["features"].pop(TCP)
    require(source_info == original_info, f"{context}: Clean-to-TCP info changed beyond TCP feature")
    require(source_pose_feature["dtype"] == "float32" and source_pose_feature["shape"] == [14],
            f"{context}: invalid source TCP feature")
    state_feature = target_info["features"].get(STATE, {})
    require(state_feature.get("dtype") == "float32" and state_feature.get("shape") == [14]
            and state_feature.get("names") in (LAYOUT, [LAYOUT]),
            f"{context}: state feature must describe the existing TCP pose layout")
    require(target_info["features"].pop(BACKUP, None) == original_info["features"][STATE],
            f"{context}: backup joint feature changed")
    target_info["features"][STATE] = original_info["features"][STATE]
    require(target_info == infos[1], f"{context}: target info changed beyond state/backup features")
    modalities = [read_json(root / "meta/modality.json") for root in (clean, source, target)]
    require(modalities[0] == modalities[1], f"{context}: Clean-to-TCP modalities changed")
    require(modalities[2].get("state") == {"tcp_pose": {"start": 0, "end": 14, "original_key": STATE}},
            f"{context}: state modality must be the single 14D current TCP")
    require(without_keys(modalities[1], {"state"}) == without_keys(modalities[2], {"state"}),
            f"{context}: action/video/language modalities changed")
    tcp_contract = read_json(source / "meta/tcp_pose.json")
    require(read_json(target / "meta/tcp_pose.json") == tcp_contract, f"{context}: future TCP contract changed")
    state_contract = read_json(target / "meta/state_pose.json")
    check_state_contract(state_contract, tcp_contract, context)
    excluded = {"info.json", "modality.json", "stats_gr00t.json", "episodes_stats.jsonl", "state_pose.json"}
    source_files, target_files = metadata_files(source / "meta"), metadata_files(target / "meta")
    require(set(source_files) | {"state_pose.json"} == set(target_files), f"{context}: unexpected metadata file changes")
    require(not list((target / "meta").rglob("*.pkl")), f"{context}: stale step-cache files copied")
    for name in set(source_files) - excluded:
        require(sha256(source_files[name]) == sha256(target_files[name]), f"{context}: metadata file changed: {name}")
    for name in ("episodes.jsonl", "tasks.jsonl"):
        require(sha256(clean / "meta" / name) == sha256(source / "meta" / name),
                f"{context}: Clean language/episode metadata changed: {name}")
    return infos, tcp_contract, state_contract


def validate_correspondence(clean_root, source_tcp_root, pose_state_root, *, expected_tasks=50,
                            expected_episodes_per_task=50, expected_frames=549787):
    """Validate all rows; smaller expectations are exposed only for fixture tests."""
    started = time.perf_counter()
    roots = [Path(root).expanduser().resolve(strict=True) for root in (clean_root, source_tcp_root, pose_state_root)]
    clean_root, source_root, target_root = roots
    require(len(set(roots)) == 3, "The three dataset roots must be distinct")
    source_manifest = read_json(source_root / "preparation_manifest.json")
    target_manifest = read_json(target_root / "preparation_manifest.json")
    require(source_manifest.get("status") == target_manifest.get("status") == "passed", "Both manifests must have status passed")
    require(target_manifest.get("state_representation") == "tcp_pose", "Target manifest must declare state_representation=tcp_pose")
    require(target_manifest.get("source_tcp_manifest_sha256") == sha256(source_root / "preparation_manifest.json"),
            "Target manifest source TCP manifest SHA mismatch")
    require(Path(target_manifest.get("source_tcp_root", "")).resolve() == source_root
            and Path(target_manifest.get("clean_root", "")).resolve() == clean_root,
            "Target manifest source TCP/Clean roots differ from the audited roots")
    require(target_manifest.get("source_modified") is False, "Target manifest must declare sources unmodified")
    if "mapping_report_sha256" in source_manifest:
        require(target_manifest.get("mapping_report_sha256") == source_manifest["mapping_report_sha256"],
                "Original RAW/Clean mapping report SHA changed")
    policy = source_manifest["target_sampling"]
    require(target_manifest.get("target_sampling") == policy and policy.get("schema_version") == 2
            and policy.get("start_offset") == 64 and policy.get("end_offset_exclusive") == 96
            and policy.get("full_window_required") is True
            and policy.get("eligible_frame_scope") == "clean_observation_frames"
            and policy.get("fallback_frame") == "n_clean - 1 = len(full_tcp_sequence) - 2",
            "Future TCP goal policy must remain Clean schema 2 with Clean-last fallback")
    task_records = {record["task"]: record for record in source_manifest["tasks"]}
    target_records = {record["task"]: record for record in target_manifest["tasks"]}
    require(len(task_records) == len(source_manifest["tasks"]) == expected_tasks
            and set(task_records) == set(target_records) and len(target_records) == len(target_manifest["tasks"]),
            "Manifest task sets/counts differ")
    for root in roots:
        actual = {path.name for path in root.iterdir() if path.is_dir() and (path / "data").is_dir()}
        require(actual == set(task_records), f"{root.name}: task folders differ from manifest")
    require(sha256(source_root / "tcp_statistics.json") == sha256(target_root / "tcp_statistics.json"),
            "Future TCP diagnostic statistics changed")
    report = {"status": "passed", "task_count": 0, "episode_count": 0, "row_count": 0,
              "video_count": 0, "sidecar_count": 0, "goal_anchor_count": 0,
              "full_window_anchors": 0, "clean_last_fallback_anchors": 0,
              "max_geometry_error_m": 0.0, "checks": "Full rows, schemas/nulls, language, metadata, statistics, sidecar SHA256, geometry and video samefile; no video decoding.",
              "tasks": []}
    for task in sorted(task_records):
        require(Path(task).name == task and task not in (".", ".."), f"Unsafe task name: {task}")
        clean, source, target = (root / task for root in roots)
        infos, tcp_contract, state_contract = check_task_metadata(clean, source, target)
        require(tcp_contract["target_sampling"] == policy, f"{task}: task/root target policy differs")
        require(target_manifest.get("state_contract") == state_contract, f"{task}: task/root state contracts differ")
        source_task, target_task = task_records[task], target_records[task]
        require(target_task.get("state_representation") == "tcp_pose", f"{task}: missing task state representation")
        require(without_keys(source_task, {"episodes"}) == without_keys(target_task, {"episodes", "state_representation"}),
                f"{task}: task identity/counts changed")
        records = {item["public_episode_id"]: item for item in source_task["episodes"]}
        new_records = {item["public_episode_id"]: item for item in target_task["episodes"]}
        require(len(records) == len(source_task["episodes"]) == expected_episodes_per_task
                and len(new_records) == len(target_task["episodes"]) == len(records) and set(records) == set(new_records),
                f"{task}: episode IDs/counts differ")
        require(source_task["episode_count"] == infos[0]["total_episodes"] == len(records),
                f"{task}: declared episode counts differ")
        episodes = read_jsonl(clean / "meta/episodes.jsonl", "episode_index")
        languages = read_jsonl(clean / "meta/tasks.jsonl", "task_index")
        require(set(episodes) == set(records), f"{task}: episode metadata IDs differ")
        all_stats = [read_json(root / "meta/stats_gr00t.json") for root in (clean, source, target)]
        episode_stats = [read_jsonl(root / "meta/episodes_stats.jsonl", "episode_index") for root in (clean, source, target)]
        require(all(set(stats) == set(records) for stats in episode_stats), f"{task}: per-episode statistics IDs differ")
        require(without_keys(all_stats[1], {TCP}) == all_stats[0], f"{task}: original aggregate statistics changed")
        require(without_keys(all_stats[2], {STATE, BACKUP}) == without_keys(all_stats[1], {STATE}),
                f"{task}: non-state aggregate statistics changed")
        require(all_stats[2].get(BACKUP) == all_stats[1][STATE]
                and all_stats[2][STATE] == all_stats[1][TCP], f"{task}: state/backup aggregate statistics do not correspond")
        actual_parquets = []
        current_poses = []
        task_rows = 0
        for episode in sorted(records):
            record, new_record = records[episode], new_records[episode]
            changed = {"parquet", "parquet_sha256", "sidecar", "source_tcp_parquet", "source_tcp_parquet_sha256",
                       "original_columns_and_types_preserved", "original_non_state_columns_and_types_preserved",
                       "clean_joint_state_backed_up", "current_tcp_state_frame_aligned"}
            require(without_keys(record, changed) == without_keys(new_record, changed),
                    f"{task}/{episode}: raw mapping/episode provenance changed")
            require(new_record.get("original_columns_and_types_preserved") is False
                    and all(new_record.get(key) is True for key in (
                        "original_non_state_columns_and_types_preserved", "clean_joint_state_backed_up",
                        "current_tcp_state_frame_aligned")), f"{task}/{episode}: incorrect state-only provenance flags")
            length = record["public_length"]
            require(type(length) is int and length > 0 and record["raw_length"] == length + 1,
                    f"{task}/{episode}: invalid Clean/RAW lengths")
            relative = Path(infos[0]["data_path"].format(episode_chunk=episode // infos[0]["chunks_size"], episode_index=episode))
            require(not relative.is_absolute() and ".." not in relative.parts, f"{task}: unsafe data path")
            paths = [root / relative for root in (clean, source, target)]
            actual_parquets.append(relative)
            tables = [pq.read_table(path) for path in paths]
            context = f"{task}/episode_{episode:06d}"
            require(all(table.num_rows == length for table in tables), f"{context}: number of rows changed")
            require(tables[1].column_names == tables[0].column_names + [TCP], f"{context}: source columns changed")
            require([name for name in tables[2].column_names if name != BACKUP] == tables[1].column_names
                    and tables[2].column_names.count(BACKUP) == 1, f"{context}: target columns changed beyond backup")
            same_columns(tables[0], tables[1], tables[0].column_names, context + " Clean/TCP")
            same_columns(tables[1], tables[2], [name for name in tables[1].column_names if name != STATE], context + " TCP/target")
            require(cleaned_arrow_metadata(tables[0], set()) == cleaned_arrow_metadata(tables[1], {TCP}),
                    f"{context}: Clean/TCP Arrow metadata changed")
            require(cleaned_arrow_metadata(tables[1], {STATE}) == cleaned_arrow_metadata(tables[2], {STATE, BACKUP}),
                    f"{context}: non-state Arrow metadata changed")
            original = pose_array(tables[0], STATE, context)
            current = pose_array(tables[1], TCP, context)
            require(np.array_equal(pose_array(tables[2], STATE, context), current), f"{context}: state is not same-frame current TCP")
            require(np.array_equal(pose_array(tables[2], BACKUP, context), original), f"{context}: backup joints changed")
            for name in IDS:
                require(tables[0][name].null_count == 0, f"{context}: null identity column {name}")
            require(np.array_equal(tables[0]["frame_index"].to_numpy(), np.arange(length)), f"{context}: frames reordered/missing")
            require(np.array_equal(tables[0]["episode_index"].to_numpy(), np.full(length, episode)), f"{context}: episode column mismatch")
            require(np.array_equal(tables[0]["index"].to_numpy(), np.arange(task_rows, task_rows + length)), f"{context}: global indices changed/noncontiguous")
            task_indices = tables[0]["task_index"].to_numpy()
            require(all(value in languages for value in task_indices), f"{context}: row task_index has no language entry")
            row_languages = {languages[value]["task"] for value in task_indices}
            require(row_languages == set(episodes[episode]["tasks"]) and episodes[episode]["length"] == length,
                    f"{context}: row language / episode task mapping differs")
            require(np.isfinite(tables[0]["timestamp"].to_numpy()).all(), f"{context}: nonfinite timestamps")
            hashes = [sha256(path) for path in paths]
            require(hashes[0] == record["source_parquet_sha256"]
                    and hashes[1] == record["parquet_sha256"] == new_record.get("source_tcp_parquet_sha256")
                    and hashes[2] == new_record["parquet_sha256"], f"{context}: manifest Parquet SHA mismatch")
            require(Path(new_record["parquet"]).resolve() == paths[2].resolve(), f"{context}: target manifest parquet path mismatch")
            require(Path(new_record["source_tcp_parquet"]).resolve() == paths[1].resolve(),
                    f"{context}: source TCP parquet provenance path mismatch")
            sidecar_relative = Path(tcp_contract["sidecar_template"].format(episode_chunk=episode // infos[1]["chunks_size"], episode_index=episode))
            require(not sidecar_relative.is_absolute() and ".." not in sidecar_relative.parts, f"{context}: unsafe sidecar path")
            sidecars = [root / sidecar_relative for root in (source, target)]
            require(sha256(sidecars[0]) == sha256(sidecars[1]) == record["sidecar_sha256"] == new_record["sidecar_sha256"],
                    f"{context}: sidecar bytes/SHA changed")
            require(Path(new_record["sidecar"]).resolve() == sidecars[1].resolve(), f"{context}: target manifest sidecar path mismatch")
            error = check_geometry(sidecars[0], task, episode, length, current, policy)
            report["max_geometry_error_m"] = max(report["max_geometry_error_m"], error)
            stats = [mapping[episode] for mapping in episode_stats]
            require(without_keys(stats[1]["stats"], {TCP}) == stats[0]["stats"], f"{context}: original per-episode statistics changed")
            require(without_keys(stats[1], {"stats"}) == without_keys(stats[2], {"stats"})
                    and without_keys(stats[2]["stats"], {STATE, BACKUP}) == without_keys(stats[1]["stats"], {STATE}),
                    f"{context}: non-state per-episode statistics changed")
            require(stats[2]["stats"].get(BACKUP) == stats[1]["stats"][STATE]
                    and stats[2]["stats"][STATE] == stats[1]["stats"][TCP], f"{context}: state/backup episode statistics differ")
            check_statistics(stats[2]["stats"][STATE], current, context)
            video_keys = [key for key, value in infos[0]["features"].items() if value.get("dtype") == "video"]
            require(len(video_keys) == 3, f"{context}: expected exactly three camera streams")
            for key in video_keys:
                relative_video = infos[0]["video_path"].format(episode_chunk=episode // infos[0]["chunks_size"], episode_index=episode, video_key=key)
                videos = [root / relative_video for root in (clean, source, target)]
                require(videos[0].is_file() and all(path.is_file() and path.samefile(videos[0]) for path in videos[1:]),
                        f"{context}: video not shared with original Clean: {key}")
            current_poses.append(current)
            full_window = max(0, length - 95)
            report["full_window_anchors"] += full_window
            report["clean_last_fallback_anchors"] += length - full_window
            report["goal_anchor_count"] += length
            report["episode_count"] += 1
            report["sidecar_count"] += 1
            report["video_count"] += len(video_keys)
            task_rows += length
        for root in (clean, source, target):
            require({path.relative_to(root) for path in (root / "data").rglob("*.parquet")} == set(actual_parquets),
                    f"{task}: unexpected/missing parquet files in {root}")
        require(task_rows == source_task["public_frames"] == infos[0]["total_frames"], f"{task}: task frame counts differ")
        require(source_task["full_raw_frames"] == task_rows + len(records), f"{task}: RAW terminal-frame counts differ")
        check_statistics(all_stats[2][STATE], np.concatenate(current_poses), task)
        report["task_count"] += 1
        report["row_count"] += task_rows
        report["tasks"].append({"task": task, "episodes": len(records), "rows": task_rows})
    expected_episodes = expected_tasks * expected_episodes_per_task
    require(report["row_count"] == expected_frames and report["episode_count"] == expected_episodes,
            f"Release totals differ: {report['episode_count']} episodes, {report['row_count']} rows")
    for manifest in (source_manifest, target_manifest):
        for key, value in {"task_count": expected_tasks, "episode_count": expected_episodes,
                           "public_frame_count": expected_frames}.items():
            require(manifest.get(key) == value, f"Manifest {key} differs from verified total")
        if "full_raw_frame_count" in manifest:
            require(manifest["full_raw_frame_count"] == expected_frames + expected_episodes,
                    "Manifest RAW frame count differs from verified total")
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean"))
    parser.add_argument("--source-tcp-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean-TCP"))
    parser.add_argument("--pose-state-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean-TCP-PoseState"))
    parser.add_argument("--report", type=Path, required=True, help="Write the JSON result outside all three dataset roots")
    args = parser.parse_args(argv)
    roots = (args.clean_root, args.source_tcp_root, args.pose_state_root)
    report_path = args.report.expanduser().resolve()
    if any(report_path.is_relative_to(root.expanduser().resolve()) for root in roots):
        print("ERROR: --report must be outside all dataset roots", file=sys.stderr)
        return 1
    started = time.perf_counter()
    try:
        report = validate_correspondence(*roots)
    except Exception as error:
        report = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                  "elapsed_seconds": round(time.perf_counter() - started, 3)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report if report["status"] == "failed" else without_keys(report, {"tasks"}), allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
