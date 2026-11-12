#!/usr/bin/env python3
"""Create a separate Clean training view with current-frame TCP poses.

All existing Parquet columns, episode IDs, language, and actions are preserved.
Videos are shared by symlink; metadata is copied privately without step caches.
No model, loader, original Clean file, or source HDF5 is modified.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TCP_COLUMN = "observation.tcp_pose"
POSE_LAYOUT = [f"{arm}_{name}" for arm in ("left", "right")
               for name in ("x", "y", "z", "qw", "qx", "qy", "qz")]
GOAL_POLICY = {
    "schema_version": 2,
    "start_offset": 64, "end_offset_exclusive": 96,
    "window": "[t + 64, t + 96)",
    "eligible_frame_scope": "clean_observation_frames",
    "eligible_frame_indices": "0 through len(full_tcp_sequence) - 2, inclusive",
    "full_window_required": True,
    "full_window_condition": "t + 96 <= n_clean, where n_clean = len(full_tcp_sequence) - 1",
    "incomplete_window_policy": "Use the last Clean observation frame, which is the penultimate RAW frame.",
    "fallback_frame": "n_clean - 1 = len(full_tcp_sequence) - 2",
    "raw_terminal_frame_usage": "Preserved for provenance only; never selected as a target.",
    "sampling": "Caller-provided numpy.random.Generator; no reseeding or fixed goal labels.",
}

RAW_LAST_GOAL_POLICY = {
    **GOAL_POLICY,
    "schema_version": 3,
    "variant": "raw_last_fallback",
    "eligible_frame_scope": "clean_observation_windows_with_raw_terminal_fallback",
    "eligible_frame_indices": "Clean observation frames for random windows; final RAW frame for fallback only",
    "incomplete_window_policy": "Use the last RAW frame, corresponding to the time of the last Clean action target.",
    "fallback_frame": "n_clean = len(full_tcp_sequence) - 1",
    "raw_terminal_frame_usage": "Selected only when the complete Clean-frame future window does not fit.",
}


def sample_target_tcp_pose(tcp_sequence, t, rng, target_sampling=None):
    """Sample a future TCP using an explicit dataset policy.

    The full sidecar includes one extra RAW terminal frame for provenance.
    Random windows are restricted to n_clean = len(tcp_sequence) - 1 Clean
    observations under both policies. A complete window requires t + 96 <=
    n_clean. The default falls back to n_clean - 1 (last Clean observation).
    RAW_LAST_GOAL_POLICY falls back to n_clean (last RAW frame). Consumers of
    a dataset variant must pass its manifest's target_sampling explicitly.
    No source array is mutated or random generator reseeded.
    """
    policy = GOAL_POLICY if target_sampling is None else target_sampling
    if not isinstance(policy, dict):
        raise ValueError("target_sampling must be a supported dataset policy dictionary.")
    if policy == GOAL_POLICY:
        use_raw_last = False
    elif policy == RAW_LAST_GOAL_POLICY:
        use_raw_last = True
    else:
        raise ValueError("Unsupported TCP target policy; read the dataset manifest's target_sampling.")
    sequence = np.asarray(tcp_sequence)
    if sequence.ndim != 2 or sequence.shape[1] != 14 or len(sequence) < 2:
        raise ValueError("Expected a full TCP sequence shaped [N + 1, 14].")
    if not isinstance(t, (int, np.integer)) or not 0 <= t < len(sequence) - 1:
        raise ValueError("Observation index must satisfy 0 <= t < len(sequence) - 1.")
    n_clean = len(sequence) - 1
    terminal_fallback = t + 96 > n_clean
    fallback_frame = n_clean if use_raw_last else n_clean - 1
    goal_frame = fallback_frame if terminal_fallback else int(rng.integers(t + 64, t + 96))
    return {"target_tcp_pose": np.array(sequence[goal_frame], dtype=np.float32, copy=True),
            "goal_frame": goal_frame, "used_terminal_fallback": bool(terminal_fallback)}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def ee_to_tcp(ee):
    """Apply EE @ Tx(0.12 m); retain source quaternion values and order."""
    ee = np.asarray(ee, dtype=np.float64)
    if ee.ndim != 2 or ee.shape[1] != 7 or not np.isfinite(ee).all():
        raise ValueError("Expected finite EE poses shaped [T, 7].")
    w, x, y, z = ee[:, 3:7].T
    norm_sq = w*w + x*x + y*y + z*z
    if not np.allclose(np.sqrt(norm_sq), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Source EE quaternions are not unit quaternions.")
    local_x_in_world = np.stack(((w*w+x*x-y*y-z*z) / norm_sq,
                                 2*(x*y+w*z) / norm_sq,
                                 2*(x*z-w*y) / norm_sq), axis=1)
    tcp = ee.copy()
    tcp[:, :3] += 0.12 * local_x_in_world
    return tcp.astype(np.float32)


def stats(array, quantiles=True, count=False):
    array = np.asarray(array)
    result = {"mean": np.mean(array, axis=0, dtype=np.float64).tolist(),
              "std": np.std(array, axis=0, dtype=np.float64).tolist(),
              "min": np.min(array, axis=0).tolist(),
              "max": np.max(array, axis=0).tolist()}
    if quantiles:
        result.update(q01=np.quantile(array, 0.01, axis=0).tolist(),
                      q99=np.quantile(array, 0.99, axis=0).tolist())
    if count:
        result["count"] = [len(array)]
    return result


def copy_metadata(source_meta, target_meta):
    target_meta.mkdir(parents=True, exist_ok=False)
    rewritten = {"info.json", "stats_gr00t.json", "episodes_stats.jsonl"}
    for path in source_meta.rglob("*"):
        relative = path.relative_to(source_meta)
        # Step caches can embed absolute source paths. Rebuild in the new view.
        if path.suffix == ".pkl" or (len(relative.parts) == 1 and path.name in rewritten):
            continue
        target = target_meta / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def arrow_joint_array(table, name):
    column = table[name].combine_chunks()
    if column.null_count or column.type.list_size != 14:
        raise ValueError(f"Expected a non-null 14D joint column: {name}")
    return np.asarray(column.values.to_numpy(zero_copy_only=False), dtype="<f4").reshape(-1, 14)


def process_task(task_record, source_root, output_root, mapping_hash):
    task = task_record["task"]
    source_task = source_root / task
    output_task = output_root / task
    output_task.mkdir(exist_ok=False)
    (output_task / "videos").symlink_to((source_task / "videos").resolve(), target_is_directory=True)
    copy_metadata(source_task / "meta", output_task / "meta")
    episode_stats, current_poses, full_poses, records = {}, [], [], []
    for entry in sorted(task_record["entries"], key=lambda value: value["public_episode_id"]):
        if (entry["status"] != "matched" or not entry["pair_is_unique"] or
                entry["alignment_rule"] != "drop_final_state_next_action"):
            raise ValueError(f"Unsupported or nonunique mapping in {task}")
        source_parquet = Path(entry["public_parquet"])
        relative = source_parquet.relative_to(source_task)
        source_hdf5 = Path(entry["raw_archive"]).parent / entry["raw_member"]
        if sha256_file(source_parquet) != entry["public_parquet_sha256"]:
            raise ValueError(f"Public Parquet SHA256 mismatch: {source_parquet}")
        if sha256_file(source_hdf5) != entry["raw_member_sha256"]:
            raise ValueError(f"Source HDF5 SHA256 mismatch: {source_hdf5}")
        table = pq.read_table(source_parquet)
        if TCP_COLUMN in table.column_names:
            raise ValueError(f"Source already contains {TCP_COLUMN}: {source_parquet}")
        n = entry["public_length"]
        with h5py.File(source_hdf5, "r") as raw:
            joints = np.asarray(raw["joint_action/vector"][:], dtype="<f4")
            left = np.asarray(raw["endpose/left_endpose"][:], dtype=np.float64)
            right = np.asarray(raw["endpose/right_endpose"][:], dtype=np.float64)
        if (table.num_rows != n or joints.shape != (n+1, 14) or
                left.shape != (n+1, 7) or right.shape != (n+1, 7)):
            raise ValueError(f"Temporal shape mismatch: {task}, episode {entry['public_episode_id']}")
        if (not np.array_equal(arrow_joint_array(table, "observation.state"), joints[:-1]) or
                not np.array_equal(arrow_joint_array(table, "action"), joints[1:])):
            raise ValueError(f"Public state/action values do not match RAW: {source_parquet}")
        ee = np.concatenate((left, right), axis=1)
        tcp = np.concatenate((ee_to_tcp(left), ee_to_tcp(right)), axis=1)
        current_tcp = tcp[:-1]
        field_type = pa.list_(pa.field("element", pa.float32()), 14)
        column = pa.FixedSizeListArray.from_arrays(pa.array(current_tcp.reshape(-1), type=pa.float32()), 14)
        augmented = table.append_column(pa.field(TCP_COLUMN, field_type, nullable=False), column)
        metadata = dict(table.schema.metadata or {})
        huggingface = json.loads(metadata.get(b"huggingface", b"{}"))
        huggingface.setdefault("info", {}).setdefault("features", {})[TCP_COLUMN] = {
            "feature": {"dtype": "float32", "_type": "Value"}, "length": 14, "_type": "List"}
        metadata[b"huggingface"] = json.dumps(huggingface).encode("utf-8")
        augmented = augmented.replace_schema_metadata(metadata)
        target_parquet = output_task / relative
        target_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(augmented, target_parquet, compression="snappy")
        written = pq.read_table(target_parquet)
        if not written.select(table.column_names).equals(table, check_metadata=False):
            raise ValueError(f"An original Parquet column changed: {target_parquet}")
        if not written.select(table.column_names).schema.remove_metadata().equals(table.schema.remove_metadata()):
            raise ValueError(f"An original Parquet type changed: {target_parquet}")
        actual_tcp = np.asarray(written[TCP_COLUMN].combine_chunks().values).reshape(n, 14)
        if not np.array_equal(actual_tcp, current_tcp):
            raise ValueError(f"TCP Parquet roundtrip mismatch: {target_parquet}")
        sidecar = output_task / "tcp" / relative.relative_to("data").with_suffix(".npz")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar_metadata = {
            "task": task, "public_episode_id": entry["public_episode_id"],
            "public_length": n, "raw_length": n+1, "raw_episode_id": entry["raw_episode_id"],
            "source_parquet": str(source_parquet), "source_parquet_sha256": entry["public_parquet_sha256"],
            "source_hdf5": str(source_hdf5), "source_hdf5_sha256": entry["raw_member_sha256"],
            "mapping_report_sha256": mapping_hash, "coordinate_frame": "world",
            "position_unit": "meters", "quaternion_order": "wxyz", "layout": POSE_LAYOUT,
            "ee_dtype": "float64", "tcp_dtype": "float32",
            "tcp_definition": "EE @ Tx(0.12 meters); p_TCP=p_EE+R(q_wxyz)@[0.12,0,0]; q_TCP=q_EE",
            "terminal_frame_preserved": True, "target_sampling": GOAL_POLICY,
        }
        with sidecar.open("xb") as stream:
            np.savez_compressed(stream, ee_pose_world_wxyz=ee, tcp_pose_world_wxyz=tcp,
                                metadata_json=np.array(json.dumps(sidecar_metadata)))
        episode_stats[entry["public_episode_id"]] = stats(current_tcp, quantiles=False, count=True)
        current_poses.append(current_tcp)
        full_poses.append(tcp)
        records.append({"public_episode_id": entry["public_episode_id"], "public_length": n,
                        "raw_episode_id": entry["raw_episode_id"], "raw_length": n+1,
                        "parquet": str(target_parquet), "parquet_sha256": sha256_file(target_parquet),
                        "sidecar": str(sidecar), "sidecar_sha256": sha256_file(sidecar),
                        "source_parquet_sha256": entry["public_parquet_sha256"],
                        "source_hdf5_sha256": entry["raw_member_sha256"],
                        "original_columns_and_types_preserved": True})
    info = json.loads((source_task / "meta/info.json").read_text())
    info["features"][TCP_COLUMN] = {"dtype": "float32", "shape": [14], "names": [POSE_LAYOUT]}
    write_json(output_task / "meta/info.json", info)
    aggregate = json.loads((source_task / "meta/stats_gr00t.json").read_text())
    aggregate[TCP_COLUMN] = stats(np.concatenate(current_poses, axis=0))
    write_json(output_task / "meta/stats_gr00t.json", aggregate)
    source_episode_stats = [json.loads(line) for line in (source_task / "meta/episodes_stats.jsonl").read_text().splitlines() if line.strip()]
    if {int(item["episode_index"]) for item in source_episode_stats} != set(episode_stats):
        raise ValueError(f"Episode statistics coverage mismatch: {task}")
    with (output_task / "meta/episodes_stats.jsonl").open("x") as stream:
        for item in source_episode_stats:
            item["stats"][TCP_COLUMN] = episode_stats[int(item["episode_index"])]
            stream.write(json.dumps(item, allow_nan=False) + "\n")
    write_json(output_task / "meta/tcp_pose.json", {
        "schema_version": 1, "feature": TCP_COLUMN, "coordinate_frame": "world",
        "position_unit": "meters", "quaternion_order": "wxyz", "layout": POSE_LAYOUT,
        "sidecar_template": "tcp/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz",
        "sidecar_arrays": {"ee_pose_world_wxyz": "float64 [public_length + 1, 14]",
                           "tcp_pose_world_wxyz": "float32 [public_length + 1, 14]"},
        "terminal_frame_preserved": True, "parquet_pose": "Current observation frame; final RAW-only terminal frame is in the sidecar.",
        "tcp_definition": "p_TCP=p_EE+R(q_wxyz)@[0.12,0,0]; q_TCP=q_EE",
        "target_sampling": GOAL_POLICY, "normalization_applied": False,
        "normalization_note": "TCP and joint-state statistics have different semantics. Position statistics may support later normalization; quaternions remain unit quaternions.",
        "source_mapping_report_sha256": mapping_hash,
    })
    return {"task": task, "episode_count": len(records),
            "public_frames": sum(item["public_length"] for item in records),
            "full_raw_frames": sum(item["raw_length"] for item in records),
            "episodes": records, "full_tcp": np.concatenate(full_poses, axis=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean"))
    parser.add_argument("--output-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean-TCP"))
    parser.add_argument("--mapping", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean.raw_alignment.json"))
    parser.add_argument("--task", help="Prepare one task only; use a separate smoke output directory.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    mapping = json.loads(args.mapping.read_text())
    if mapping["status"] != "passed" or mapping["matched_episode_count"] != 2500:
        raise ValueError("A complete passing 2,500-episode mapping is required.")
    if source_root != Path(mapping["public_root"]).resolve():
        raise ValueError("source-root must match the verified public dataset root.")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    tasks = [task for task in mapping["tasks"] if args.task is None or task["task"] == args.task]
    if not tasks or any(task["status"] != "passed" or len(task["entries"]) != 50 for task in tasks):
        raise ValueError("Expected complete, passing 50-episode task records.")
    output_root = args.output_root.expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an existing output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    mapping_hash = sha256_file(args.mapping)
    manifest = {
        "schema_version": 1, "status": "preparing", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root), "output_root": str(output_root),
        "mapping_report": str(args.mapping.resolve()), "mapping_report_sha256": mapping_hash,
        "source_modified": False, "original_columns_preserved": True,
        "state_column_unchanged": True, "added_column": TCP_COLUMN,
        "videos_shared_by_symlink": True, "metadata_is_private_copy": True,
        "step_caches_copied": False, "target_sampling": GOAL_POLICY,
        "fixed_random_targets_generated": False, "tasks": [], "errors": [],
    }
    report_path = output_root / "preparation_manifest.json"
    def save_manifest():
        temporary = report_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(report_path)
    save_manifest()
    start = time.perf_counter()
    full_tcp = []
    print(f"Preparing {len(tasks)} tasks into {output_root}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_task, task, source_root, output_root, mapping_hash): task["task"] for task in tasks}
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                result = future.result()
                full_tcp.append(result.pop("full_tcp"))
                manifest["tasks"].append(result)
                print(f"Completed {len(manifest['tasks'])}/{len(tasks)} tasks: {task_name}, {result['episode_count']} episodes", flush=True)
            except Exception as error:
                manifest["errors"].append({"task": task_name, "type": type(error).__name__, "message": str(error)})
                print(f"ERROR: {task_name}: {type(error).__name__}: {error}", flush=True)
            save_manifest()
    if full_tcp:
        all_tcp = np.concatenate(full_tcp, axis=0)
        write_json(output_root / "tcp_statistics.json", {
            "scope": "Diagnostic statistics over all full N+1 RAW training TCP sequences, including terminal frames.",
            "frame_count": len(all_tcp), "layout": POSE_LAYOUT, "position_indices": [0, 1, 2, 7, 8, 9],
            "quaternion_indices": [3, 4, 5, 6, 10, 11, 12, 13], "normalization_applied": False,
            "note": "Position statistics may be used by a future encoder. Do not apply joint-state normalization to TCP poses or componentwise normalization to unit quaternions.",
            "stats": stats(all_tcp),
        })
    manifest["tasks"].sort(key=lambda item: item["task"])
    manifest["task_count"] = len(manifest["tasks"])
    manifest["episode_count"] = sum(item["episode_count"] for item in manifest["tasks"])
    manifest["public_frame_count"] = sum(item["public_frames"] for item in manifest["tasks"])
    manifest["full_raw_frame_count"] = sum(item["full_raw_frames"] for item in manifest["tasks"])
    manifest["elapsed_seconds"] = time.perf_counter() - start
    manifest["status"] = "passed" if not manifest["errors"] and len(manifest["tasks"]) == len(tasks) else "failed"
    save_manifest()
    print(f"Final status: {manifest['status']}; tasks={manifest['task_count']}; episodes={manifest['episode_count']}", flush=True)
    if manifest["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
