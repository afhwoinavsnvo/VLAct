#!/usr/bin/env python3
"""Create a separate XYZW TCP-pose-state dataset without changing its source.

Both current-pose columns and every future TCP sidecar row receive the exact
same component permutation. No geometry is recomputed and no frame is sampled,
dropped, shifted, sign-flipped or normalized. Original EE WXYZ arrays remain
explicitly marked provenance, rather than training inputs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from examples.Robotwin.train_files.prepare_clean_tcp_dataset import GOAL_POLICY, sha256_file, stats

STATE, TCP, JOINT = "observation.state", "observation.tcp_pose", "observation.joint_state"
OLD_TCP_ARRAY, TCP_ARRAY, EE_ARRAY = "tcp_pose_world_wxyz", "tcp_pose_world_xyzw", "ee_pose_world_wxyz"
WXYZ_LAYOUT = [f"{arm}_{part}" for arm in ("left", "right")
               for part in ("x", "y", "z", "qw", "qx", "qy", "qz")]
POSE_LAYOUT = [f"{arm}_{part}" for arm in ("left", "right")
               for part in ("x", "y", "z", "qx", "qy", "qz", "qw")]
PERMUTATION = [0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 11, 12, 13, 10]
INVERSE_PERMUTATION = np.argsort(PERMUTATION).tolist()
TCP_DEFINITION = "p_TCP=p_EE+R(q_xyzw)@[0.12,0,0]; q_TCP=q_EE"
ARRAY_QUATERNION_ORDERS = {EE_ARRAY: "wxyz", TCP_ARRAY: "xyzw"}
STATISTIC_KEYS = {"mean", "std", "min", "max", "q01", "q99"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reorder_pose(values):
    """Copy raw float32 TCP values using an exactly reversible permutation."""
    values = np.asarray(values)
    require(values.dtype == np.float32 and values.ndim == 2 and values.shape[1] == 14,
            "TCP values must be a float32 [N,14] matrix")
    require(np.isfinite(values).all(), "TCP values must be finite")
    return values[:, PERMUTATION].copy()


def pose_column(table, name):
    column = table[name].combine_chunks()
    require(pa.types.is_fixed_size_list(column.type) and column.type.list_size == 14
            and column.type.value_type == pa.float32() and column.null_count == 0
            and column.values.null_count == 0, f"Expected non-null float32[14] {name}")
    values = column.values.to_numpy(zero_copy_only=False).reshape(-1, 14)
    require(np.isfinite(values).all(), f"Nonfinite pose values in {name}")
    return values


def convert_pose_table(table):
    """Change only the two pose arrays and add explicit Arrow pose metadata."""
    require(JOINT in table.column_names, "Source must already preserve observation.joint_state")
    current, duplicate = pose_column(table, STATE), pose_column(table, TCP)
    require(np.array_equal(current, duplicate), "Source current state and TCP column disagree")
    converted = table
    for name in (STATE, TCP):
        values = reorder_pose(pose_column(table, name))
        array = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), 14)
        converted = converted.set_column(table.schema.get_field_index(name), table.schema.field(name), array)
    metadata = dict(table.schema.metadata or {})
    metadata[b"tcp_pose_contract"] = json.dumps({
        "schema_version": 2, "features": [STATE, TCP], "quaternion_order": "xyzw",
        "layout": POSE_LAYOUT, "coordinate_frame": "world", "position_unit": "meters",
        "normalization_applied": False,
    }).encode("utf-8")
    # Hugging Face feature dtypes stay fixed float32[14]. Its metadata has no
    # component names; authoritative names also live in meta/info.json.
    return converted.replace_schema_metadata(metadata)


def reorder_statistics(record):
    """Permute componentwise statistics exactly, without changing counts."""
    result = deepcopy(record)
    for key in STATISTIC_KEYS:
        if key in result:
            values = np.asarray(result[key])
            require(values.shape == (14,) and np.isfinite(values).all(), f"Invalid pose statistic {key}")
            result[key] = values[PERMUTATION].tolist()
    require({"mean", "std", "min", "max"}.issubset(result), "Incomplete pose statistics")
    return result


def check_statistics(record, values, context):
    actual = stats(values, quantiles="q01" in record or "q99" in record, count="count" in record)
    for key, expected in actual.items():
        require(key in record and np.asarray(record[key]).shape == np.asarray(expected).shape
                and np.allclose(record[key], expected, rtol=1e-7, atol=1e-8),
                f"{context}: {key} statistics do not match pose values")


def convert_state_contract(source):
    result = deepcopy(source)
    result.update(schema_version=2, quaternion_order="xyzw", layout=list(POSE_LAYOUT),
                  tcp_definition=TCP_DEFINITION, target_pose_array=TCP_ARRAY)
    return result


def convert_tcp_contract(source):
    result = deepcopy(source)
    require(OLD_TCP_ARRAY in result["sidecar_arrays"] and TCP_ARRAY not in result["sidecar_arrays"],
            "Expected the original WXYZ TCP sidecar schema")
    array_description = result["sidecar_arrays"].pop(OLD_TCP_ARRAY)
    result["sidecar_arrays"][TCP_ARRAY] = array_description
    result.update(schema_version=2, quaternion_order="xyzw", layout=list(POSE_LAYOUT),
                  tcp_definition=TCP_DEFINITION, target_pose_array=TCP_ARRAY,
                  array_quaternion_orders=deepcopy(ARRAY_QUATERNION_ORDERS),
                  provenance_arrays=[EE_ARRAY])
    return result


def convert_sidecar_arrays(arrays, *, source_path, source_sha256):
    require(OLD_TCP_ARRAY in arrays and TCP_ARRAY not in arrays and EE_ARRAY in arrays,
            "Expected WXYZ TCP and provenance EE arrays")
    tcp = arrays[OLD_TCP_ARRAY]
    ee = arrays[EE_ARRAY]
    require(ee.dtype == np.float64 and ee.shape == tcp.shape and np.isfinite(ee).all(),
            "Invalid provenance EE float64 matrix")
    metadata = json.loads(str(arrays["metadata_json"]))
    require(metadata.get("quaternion_order") == "wxyz" and metadata.get("layout") == WXYZ_LAYOUT
            and metadata.get("target_sampling") == GOAL_POLICY, "Invalid source sidecar contract")
    result = {key: value.copy() for key, value in arrays.items() if key not in (OLD_TCP_ARRAY, "metadata_json")}
    result[TCP_ARRAY] = reorder_pose(tcp)
    metadata.update(quaternion_order="xyzw", layout=list(POSE_LAYOUT), tcp_definition=TCP_DEFINITION,
                    target_pose_array=TCP_ARRAY, array_quaternion_orders=deepcopy(ARRAY_QUATERNION_ORDERS),
                    provenance_arrays=[EE_ARRAY], source_pose_state_sidecar=str(source_path),
                    source_pose_state_sidecar_sha256=source_sha256)
    result["metadata_json"] = np.array(json.dumps(metadata, allow_nan=False))
    return result


def _relative_data_path(info, eid):
    path = Path(info["data_path"].format(episode_index=eid, episode_chunk=eid // int(info["chunks_size"])))
    require(not path.is_absolute() and ".." not in path.parts and path.parts[0] == "data", "Unsafe data path")
    return path


def _relative_sidecar_path(contract, info, eid):
    path = Path(contract["sidecar_template"].format(episode_index=eid, episode_chunk=eid // int(info["chunks_size"])))
    require(not path.is_absolute() and ".." not in path.parts and path.parts[0] == "tcp", "Unsafe sidecar path")
    return path


def inspect_source(source_root):
    """Read metadata and row counts; never create an output dataset."""
    manifest_path = source_root / "preparation_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "passed" and manifest.get("state_representation") == "tcp_pose"
            and manifest.get("target_sampling") == GOAL_POLICY, "Expected a passed Clean-last PoseState source")
    state_contract = manifest.get("state_contract", {})
    require(state_contract.get("schema_version") == 1 and state_contract.get("quaternion_order") == "wxyz"
            and state_contract.get("layout") == WXYZ_LAYOUT and state_contract.get("normalization_applied") is False,
            "Expected a raw WXYZ PoseState contract; repeat migrations are forbidden")
    tasks = manifest["tasks"]
    require(len(tasks) == 50 and len({task["task"] for task in tasks}) == 50, "Expected all fifty tasks")
    count = frames = 0
    for task in tasks:
        name = task["task"]
        require(Path(name).name == name and name not in (".", ".."), "Unsafe task name")
        task_dir = source_root / name
        info = read_json(task_dir / "meta/info.json")
        tcp_contract = read_json(task_dir / "meta/tcp_pose.json")
        require(read_json(task_dir / "meta/state_pose.json") == state_contract, f"{name}: state contract differs")
        require(tcp_contract.get("quaternion_order") == "wxyz" and tcp_contract.get("layout") == WXYZ_LAYOUT
                and tcp_contract.get("target_sampling") == GOAL_POLICY, f"{name}: invalid TCP contract")
        require((task_dir / "videos").resolve(strict=True).is_dir(), f"{name}: missing original video directory")
        episodes = task["episodes"]
        require(len(episodes) == 50 and {int(ep["public_episode_id"]) for ep in episodes} == set(range(50)),
                f"{name}: expected all fifty episodes")
        for episode in episodes:
            eid, length = int(episode["public_episode_id"]), int(episode["public_length"])
            parquet = task_dir / _relative_data_path(info, eid)
            require(pq.ParquetFile(parquet).metadata.num_rows == length, f"{name}/{eid}: incorrect row count")
            require((task_dir / _relative_sidecar_path(tcp_contract, info, eid)).is_file(), f"{name}/{eid}: missing sidecar")
            frames += length
            count += 1
    require(count == 2500 and frames == 549787 and manifest.get("public_frame_count") == frames
            and manifest.get("full_raw_frame_count") == frames + count, "Source corpus is incomplete")
    return manifest, {
        "status": "inputs_validated", "task_count": 50, "episode_count": count, "public_frame_count": frames,
        "source_pose_state_manifest_sha256": sha256_file(manifest_path),
        "source_root": str(source_root), "source_order": "wxyz", "target_order": "xyzw",
        "scope": "Structural validation only. Conversion additionally verifies hashes, arrays and unchanged columns.",
    }


def _write_json(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _copy_regular(source, destination):
    require(source.is_file() and not source.is_symlink(), f"Expected a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)


def convert_task(task, source_root, output_root, state_contract):
    name = task["task"]
    source, target = source_root / name, output_root / name
    info = read_json(source / "meta/info.json")
    old_contract = read_json(source / "meta/tcp_pose.json")
    target.mkdir(exist_ok=False)
    (target / "meta").mkdir()
    (target / "videos").symlink_to((source / "videos").resolve(strict=True), target_is_directory=True)
    rewritten = {"info.json", "tcp_pose.json", "state_pose.json", "stats_gr00t.json", "episodes_stats.jsonl"}
    for path in (source / "meta").iterdir():
        if path.is_file() and path.suffix in (".json", ".jsonl") and path.name not in rewritten and not path.name.startswith("."):
            _copy_regular(path, target / "meta" / path.name)
    episode_stats = {int(row["episode_index"]): row for row in read_jsonl(source / "meta/episodes_stats.jsonl")}
    aggregate = read_json(source / "meta/stats_gr00t.json")
    records, task_tcp, full_tcp = [], [], []
    for episode in sorted(task["episodes"], key=lambda ep: int(ep["public_episode_id"])):
        eid, length = int(episode["public_episode_id"]), int(episode["public_length"])
        relative = _relative_data_path(info, eid)
        parquet, target_parquet = source / relative, target / relative
        source_hash = sha256_file(parquet)
        require(source_hash == episode["parquet_sha256"], f"{name}/{eid}: source Parquet hash changed")
        table = pq.read_table(parquet)
        old_values = pose_column(table, STATE)
        converted = convert_pose_table(table)
        new_values = pose_column(converted, STATE)
        require(np.array_equal(new_values[:, INVERSE_PERMUTATION], old_values), f"{name}/{eid}: irreversible permutation")
        target_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(converted, target_parquet, compression="snappy")
        actual = pq.read_table(target_parquet)
        unchanged = [column for column in table.column_names if column not in (STATE, TCP)]
        require(actual.select(unchanged).equals(table.select(unchanged), check_metadata=False),
                f"{name}/{eid}: an action, joint backup, ID or other non-pose column changed")
        for column in (STATE, TCP):
            require(actual[column].equals(converted[column]), f"{name}/{eid}: pose Parquet roundtrip changed values")
        sidecar_relative = _relative_sidecar_path(old_contract, info, eid)
        sidecar, target_sidecar = source / sidecar_relative, target / sidecar_relative
        sidecar_hash = sha256_file(sidecar)
        require(sidecar_hash == episode["sidecar_sha256"], f"{name}/{eid}: source sidecar hash changed")
        with np.load(sidecar, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        require(arrays[OLD_TCP_ARRAY].shape == (length + 1, 14)
                and np.array_equal(arrays[OLD_TCP_ARRAY][:-1], old_values), f"{name}/{eid}: shifted current state")
        migrated = convert_sidecar_arrays(arrays, source_path=sidecar, source_sha256=sidecar_hash)
        require(np.array_equal(migrated[TCP_ARRAY][:-1], new_values), f"{name}/{eid}: migrated target/state mismatch")
        target_sidecar.parent.mkdir(parents=True, exist_ok=True)
        with target_sidecar.open("xb") as stream:
            np.savez_compressed(stream, **migrated)
        with np.load(target_sidecar, allow_pickle=False) as data:
            require(OLD_TCP_ARRAY not in data.files and set(data.files) == set(migrated), f"{name}/{eid}: wrong sidecar keys")
            for key, values in migrated.items():
                require(np.array_equal(data[key], values), f"{name}/{eid}: sidecar roundtrip changed {key}")
        for column in (STATE, TCP):
            old_stats = episode_stats[eid]["stats"][column]
            check_statistics(old_stats, old_values, f"{name}/{eid} source {column}")
            episode_stats[eid]["stats"][column] = reorder_statistics(old_stats)
            check_statistics(episode_stats[eid]["stats"][column], new_values, f"{name}/{eid} new {column}")
        task_tcp.append(new_values)
        full_tcp.append(migrated[TCP_ARRAY])
        record = deepcopy(episode)
        record.update(parquet=str(target_parquet), parquet_sha256=sha256_file(target_parquet),
                      sidecar=str(target_sidecar), sidecar_sha256=sha256_file(target_sidecar),
                      source_pose_state_parquet=str(parquet), source_pose_state_parquet_sha256=source_hash,
                      source_pose_state_sidecar=str(sidecar), source_pose_state_sidecar_sha256=sidecar_hash,
                      original_non_state_columns_and_types_preserved=False,
                      original_non_pose_columns_and_types_preserved=True,
                      changed_columns=[STATE, TCP], quaternion_order="xyzw")
        records.append(record)
    values = np.concatenate(task_tcp)
    for column in (STATE, TCP):
        aggregate[column] = reorder_statistics(aggregate[column])
        check_statistics(aggregate[column], values, f"{name} aggregate {column}")
        info["features"][column]["names"] = list(POSE_LAYOUT)
    _write_json(target / "meta/info.json", info)
    _write_json(target / "meta/tcp_pose.json", convert_tcp_contract(old_contract))
    _write_json(target / "meta/state_pose.json", state_contract)
    _write_json(target / "meta/stats_gr00t.json", aggregate)
    with (target / "meta/episodes_stats.jsonl").open("x", encoding="utf-8") as stream:
        for eid in sorted(episode_stats):
            stream.write(json.dumps(episode_stats[eid], allow_nan=False) + "\n")
    result = deepcopy(task)
    result.update(episodes=records, quaternion_order="xyzw")
    return result, full_tcp


def prepare_dataset(source_root, output_root, *, validate_only=False):
    source_root = Path(source_root).resolve(strict=True)
    output_root = Path(output_root).expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {output_root}")
    require(not output_root.resolve().is_relative_to(source_root), "Output must be separate from source")
    source_manifest, validation = inspect_source(source_root)
    if validate_only:
        return {**validation, "validate_only": True, "output_root": str(output_root), "output_created": False}
    start = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    manifest = deepcopy(source_manifest)
    manifest.update(status="preparing", created_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_root=str(source_root), source_pose_state_root=str(source_root), output_root=str(output_root),
                    source_pose_state_manifest_sha256=validation["source_pose_state_manifest_sha256"],
                    state_contract=convert_state_contract(source_manifest["state_contract"]),
                    quaternion_order="xyzw", target_pose_array=TCP_ARRAY,
                    array_quaternion_orders=deepcopy(ARRAY_QUATERNION_ORDERS),
                    original_non_state_columns_preserved=False, original_non_pose_columns_preserved=True,
                    changed_columns=[STATE, TCP], source_modified=False, training_ready_claimed=False,
                    orientation_migration={"schema_version": 1, "source_order": "wxyz", "target_order": "xyzw",
                                           "permutation": PERMUTATION, "geometry_recomputed": False,
                                           "quaternion_normalization_applied": False, "quaternion_sign_changed": False,
                                           "source_tcp_definition": source_manifest["state_contract"]["tcp_definition"]},
                    tasks=[], errors=[])
    manifest_path = output_root / "preparation_manifest.json"

    def save_manifest():
        temporary = manifest_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    save_manifest()
    try:
        full_tcp = []
        for task in sorted(source_manifest["tasks"], key=lambda task: task["task"]):
            result, arrays = convert_task(task, source_root, output_root, manifest["state_contract"])
            manifest["tasks"].append(result)
            full_tcp.extend(arrays)
            save_manifest()
            print(f"Converted {len(manifest['tasks'])}/50 tasks: {task['task']}", file=sys.stderr, flush=True)
        diagnostics = read_json(source_root / "tcp_statistics.json")
        require(diagnostics.get("layout") == WXYZ_LAYOUT, "Unexpected source diagnostic layout")
        diagnostics.update(layout=list(POSE_LAYOUT), quaternion_order="xyzw", target_pose_array=TCP_ARRAY)
        diagnostics["stats"] = reorder_statistics(diagnostics["stats"])
        check_statistics(diagnostics["stats"], np.concatenate(full_tcp), "Full RAW TCP diagnostics")
        _write_json(output_root / "tcp_statistics.json", diagnostics)
        _copy_regular(source_root / "correspondence.csv", output_root / "correspondence.csv")
        require(sha256_file(source_root / "preparation_manifest.json") == validation["source_pose_state_manifest_sha256"],
                "Source manifest changed during conversion")
        manifest.update(status="passed", source_manifest_unchanged=True, elapsed_seconds=time.perf_counter() - start)
        save_manifest()
        (output_root / "README.md").write_text(
            "# RoboTwin-Clean-TCP-PoseState-XYZW\n\n"
            "Current observation.state and observation.tcp_pose use raw float32[14]: "
            "[left x,y,z,qx,qy,qz,qw, right x,y,z,qx,qy,qz,qw], world coordinates and meters. "
            "Future target_tcp_pose is sampled from the matching tcp_pose_world_xyzw sidecar. "
            "The original EE ee_pose_world_wxyz array is unchanged provenance with its own declared order.\n\n"
            "This is an exact component permutation of the WXYZ PoseState dataset. No TCP geometry, "
            "quaternion sign, normalization, frame, action, joint-state backup, task/episode mapping or "
            "target-frame policy is changed. All fifty tasks and 2,500 episodes remain complete. "
            "Videos link to the same original Clean directories. Cache/pickle/notebook files are excluded.\n\n"
            "State/TCP contracts use schema 2 and explicitly require XYZW-aware loaders, serving and "
            "newly trained checkpoints. The target-frame sampling policy independently remains schema 2 "
            "(complete [t+64,t+96) window, otherwise the final Clean observation). Previous WXYZ "
            "checkpoints continue to require the original WXYZ dataset. Source manifest, Parquet and "
            "sidecar SHA256 values remain recorded for provenance; no source is modified.\n",
            encoding="utf-8")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
        save_manifest()
        raise
    return {"status": "passed", "output_root": str(output_root), "quaternion_order": "xyzw",
            "task_count": 50, "episode_count": 2500, "public_frame_count": 549787,
            "source_manifest_unchanged": True, "scope": "Dataset migration only; no training, evaluation or upload."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean-TCP-PoseState"))
    parser.add_argument("--output-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean-TCP-PoseState-XYZW"))
    parser.add_argument("--validate-only", action="store_true", help="Read source structure without creating output")
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.source_root, args.output_root, validate_only=args.validate_only), indent=2))


if __name__ == "__main__":
    main()
