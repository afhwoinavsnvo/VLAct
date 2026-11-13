#!/usr/bin/env python3
"""Create a separate current-TCP state dataset without changing its sources.

``observation.state`` becomes the same frame's world-frame TCP pose. The
original joint state is preserved in ``observation.joint_state``; actions,
future-target TCP sidecars, observation order and video references are retained.
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

from examples.Robotwin.train_files.prepare_clean_tcp_dataset import (
    GOAL_POLICY, POSE_LAYOUT, sha256_file, stats,
)

STATE = "observation.state"
TCP = "observation.tcp_pose"
JOINT = "observation.joint_state"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state_contract(tcp_definition):
    return {
        "schema_version": 1, "state_representation": "tcp_pose", "feature": STATE,
        "backup_joint_feature": JOINT, "source_feature": TCP,
        "dtype": "float32", "shape": [14], "current_frame": True,
        "normalization_applied": False, "coordinate_frame": "world",
        "position_unit": "meters", "quaternion_order": "wxyz",
        "layout": list(POSE_LAYOUT), "tcp_definition": tcp_definition,
    }


def column_array(table, name):
    column = table[name].combine_chunks()
    if (not pa.types.is_fixed_size_list(column.type) or column.type.list_size != 14
            or column.null_count or column.values.null_count):
        raise ValueError(f"Expected a non-null fixed-size 14D column: {name}")
    return column.values.to_numpy(zero_copy_only=False).reshape(-1, 14)


def build_pose_state_table(table):
    """Return a new Arrow table; do not mutate the original table or arrays."""
    if JOINT in table.column_names:
        raise ValueError(f"Input already contains the joint-state backup {JOINT}")
    tcp = column_array(table, TCP)
    if table.schema.field(TCP).type.value_type != pa.float32() or not np.isfinite(tcp).all():
        raise ValueError("Current TCP pose must contain finite float32[14] values")
    joint_field = table.schema.field(STATE)
    joint_column = table[STATE]
    tcp_field = table.schema.field(TCP)
    updated = table.set_column(
        table.schema.get_field_index(STATE),
        pa.field(STATE, tcp_field.type, nullable=tcp_field.nullable, metadata=tcp_field.metadata),
        table[TCP],
    )
    updated = updated.append_column(
        pa.field(JOINT, joint_field.type, nullable=joint_field.nullable, metadata=joint_field.metadata),
        joint_column,
    )
    metadata = dict(table.schema.metadata or {})
    hf = json.loads(metadata.get(b"huggingface", b"{}"))
    features = hf.setdefault("info", {}).setdefault("features", {})
    old_state_feature = deepcopy(features.get(STATE, {
        "feature": {"dtype": "float32", "_type": "Value"}, "length": 14, "_type": "List",
    }))
    features[STATE] = deepcopy(features.get(TCP, {
        "feature": {"dtype": "float32", "_type": "Value"}, "length": 14, "_type": "List",
    }))
    features[JOINT] = old_state_feature
    metadata[b"huggingface"] = json.dumps(hf).encode("utf-8")
    return updated.replace_schema_metadata(metadata)


def _statistics_match(declared, actual, label):
    for key, expected in actual.items():
        if key not in declared:
            raise ValueError(f"Missing {label} statistic {key}")
        values = np.asarray(declared[key])
        if (values.shape != np.asarray(expected).shape
                or not np.allclose(values, expected, rtol=1e-7, atol=1e-8, equal_nan=False)):
            raise ValueError(f"Source {label} statistic {key} disagrees with current-frame TCP values")


def _episode_path(task_dir, info, episode_id):
    relative = Path(info["data_path"].format(
        episode_index=episode_id, episode_chunk=episode_id // int(info["chunks_size"]),
    ))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "data":
        raise ValueError(f"Invalid episode path in {task_dir}: {relative}")
    return relative


def _sidecar_path(task_dir, contract, info, episode_id):
    relative = Path(contract["sidecar_template"].format(
        episode_index=episode_id, episode_chunk=episode_id // int(info["chunks_size"]),
    ))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "tcp":
        raise ValueError(f"Invalid TCP sidecar path in {task_dir}: {relative}")
    return relative


def inspect_sources(source_tcp_root, clean_root):
    """Read structural metadata only; this function never creates output files."""
    source_tcp_root, clean_root = Path(source_tcp_root).resolve(strict=True), Path(clean_root).resolve(strict=True)
    manifest_path = source_tcp_root / "preparation_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "passed" or manifest.get("target_sampling") != GOAL_POLICY:
        raise ValueError("Source TCP dataset must use the passing Clean-last target policy schema 2")
    if manifest.get("state_representation") == "tcp_pose" or manifest.get("state_column_unchanged") is not True:
        raise ValueError("Expected the original joint-state TCP source, not an already converted pose-state dataset")
    if Path(manifest["source_root"]).resolve(strict=True) != clean_root:
        raise ValueError("clean-root must identify the verified original Clean source recorded in the TCP manifest")
    tasks = manifest.get("tasks", [])
    names = [task["task"] for task in tasks]
    if len(names) != 50 or len(set(names)) != 50:
        raise ValueError("Expected all fifty source tasks")
    total_episodes = total_frames = 0
    contract_definition = None
    for task in tasks:
        name = task["task"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid task name {name!r}")
        source_dir, clean_dir = source_tcp_root / name, clean_root / name
        source_info, clean_info = read_json(source_dir / "meta/info.json"), read_json(clean_dir / "meta/info.json")
        tcp_contract = read_json(source_dir / "meta/tcp_pose.json")
        if tcp_contract.get("target_sampling") != GOAL_POLICY:
            raise ValueError(f"Task {name} does not use source target policy schema 2")
        if contract_definition is None:
            contract_definition = tcp_contract["tcp_definition"]
        elif tcp_contract["tcp_definition"] != contract_definition:
            raise ValueError("Task TCP definitions disagree")
        source_episodes = read_jsonl(source_dir / "meta/episodes.jsonl")
        clean_episodes = read_jsonl(clean_dir / "meta/episodes.jsonl")
        if source_episodes != clean_episodes:
            raise ValueError(f"Source and original Clean episode metadata disagree for {name}")
        episodes = task["episodes"]
        ids = [int(episode["public_episode_id"]) for episode in episodes]
        if len(ids) != 50 or set(ids) != set(range(50)):
            raise ValueError(f"Task {name} must retain all fifty public episodes")
        declared_lengths = {int(ep["episode_index"]): int(ep["length"]) for ep in clean_episodes}
        for episode in episodes:
            eid, length = int(episode["public_episode_id"]), int(episode["public_length"])
            if declared_lengths.get(eid) != length:
                raise ValueError(f"Manifest/Clean episode length mismatch for {name}/{eid}")
            src = source_dir / _episode_path(source_dir, source_info, eid)
            original = clean_dir / _episode_path(clean_dir, clean_info, eid)
            for parquet in (src, original):
                if pq.ParquetFile(parquet).metadata.num_rows != length:
                    raise ValueError(f"Parquet row count differs for {parquet}")
            sidecar = source_dir / _sidecar_path(source_dir, tcp_contract, source_info, eid)
            if not sidecar.is_file():
                raise FileNotFoundError(sidecar)
            total_episodes += 1
            total_frames += length
        if (source_dir / "videos").resolve(strict=True) != (clean_dir / "videos").resolve(strict=True):
            raise ValueError(f"Source and Clean video references disagree for {name}")
    if total_episodes != 2500 or total_frames != manifest.get("public_frame_count"):
        raise ValueError("Source TCP manifest does not cover the complete original Clean corpus")
    for name in ("tcp_statistics.json", "correspondence.csv"):
        if not (source_tcp_root / name).is_file():
            raise FileNotFoundError(source_tcp_root / name)
    return manifest, {
        "status": "inputs_validated", "source_tcp_root": str(source_tcp_root), "clean_root": str(clean_root),
        "source_tcp_manifest_sha256": sha256_file(manifest_path), "task_count": 50,
        "episode_count": total_episodes, "public_frame_count": total_frames,
        "state_contract": state_contract(contract_definition),
        "scope": "Structural input validation only; conversion verifies every original column and state value.",
    }


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _copy_regular(source, destination):
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Expected a regular source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)


def convert_task(task, source_root, clean_root, output_root, contract):
    name = task["task"]
    source_dir, clean_dir, target_dir = source_root / name, clean_root / name, output_root / name
    info, clean_info = read_json(source_dir / "meta/info.json"), read_json(clean_dir / "meta/info.json")
    tcp_contract = read_json(source_dir / "meta/tcp_pose.json")
    target_dir.mkdir(exist_ok=False)
    (target_dir / "videos").symlink_to((clean_dir / "videos").resolve(strict=True), target_is_directory=True)
    (target_dir / "meta").mkdir()
    rewritten = {"info.json", "modality.json", "stats_gr00t.json", "episodes_stats.jsonl", "state_pose.json"}
    for path in (source_dir / "meta").iterdir():
        if path.is_file() and path.suffix in (".json", ".jsonl") and path.name not in rewritten and not path.name.startswith("."):
            _copy_regular(path, target_dir / "meta" / path.name)
    episode_statistics = {int(row["episode_index"]): row for row in read_jsonl(source_dir / "meta/episodes_stats.jsonl")}
    aggregate = read_json(source_dir / "meta/stats_gr00t.json")
    all_tcp, episode_records = [], []
    for episode in sorted(task["episodes"], key=lambda row: int(row["public_episode_id"])):
        eid, length = int(episode["public_episode_id"]), int(episode["public_length"])
        relative = _episode_path(source_dir, info, eid)
        source_parquet = source_dir / relative
        clean_parquet = clean_dir / _episode_path(clean_dir, clean_info, eid)
        source_hash = sha256_file(source_parquet)
        clean_hash = sha256_file(clean_parquet)
        if source_hash != episode["parquet_sha256"] or clean_hash != episode["source_parquet_sha256"]:
            raise ValueError(f"Verified source Parquet hashes changed for {name}/{eid}")
        table, clean = pq.read_table(source_parquet), pq.read_table(clean_parquet)
        if not table.select(clean.column_names).equals(clean, check_metadata=False):
            raise ValueError(f"Source TCP original columns differ from original Clean for {name}/{eid}")
        if not table.select(clean.column_names).schema.remove_metadata().equals(clean.schema.remove_metadata()):
            raise ValueError(f"Source TCP original column types differ from original Clean for {name}/{eid}")
        if not np.array_equal(table["frame_index"].to_numpy(), np.arange(length)):
            raise ValueError(f"Frame indices are not contiguous for {name}/{eid}")
        if not np.all(table["episode_index"].to_numpy() == eid):
            raise ValueError(f"Episode IDs disagree for {name}/{eid}")
        tcp = column_array(table, TCP)
        sidecar_relative = _sidecar_path(source_dir, tcp_contract, info, eid)
        source_sidecar = source_dir / sidecar_relative
        sidecar_hash = sha256_file(source_sidecar)
        if sidecar_hash != episode["sidecar_sha256"]:
            raise ValueError(f"Verified future-target sidecar changed for {name}/{eid}")
        with np.load(source_sidecar, allow_pickle=False) as arrays:
            sequence = arrays["tcp_pose_world_wxyz"]
            details = json.loads(str(arrays["metadata_json"]))
        if (sequence.shape != (length + 1, 14) or details.get("target_sampling") != GOAL_POLICY
                or not np.array_equal(sequence[:-1], tcp)):
            raise ValueError(f"Current TCP rows do not match the same episode's sidecar for {name}/{eid}")
        converted = build_pose_state_table(table)
        target_parquet = target_dir / relative
        target_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(converted, target_parquet, compression="snappy")
        actual = pq.read_table(target_parquet)
        unchanged_columns = [column for column in table.column_names if column != STATE]
        if not actual.select(unchanged_columns).equals(table.select(unchanged_columns), check_metadata=False):
            raise ValueError(f"A non-state source column changed for {name}/{eid}")
        if not actual[JOINT].equals(clean[STATE]) or not actual[STATE].equals(table[TCP]):
            raise ValueError(f"TCP state or original joint-state backup is not frame-aligned for {name}/{eid}")
        _copy_regular(source_sidecar, target_dir / sidecar_relative)
        if sha256_file(target_dir / sidecar_relative) != sidecar_hash:
            raise ValueError(f"Copied sidecar differs for {name}/{eid}")
        ep_stats = episode_statistics[eid]["stats"]
        current_stats = stats(tcp, quantiles=False, count=True)
        _statistics_match(ep_stats[TCP], current_stats, f"{name}/{eid} TCP")
        ep_stats[JOINT] = deepcopy(ep_stats[STATE])
        ep_stats[STATE] = current_stats
        all_tcp.append(tcp)
        record = deepcopy(episode)
        record.update({
            "parquet": str(target_parquet), "parquet_sha256": sha256_file(target_parquet),
            "source_tcp_parquet": str(source_parquet), "source_tcp_parquet_sha256": source_hash,
            "source_parquet_sha256": clean_hash,
            "sidecar": str(target_dir / sidecar_relative), "sidecar_sha256": sidecar_hash,
            "original_columns_and_types_preserved": False,
            "original_non_state_columns_and_types_preserved": True,
            "clean_joint_state_backed_up": True, "current_tcp_state_frame_aligned": True,
        })
        episode_records.append(record)
    current_stats = stats(np.concatenate(all_tcp, axis=0))
    _statistics_match(aggregate[TCP], current_stats, f"{name} aggregate TCP")
    aggregate[JOINT] = deepcopy(aggregate[STATE])
    aggregate[STATE] = current_stats
    joint_feature = deepcopy(info["features"][STATE])
    info["features"][STATE] = {"dtype": "float32", "shape": [14], "names": list(POSE_LAYOUT)}
    info["features"][JOINT] = joint_feature
    modality = read_json(source_dir / "meta/modality.json")
    modality["state"] = {"tcp_pose": {"start": 0, "end": 14, "original_key": STATE}}
    _write_json(target_dir / "meta/info.json", info)
    _write_json(target_dir / "meta/modality.json", modality)
    _write_json(target_dir / "meta/stats_gr00t.json", aggregate)
    _write_json(target_dir / "meta/state_pose.json", contract)
    with (target_dir / "meta/episodes_stats.jsonl").open("x", encoding="utf-8") as stream:
        for eid in sorted(episode_statistics):
            stream.write(json.dumps(episode_statistics[eid], allow_nan=False) + "\n")
    output_task = deepcopy(task)
    output_task["episodes"] = episode_records
    output_task["state_representation"] = "tcp_pose"
    return output_task


def prepare_dataset(source_tcp_root, clean_root, output_root, *, validate_only=False):
    source_root, clean_root = Path(source_tcp_root).resolve(strict=True), Path(clean_root).resolve(strict=True)
    output_root = Path(output_root).expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {output_root}")
    resolved_output = output_root.resolve()
    if resolved_output.is_relative_to(source_root) or resolved_output.is_relative_to(clean_root):
        raise ValueError("Output must be separate from both source datasets")
    source_manifest, validation = inspect_sources(source_root, clean_root)
    if validate_only:
        return {**validation, "validate_only": True, "output_root": str(output_root), "output_created": False}
    start = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    manifest = deepcopy(source_manifest)
    manifest.update({
        "status": "preparing", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root), "source_tcp_root": str(source_root), "clean_root": str(clean_root),
        "output_root": str(output_root), "source_tcp_manifest_sha256": validation["source_tcp_manifest_sha256"],
        "state_representation": "tcp_pose", "state_contract": validation["state_contract"],
        "source_modified": False, "original_columns_preserved": False, "state_column_unchanged": False,
        "original_non_state_columns_preserved": True, "original_joint_state_backed_up": True,
        "videos_shared_by_symlink": True, "metadata_is_private_copy": True, "step_caches_copied": False,
        "training_ready_claimed": False, "tasks": [], "errors": [],
    })
    manifest_path = output_root / "preparation_manifest.json"

    def save_manifest():
        temporary = manifest_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(manifest_path)

    save_manifest()
    try:
        for task in sorted(source_manifest["tasks"], key=lambda item: item["task"]):
            manifest["tasks"].append(convert_task(task, source_root, clean_root, output_root, validation["state_contract"]))
            save_manifest()
            print(f"Converted {len(manifest['tasks'])}/50 tasks: {task['task']}", file=sys.stderr, flush=True)
        for name in ("tcp_statistics.json", "correspondence.csv"):
            _copy_regular(source_root / name, output_root / name)
        if sha256_file(source_root / "preparation_manifest.json") != validation["source_tcp_manifest_sha256"]:
            raise ValueError("Source TCP manifest changed during conversion")
        manifest.update({
            "status": "passed", "task_count": len(manifest["tasks"]),
            "episode_count": sum(len(task["episodes"]) for task in manifest["tasks"]),
            "public_frame_count": sum(task["public_frames"] for task in manifest["tasks"]),
            "source_manifest_unchanged": True, "elapsed_seconds": time.perf_counter() - start,
        })
        save_manifest()
        (output_root / "README.md").write_text(
            "# RoboTwin-Clean-TCP-PoseState\n\n"
            "This separate dataset replaces observation.state with the same frame's raw current TCP pose "
            "(float32[14], left7 then right7, world coordinates, meters, wxyz). The original joint state is "
            "retained as observation.joint_state. observation.tcp_pose, actions, language, all frame/episode "
            "indices and the complete future-target sidecars are preserved. Future target_tcp_pose remains "
            "a separate dynamically sampled field using the original Clean-last schema-2 policy.\n\n"
            "The modality state key is state.tcp_pose, with no joint/gripper normalization, wrapping or "
            "binarization. Use a TCP-aware DataConfig; the old joint-state DataConfig is not compatible. "
            "No training or model-compatibility claim is made by this conversion. Videos are shared by "
            "links to the original Clean video directories. Cache/pickle/notebook files are not copied.\n\n"
            "preparation_manifest.json records the new Parquet paths/hashes, source TCP hashes and unchanged "
            "original Clean hashes for task/episode/frame correspondence. Sidecar provenance paths refer "
            "to the original preparation environment and are retained unchanged.\n",
            encoding="utf-8",
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
        save_manifest()
        raise
    return {
        "status": "passed", "output_root": str(output_root), "state_representation": "tcp_pose",
        "task_count": manifest["task_count"], "episode_count": manifest["episode_count"],
        "public_frame_count": manifest["public_frame_count"], "source_manifest_unchanged": True,
        "scope": "Converted state values and original Clean correspondence; no training, evaluation or upload performed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tcp-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean-TCP"))
    parser.add_argument("--clean-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean"))
    parser.add_argument("--output-root", type=Path, default=Path("/scratch/jliu3375/datasets/RoboTwin-Clean-TCP-PoseState"))
    parser.add_argument("--validate-only", action="store_true", help="Read input metadata/structure without creating output")
    args = parser.parse_args()
    result = prepare_dataset(args.source_tcp_root, args.clean_root, args.output_root, validate_only=args.validate_only)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
