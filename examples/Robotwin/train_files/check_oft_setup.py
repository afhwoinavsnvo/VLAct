#!/usr/bin/env python3
"""Validate the frozen-VLM OFT state/TCP training setup without loading a model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from examples.Robotwin.train_files.prepare_clean_tcp_dataset import GOAL_POLICY, POSE_LAYOUT


def _sections(config):
    framework = config.get("framework", {})
    data = config.get("datasets", {}).get("vla_data", {})
    return framework, framework.get("action_model", {}), framework.get("conditioning", {}), data, config.get("trainer", {})


def _boolean(mapping, key, prefix, issues, *, expected=None):
    value = mapping.get(key, False)
    if type(value) is not bool:
        issues.append(f"{prefix}.{key} must be a boolean")
        return None
    if expected is not None and value is not expected:
        issues.append(f"{prefix}.{key} must be {str(expected).lower()}")
    return value


def _integer_is(mapping, key, expected, prefix, issues):
    value = mapping.get(key)
    if type(value) is not int or value != expected:
        issues.append(f"{prefix}.{key} must be {expected}")


def check_training_config(config) -> list[str]:
    """Check only configuration semantics; no data or checkpoint files are read."""
    framework, action, conditioning, data, trainer = _sections(config)
    issues = []
    if framework.get("name") != "QwenOFT":
        issues.append("framework.name must be QwenOFT; PI configurations are not supported by this launcher")
    for key, expected in (("action_dim", 14), ("action_horizon", 32),
                          ("future_action_window_size", 31), ("past_action_window_size", 0)):
        _integer_is(action, key, expected, "framework.action_model", issues)
    for key in ("enabled", "freeze_vlm"):
        _boolean(conditioning, key, "framework.conditioning", issues, expected=True)
    for key in ("include_state", "include_tcp"):
        model_value = _boolean(conditioning, key, "framework.conditioning", issues)
        data_value = _boolean(data, key, "datasets.vla_data", issues)
        if model_value is not None and data_value is not None and model_value != data_value:
            issues.append(f"Model and dataset {key} switches disagree")
    for key in ("state_dim", "tcp_dim"):
        _integer_is(conditioning, key, 14, "framework.conditioning", issues)
    _boolean(action, "use_time_budget", "framework.action_model", issues, expected=False)
    _boolean(data, "include_time_budget", "datasets.vla_data", issues, expected=False)
    _boolean(data, "include_target_pose", "datasets.vla_data", issues, expected=False)
    if "use_target_tcp" in data:
        legacy = _boolean(data, "use_target_tcp", "datasets.vla_data", issues)
        if legacy is not None and type(data.get("include_tcp")) is bool and legacy != data["include_tcp"]:
            issues.append("datasets.vla_data.include_tcp and use_target_tcp explicitly disagree")
    if data.get("data_mix") != "robotwin_wrap_32":
        issues.append("datasets.vla_data.data_mix must be robotwin_wrap_32")
    for key, expected in (("tcp_target_schema_version", 2), ("expected_task_count", 50),
                          ("expected_episodes_per_task", 50)):
        _integer_is(data, key, expected, "datasets.vla_data", issues)
    root_value = data.get("data_root_dir")
    if not isinstance(root_value, str) or not root_value.strip():
        issues.append("datasets.vla_data.data_root_dir must name the selected RoboTwin-Clean-TCP directory")
    elif Path(root_value).name == "RoboTwin-Clean-TCP-RawLast":
        issues.append("RoboTwin-Clean-TCP-RawLast is not the selected Clean-last dataset")
    if data.get("train_start_frames") is not None:
        issues.append("Full-frame training requires train_start_frames=null")
    for key in ("drop_incomplete_action_chunks", "delete_pause_frame", "persistent_workers"):
        _boolean(data, key, "datasets.vla_data", issues, expected=False)
    if "action_horizon" in data:
        _integer_is(data, "action_horizon", 32, "datasets.vla_data", issues)
    workers = data.get("num_workers", 4)
    if type(workers) is not int or workers < 0:
        issues.append("datasets.vla_data.num_workers must be a nonnegative integer")
    _boolean(data, "pin_memory", "datasets.vla_data", issues)
    for key in ("reload_modules", "freeze_modules"):
        if trainer.get(key) != "qwen_vl_interface":
            issues.append(f"trainer.{key} must be 'qwen_vl_interface'")
    for key in ("random_init_action_model", "shortest_angular_joint_loss"):
        _boolean(trainer, key, "trainer", issues, expected=True)
    _boolean(trainer, "is_resume", "trainer", issues, expected=False)
    if not trainer.get("pretrained_checkpoint"):
        issues.append("trainer.pretrained_checkpoint must identify the source OFT checkpoint")
    return issues


def _path(value) -> Path:
    result = Path(value).expanduser()
    return result if result.is_absolute() else REPO / result


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_tasks() -> set[str]:
    """Read the local official mixture literal without importing its registry."""
    source = REPO / "starVLA/dataloader/gr00t_lerobot/mixtures.py"
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "robotwin":
                    return {row[0] for row in ast.literal_eval(value)}
    raise ValueError("Cannot locate the official robotwin task mixture")


def _check_checkpoint(config, issues):
    _, _, _, data, trainer = _sections(config)
    raw_path = trainer.get("pretrained_checkpoint")
    summary = {"path": raw_path, "tensor_keys_checked": False, "full_weights_loaded": False}
    if not isinstance(raw_path, str) or not raw_path:
        return summary
    checkpoint = _path(raw_path)
    summary["path"] = str(checkpoint)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        issues.append(f"Source checkpoint is missing or empty: {checkpoint}")
        return summary
    summary["size_bytes"] = checkpoint.stat().st_size
    try:
        if zipfile.is_zipfile(checkpoint):
            with zipfile.ZipFile(checkpoint) as archive:
                names = archive.namelist()
            has_pickle = any(name.endswith("/data.pkl") or name == "data.pkl" for name in names)
            has_version = any(name.endswith("/version") or name == "version" for name in names)
            has_storage = any("/data/" in name or name.startswith("data/") for name in names)
            if not (has_pickle and has_version and has_storage):
                issues.append(f"Source checkpoint lacks a PyTorch tensor archive structure: {checkpoint}")
            summary.update(format="pytorch_zip", archive_members=len(names))
        else:
            with checkpoint.open("rb") as stream:
                header = stream.read(2)
            if len(header) != 2 or header[0] != 0x80 or header[1] not in (2, 3, 4, 5):
                issues.append(f"Source checkpoint is not a recognized PyTorch archive: {checkpoint}")
            summary["format"] = "legacy_pickle_header_only"
    except (OSError, zipfile.BadZipFile) as exc:
        issues.append(f"Cannot inspect source checkpoint: {exc}")
    source_config_path = checkpoint.parent.parent / "config.yaml"
    if not source_config_path.is_file():
        issues.append(f"Source checkpoint config is missing: {source_config_path}")
    else:
        try:
            source = OmegaConf.load(source_config_path)
            summary["source_framework"] = source.get("framework", {}).get("name")
            if summary["source_framework"] != "QwenOFT":
                issues.append("Source checkpoint framework must be QwenOFT")
            source_action = source.framework.action_model
            source_horizon = source_action.get("action_horizon")
            if source_horizon is None:
                source_horizon = (source_action.get("past_action_window_size", 0) + 1
                                  + source_action.get("future_action_window_size", 0))
            if source_action.get("action_dim") != 14 or source_horizon != 32:
                issues.append("Source OFT checkpoint must use 14D actions and a 32-step horizon")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            issues.append(f"Cannot read source checkpoint config: {exc}")
    statistics = data.get("pretrained_stats_path", str(checkpoint.parent.parent / "dataset_statistics.json"))
    if not isinstance(statistics, str) or not statistics:
        issues.append("Pretrained dataset statistics path must be configured")
    else:
        statistics_path = _path(statistics)
        try:
            values = _read_json(statistics_path)
            if not isinstance(values, dict) or not values:
                raise ValueError("Expected a nonempty statistics mapping")
            summary["statistics_path"] = str(statistics_path)
            summary["statistics_keys"] = list(values)
        except (OSError, ValueError) as exc:
            issues.append(f"Cannot read pretrained dataset statistics at {statistics_path}: {exc}")
    return summary


def _sidecar_header(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    with zipfile.ZipFile(path) as archive:
        with archive.open("tcp_pose_world_wxyz.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version in ((2, 0), (3, 0)):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported NumPy array header version {version}")
    return metadata, shape, dtype


def _check_data(config, issues):
    _, _, _, data, _ = _sections(config)
    raw_root = data.get("data_root_dir")
    summary = {"root": raw_root, "tasks": [], "task_count": 0, "episode_count": 0, "public_frame_count": 0}
    if not isinstance(raw_root, str) or not raw_root:
        return summary
    root = _path(raw_root)
    summary["root"] = str(root)
    manifest_path = root / "preparation_manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError) as exc:
        issues.append(f"Cannot read TCP dataset manifest at {manifest_path}: {exc}")
        return summary
    summary["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    policy = manifest.get("target_sampling")
    if manifest.get("status") != "passed":
        issues.append("TCP dataset manifest status must be passed")
    if policy != GOAL_POLICY:
        issues.append("TCP dataset target_sampling must match Clean-last GOAL_POLICY schema 2; RawLast is not allowed")
    summary["target_schema_version"] = policy.get("schema_version") if isinstance(policy, dict) else None
    tasks = manifest.get("tasks", [])
    if not isinstance(tasks, list):
        issues.append("TCP manifest tasks must be a list")
        return summary
    names = [task.get("task") for task in tasks]
    if len(tasks) != 50 or len(set(names)) != 50:
        issues.append(f"Expected 50 distinct task records, found {len(tasks)}")
    try:
        expected_names = _canonical_tasks()
        if set(names) != expected_names:
            issues.append(f"TCP task coverage differs from robotwin_wrap_32: missing={sorted(expected_names - set(names))}, extra={sorted(set(names) - expected_names)}")
    except (OSError, ValueError, TypeError) as exc:
        issues.append(f"Cannot verify the official 50-task mixture: {exc}")
    for task in tasks:
        name = task.get("task")
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            issues.append(f"Invalid TCP task name: {name!r}")
            continue
        task_dir = root / name
        record = {"task": name, "episodes": 0, "frames": 0, "checked_sidecars": 0}
        try:
            info = _read_json(task_dir / "meta/info.json")
            contract = _read_json(task_dir / "meta/tcp_pose.json")
            episodes = task.get("episodes", [])
            public_episodes = [json.loads(line) for line in (task_dir / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
            if contract.get("target_sampling") != GOAL_POLICY:
                issues.append(f"{name}: task target policy must be Clean-last schema 2")
            for key, value in (("coordinate_frame", "world"), ("position_unit", "meters"), ("quaternion_order", "wxyz"), ("layout", POSE_LAYOUT)):
                if contract.get(key) != value:
                    issues.append(f"{name}: invalid TCP {key}")
            feature = info.get("features", {}).get("observation.tcp_pose", {})
            if feature.get("dtype") != "float32" or feature.get("shape") != [14]:
                issues.append(f"{name}: TCP feature must be float32[14]")
            record["episodes"] = len(episodes)
            record["frames"] = sum(int(ep["public_length"]) for ep in episodes)
            ids = [ep["public_episode_id"] for ep in episodes]
            if len(episodes) != 50 or set(ids) != set(range(50)):
                issues.append(f"{name}: expected all 50 public episodes")
            declared = {int(ep["episode_index"]): int(ep["length"]) for ep in public_episodes}
            if declared != {int(ep["public_episode_id"]): int(ep["public_length"]) for ep in episodes}:
                issues.append(f"{name}: episode metadata disagrees with the TCP manifest")
            if info.get("total_episodes") != len(episodes) or info.get("total_frames") != record["frames"]:
                issues.append(f"{name}: info.json episode/frame totals disagree")
            chunks = info["chunks_size"]
            if type(chunks) is not int or chunks < 1:
                raise ValueError("chunks_size must be a positive integer")
            for episode in episodes:
                eid, length = int(episode["public_episode_id"]), int(episode["public_length"])
                values = {"episode_index": eid, "episode_chunk": eid // chunks}
                parquet = task_dir / info["data_path"].format(**values)
                sidecar = task_dir / contract["sidecar_template"].format(**values)
                try:
                    metadata = pq.ParquetFile(parquet)
                    missing = {"observation.state", "action", "frame_index", "observation.tcp_pose"} - set(metadata.schema_arrow.names)
                    if missing:
                        issues.append(f"{parquet}: missing columns {sorted(missing)}")
                    else:
                        field = metadata.schema_arrow.field("observation.tcp_pose").type
                        if not pa.types.is_fixed_size_list(field) or field.list_size != 14 or field.value_type != pa.float32():
                            issues.append(f"{parquet}: TCP schema must be fixed-size float32[14]")
                    if metadata.metadata.num_rows != length:
                        issues.append(f"{parquet}: row count disagrees with public_length")
                    details, shape, dtype = _sidecar_header(sidecar)
                    if (details.get("task") != name or details.get("public_episode_id") != eid
                            or details.get("public_length") != length or details.get("raw_length") != length + 1
                            or details.get("target_sampling") != GOAL_POLICY):
                        issues.append(f"{sidecar}: identity/length/Clean-last target policy mismatch")
                    if shape != (length + 1, 14) or dtype != np.dtype("float32"):
                        issues.append(f"{sidecar}: TCP array must be float32[public_length+1,14]")
                    record["checked_sidecars"] += 1
                except (OSError, ValueError, KeyError, zipfile.BadZipFile, pa.ArrowException) as exc:
                    issues.append(f"Cannot inspect episode {name}/{eid}: {exc}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"Cannot inspect TCP task {name}: {exc}")
        summary["tasks"].append(record)
    summary["task_count"] = len(summary["tasks"])
    summary["episode_count"] = sum(task["episodes"] for task in summary["tasks"])
    summary["public_frame_count"] = sum(task["frames"] for task in summary["tasks"])
    summary["checked_sidecars"] = sum(task["checked_sidecars"] for task in summary["tasks"])
    for key in ("task_count", "episode_count", "public_frame_count"):
        if manifest.get(key) != summary[key]:
            issues.append(f"TCP manifest {key} disagrees with its task records")
    if summary["episode_count"] != 2500:
        issues.append(f"Expected 2500 public episodes, found {summary['episode_count']}")
    return summary


def check_setup(config, config_only=False) -> dict:
    configuration_issues = check_training_config(config)
    issues = list(configuration_issues)
    report = {
        "ready": False, "configuration_valid": not configuration_issues, "config_only": bool(config_only),
        "framework": config.get("framework", {}).get("name"), "issues": issues,
        "scope": "Configuration and dataset/checkpoint structure only. No model weights, images or full raw arrays are loaded; this is not training, inference validation, or a RoboTwin benchmark result.",
    }
    if not config_only:
        report["checkpoint"] = _check_checkpoint(config, issues)
        report["data"] = _check_data(config, issues)
    report["ready"] = not issues
    report["issue_count"] = len(issues)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", type=Path, required=True)
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_setup(OmegaConf.load(args.config_yaml), config_only=args.config_only)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    data = report.get("data", {})
    print(json.dumps({
        "ready": report["ready"], "configuration_valid": report["configuration_valid"],
        "config_only": report["config_only"], "issue_count": report["issue_count"],
        "task_count": data.get("task_count"), "episode_count": data.get("episode_count"),
        "public_frame_count": data.get("public_frame_count"), "checked_sidecars": data.get("checked_sidecars"),
        "first_issues": report["issues"][:5], "scope": report["scope"],
    }, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
