"""Add dynamic TCP targets to an existing LeRobot mixture dataset.

Images, language, normalized joint state, and actions come from the original
dataset unchanged. Only TCP target fields are added. The dataset manifest's
goal policy is always passed explicitly to the shared sampling helper.
"""

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from examples.Robotwin.train_files.prepare_clean_tcp_dataset import (
    GOAL_POLICY,
    RAW_LAST_GOAL_POLICY,
    sample_target_tcp_pose,
)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read TCP dataset metadata at {path}: {error}") from error


def resolve_include_tcp(data_cfg) -> bool:
    """Prefer include_tcp while retaining the existing use_target_tcp alias."""
    supplied = {}
    for name in ("include_tcp", "use_target_tcp"):
        if name in data_cfg:
            value = data_cfg[name]
            if type(value) is not bool:
                raise TypeError(f"datasets.vla_data.{name} must be a boolean.")
            supplied[name] = value
    if len(supplied) == 2 and supplied["include_tcp"] != supplied["use_target_tcp"]:
        raise ValueError("datasets.vla_data.include_tcp and use_target_tcp explicitly disagree.")
    return supplied.get("include_tcp", supplied.get("use_target_tcp", False))


class TCPGoalDataset(Dataset):
    """Wrap a mixture and sample a future TCP from its public episode sidecar.

    Each process has a lazy LRU cache and an independent NumPy Generator. PID
    changes reset process-local state; PIDs are not seed entropy, so seeded
    runs remain reproducible. With persistent DataLoader workers, recreate the
    DataLoader/workers after changing epochs; recreating only its iterator
    does not propagate the new epoch into existing worker processes.
    """

    def __init__(
        self, base, data_root: Path, *, seed: int | None = None, cache_size: int = 32,
        include_state: bool = True, expected_target_schema_version: int | None = None,
    ):
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 1:
            raise ValueError("TCP cache_size must be a positive integer.")
        if seed is not None and (not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0):
            raise ValueError("TCP seed must be a nonnegative integer or None.")
        if type(include_state) is not bool:
            raise TypeError("TCP include_state must be a boolean.")
        if expected_target_schema_version is not None and (
                type(expected_target_schema_version) is not int or expected_target_schema_version < 1):
            raise ValueError("TCP expected_target_schema_version must be a positive integer or None.")
        self.base = base
        self.data_root = Path(data_root).expanduser().resolve(strict=True)
        self.seed = None if seed is None else int(seed)
        self.cache_size = cache_size
        self.include_state = include_state
        self._manifest_path = self.data_root / "preparation_manifest.json"
        manifest = _read_json(self._manifest_path)
        if manifest.get("status") != "passed":
            raise ValueError(f"TCP dataset manifest must have status 'passed': {self._manifest_path}")
        self.target_sampling = deepcopy(manifest.get("target_sampling"))
        if self.target_sampling not in (GOAL_POLICY, RAW_LAST_GOAL_POLICY):
            raise ValueError(f"Unsupported TCP target_sampling policy in {self._manifest_path}")
        if (expected_target_schema_version is not None
                and self.target_sampling.get("schema_version") != expected_target_schema_version):
            raise ValueError(
                f"TCP target_sampling schema must be {expected_target_schema_version}, "
                f"found {self.target_sampling.get('schema_version')} in {self._manifest_path}"
            )
        self._episode_lengths = {}
        for task in manifest.get("tasks", []):
            for episode in task["episodes"]:
                self._episode_lengths[(task["task"], int(episode["public_episode_id"]))] = int(episode["public_length"])
        if not self._episode_lengths:
            raise ValueError(f"No public episodes listed in TCP manifest: {self._manifest_path}")
        self._task_metadata = {}
        self._checked_subsets = set()
        self._epoch = int(getattr(base, "epoch", 0))
        self._rank = (torch.distributed.get_rank() if torch.distributed.is_initialized()
                      else int(os.environ.get("RANK", "0")))
        self._reset_runtime()

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        # During unpickling, base may not exist yet. Do not delegate Python's
        # special hooks, including PyTorch's optional bulk-fetch hook.
        if name.startswith("__"):
            raise AttributeError(name)
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_runtime_key=None, _rng=None, _cache=OrderedDict())
        return state

    def _reset_runtime(self):
        self._runtime_key = None
        self._rng = None
        self._cache = OrderedDict()

    def _ensure_runtime(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        worker_seed = int(worker.seed if worker is not None else torch.initial_seed())
        key = (os.getpid(), worker_id, worker_seed, self._epoch)
        if key != self._runtime_key:
            entropy = [worker_seed, self._rank, worker_id + 1, self._epoch]
            if self.seed is not None:
                entropy.append(self.seed)
            self._rng = np.random.default_rng(np.random.SeedSequence(entropy))
            self._cache = OrderedDict()
            self._runtime_key = key

    def set_epoch(self, epoch: int):
        if not isinstance(epoch, (int, np.integer)) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("TCP epoch must be a nonnegative integer.")
        setter = getattr(self.base, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))
        self._epoch = int(epoch)
        self._reset_runtime()

    def _metadata_for_task(self, name, subset_path):
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Expected a task-level dataset_name, got {name!r}")
        task_dir = self.data_root / name
        if name not in self._task_metadata:
            info = _read_json(task_dir / "meta/info.json")
            contract = _read_json(task_dir / "meta/tcp_pose.json")
            if contract.get("target_sampling") != self.target_sampling:
                raise ValueError(f"Task/root TCP target policy mismatch: {task_dir}")
            feature = info.get("features", {}).get("observation.tcp_pose", {})
            if feature.get("dtype") != "float32" or feature.get("shape") != [14]:
                raise ValueError(f"Expected observation.tcp_pose float32[14] in {task_dir / 'meta/info.json'}")
            if not isinstance(info.get("chunks_size"), int) or info["chunks_size"] < 1:
                raise ValueError(f"Invalid chunks_size in {task_dir / 'meta/info.json'}")
            if not isinstance(contract.get("sidecar_template"), str):
                raise ValueError(f"Missing TCP sidecar_template in {task_dir / 'meta/tcp_pose.json'}")
            self._task_metadata[name] = (info, contract)
        info, contract = self._task_metadata[name]
        subset = Path(subset_path).expanduser()
        if subset.name != name:
            raise ValueError(f"subset_path {subset} does not identify dataset_name {name!r}")
        subset_key = (name, str(subset.resolve()))
        if subset_key not in self._checked_subsets:
            # A private metadata view is supported, but it must describe the
            # same task and target policy as the canonical TCP data root.
            if subset.resolve() != task_dir.resolve():
                if _read_json(subset / "meta/info.json") != info:
                    raise ValueError(f"Temporary subset metadata differs from TCP task metadata: {subset}")
                view_contract = _read_json(subset / "meta/tcp_pose.json")
                if view_contract != contract:
                    raise ValueError(f"Temporary subset TCP contract differs from {task_dir}: {subset}")
            self._checked_subsets.add(subset_key)
        return task_dir, info, contract

    def _full_tcp(self, name, episode_id, task_dir, info, contract):
        key = (name, episode_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if key not in self._episode_lengths:
            raise ValueError(f"Public episode {episode_id} of {name!r} is absent from {self._manifest_path}")
        n_clean = self._episode_lengths[key]
        relative = Path(contract["sidecar_template"].format(
            episode_chunk=episode_id // info["chunks_size"], episode_index=episode_id))
        path = task_dir / relative
        # Private metadata views may share the complete tcp directory by
        # symlink. Individual sidecars must stay inside that selected tree.
        tcp_root = (task_dir / "tcp").resolve()
        if (relative.is_absolute() or not relative.parts or relative.parts[0] != "tcp"
                or ".." in relative.parts or not path.resolve().is_relative_to(tcp_root)):
            raise ValueError(f"TCP sidecar_template resolves outside its task TCP tree: {path}")
        try:
            with np.load(path, allow_pickle=False) as sidecar:
                sequence = sidecar["tcp_pose_world_wxyz"].copy()
                metadata = json.loads(str(sidecar["metadata_json"]))
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"Cannot load TCP episode sidecar {path}: {error}") from error
        if sequence.dtype != np.float32 or sequence.shape != (n_clean + 1, 14) or not np.isfinite(sequence).all():
            raise ValueError(f"Expected finite float32[{n_clean + 1},14] TCP sequence in {path}")
        if (metadata.get("task") != name or metadata.get("public_episode_id") != episode_id
                or metadata.get("public_length") != n_clean or metadata.get("raw_length") != n_clean + 1):
            raise ValueError(f"TCP sidecar public episode identity/length mismatch: {path}")
        if metadata.get("target_sampling") != self.target_sampling:
            raise ValueError(f"Sidecar/task/root TCP target policy mismatch: {path}")
        sequence.setflags(write=False)
        self._cache[key] = sequence
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return sequence

    def __getitem__(self, index):
        sample = dict(self.base[index])
        required = ("subset_path", "dataset_name", "trajectory_id", "step_index")
        if self.include_state:
            required = ("state",) + required
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"TCP wrapper requires base sample fields {missing}.")
        if self.include_state and (sample["state"] is None or tuple(getattr(sample["state"], "shape", ())) != (1, 14)):
            raise ValueError("TCP wrapper requires the original normalized joint state shaped [1,14]; enable include_state=True.")
        for key in ("trajectory_id", "step_index"):
            value = sample[key]
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Base sample {key} must be a nonnegative integer, got {value!r}")
        self._ensure_runtime()
        name, episode_id, step = sample["dataset_name"], int(sample["trajectory_id"]), int(sample["step_index"])
        task_dir, info, contract = self._metadata_for_task(name, sample["subset_path"])
        sequence = self._full_tcp(name, episode_id, task_dir, info, contract)
        goal = sample_target_tcp_pose(sequence, step, self._rng, target_sampling=self.target_sampling)
        conflicts = goal.keys() & sample.keys()
        if conflicts:
            raise ValueError(f"Base sample already contains TCP target fields: {sorted(conflicts)}")
        sample.update(goal)
        return sample

    def __getitems__(self, indices):
        # DataLoader may prefer this hook to __getitem__; keep augmentation.
        return [self[index] for index in indices]

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json"):
        """Preserve base statistics and save the TCP contract in separate files."""
        save_path = Path(save_path)
        stats_source = self.data_root / "tcp_statistics.json"
        if not stats_source.is_file():
            raise FileNotFoundError(f"Missing TCP training statistics: {stats_source}")
        self.base.save_dataset_statistics(save_path, format=format)
        stats_target = save_path.parent / "tcp_pose_statistics.json"
        if stats_target.resolve() == stats_source.resolve():
            raise ValueError("Run statistics destination must differ from source TCP statistics.")
        shutil.copy2(stats_source, stats_target)
        first_name = next(iter(self._episode_lengths))[0]
        _, _, pose_contract = self._metadata_for_task(first_name, self.data_root / first_name)
        contract = {
            "schema_version": 1, "data_root": str(self.data_root),
            "dataset_manifest": str(self._manifest_path),
            "dataset_manifest_sha256": hashlib.sha256(self._manifest_path.read_bytes()).hexdigest(),
            "target_sampling": self.target_sampling,
            "sampler": {"module": sample_target_tcp_pose.__module__, "function": "sample_target_tcp_pose",
                        "explicit_target_sampling_required": True},
            "output": {"target_tcp_pose": "float32 [14]", "goal_frame": "int", "used_terminal_fallback": "bool"},
            "pose_contract": pose_contract,
            "normalization_applied_by_wrapper": False,
            "include_state": self.include_state,
            "state": ("Original loader output [1,14], preserved without reordering or renormalization."
                      if self.include_state else "Joint state is disabled; TCP targets do not require joint state."),
            "tcp_statistics_file": stats_target.name,
            "statistics_note": "Training full-sequence diagnostics. Positions may support later normalization; keep wxyz quaternions unit and do not apply joint-state normalization to TCP.",
        }
        (save_path.parent / "target_tcp_contract.json").write_text(
            json.dumps(contract, indent=2, allow_nan=False) + "\n", encoding="utf-8")
