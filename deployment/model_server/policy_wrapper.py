"""Opt-in OFT input preparation with VLAct's existing action wire format.

The wrapped policy still returns ``normalized_actions``. This wrapper never
decodes actions, since RoboTwin's existing client performs that operation.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np


TCP_KEY = "target_tcp_pose"
TCP_ALIASES = ("tcp_pose", "target_pose")


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _plain_config(config):
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config


def _conditioning(model_cfg):
    framework = model_cfg.get("framework", {})
    if framework.get("name") != "QwenOFT":
        return None
    contract = framework.get("conditioning", {})
    if not isinstance(contract, Mapping):
        raise ValueError("framework.conditioning must be a mapping")
    if not _boolean(contract.get("enabled", False), "conditioning.enabled"):
        return None
    for key in ("include_state", "include_tcp"):
        _boolean(contract.get(key, False), f"conditioning.{key}")
    if contract.get("include_state", False) and contract.get("state_input_space", "raw") not in ("raw", "normalized"):
        raise ValueError("conditioning.state_input_space must be 'raw' or 'normalized'")
    return contract


def _canonical_tcp(example, index):
    supplied = [(key, example[key]) for key in (TCP_KEY, *TCP_ALIASES) if key in example and example[key] is not None]
    if not supplied:
        raise ValueError(f"OFT example {index} requires {TCP_KEY} when include_tcp=True")
    canonical = supplied[0][1]
    reference = np.asarray(canonical, dtype=np.float32).reshape(-1)
    for _, value in supplied[1:]:
        alias = np.asarray(value, dtype=np.float32).reshape(-1)
        if reference.shape != alias.shape or not np.allclose(reference, alias, rtol=1e-6, atol=1e-7, equal_nan=False):
            raise ValueError(f"OFT example {index} has conflicting TCP pose aliases")
    return canonical


def _public_tcp_contract(contract):
    """Expose only coordinate semantics, never source paths or episode data."""
    if not isinstance(contract, Mapping):
        return None
    fields = ("coordinate_frame", "position_unit", "quaternion_order", "layout", "tcp_definition")
    result = {key: contract[key] for key in fields if key in contract}
    return result or None


def _read_tcp_contract(model_cfg, checkpoint_path):
    if checkpoint_path is not None:
        saved = Path(checkpoint_path).expanduser().parent.parent / "target_tcp_contract.json"
        if saved.is_file():
            return _public_tcp_contract(json.loads(saved.read_text()).get("pose_contract"))
    data_root = model_cfg.get("datasets", {}).get("vla_data", {}).get("data_root_dir")
    if not data_root:
        return None
    root = Path(data_root).expanduser()
    manifest_path = root / "preparation_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "passed":
        raise ValueError("TCP dataset preparation manifest has not passed validation")
    contracts = []
    for item in manifest.get("tasks", []):
        name = item.get("task")
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("TCP manifest contains an invalid task name")
        contracts.append(_public_tcp_contract(json.loads((root / name / "meta/tcp_pose.json").read_text())))
    if not contracts:
        return None
    if any(contract != contracts[0] for contract in contracts):
        raise ValueError("TCP dataset tasks have different coordinate contracts")
    return contracts[0]


class PolicyServerWrapper:
    """Wrap an already loaded framework; loading/device placement stay upstream."""

    def __init__(self, policy, *, checkpoint_path=None):
        self.policy = policy
        self.model_cfg = _plain_config(policy.config)
        self.norm_stats = getattr(policy, "norm_stats", {})
        self.conditioning = _conditioning(self.model_cfg)
        self._normalizers = {}
        self._tcp_contract = _read_tcp_contract(self.model_cfg, checkpoint_path) if (
            self.conditioning is not None and self.conditioning.get("include_tcp", False)
        ) else None

    def _state_normalizer(self, unnorm_key):
        from deployment.model_server.state_normalization import RoboTwinStateNormalizer

        if unnorm_key is None:
            if len(self.norm_stats) != 1:
                raise ValueError("Raw state requires an explicit unnorm_key when checkpoint statistics are ambiguous")
            unnorm_key = next(iter(self.norm_stats))
        if unnorm_key not in self.norm_stats:
            raise ValueError("Requested unnorm_key is absent from checkpoint statistics")
        if unnorm_key not in self._normalizers:
            data_cfg = self.model_cfg.get("datasets", {}).get("vla_data", {})
            self._normalizers[unnorm_key] = RoboTwinStateNormalizer(data_cfg, self.norm_stats[unnorm_key])
        return self._normalizers[unnorm_key]

    @property
    def metadata(self):
        action = self.model_cfg.get("framework", {}).get("action_model", {})
        contract = self.conditioning or {}
        active = self.conditioning is not None
        include_state = active and contract.get("include_state", False)
        include_tcp = active and contract.get("include_tcp", False)
        horizon = getattr(self.policy, "chunk_len", None)
        if horizon is None:
            horizon = action.get("action_horizon")
        if horizon is None:
            horizon = int(action.get("past_action_window_size", 0)) + 1 + int(action.get("future_action_window_size", 0))
        return {
            "env": "vlact_policy_server",
            "framework": self.model_cfg.get("framework", {}).get("name"),
            "action_chunk_size": int(horizon),
            "action_output_key": "normalized_actions",
            "action_decoding": "client",
            "available_unnorm_keys": list(self.norm_stats),
            "conditioning_enabled": active,
            "include_state": include_state,
            "include_tcp": include_tcp,
            "state_dim": int(contract.get("state_dim", 14)) if include_state else 0,
            "tcp_dim": int(contract.get("tcp_dim", 14)) if include_tcp else 0,
            "state_input_space": contract.get("state_input_space", "raw") if include_state else None,
            "model_state_space": "normalized" if include_state else None,
            "tcp_pose_key": TCP_KEY if include_tcp else None,
            "tcp_pose_aliases": list(TCP_ALIASES) if include_tcp else [],
            "tcp_pose_contract": self._tcp_contract,
            "tcp_normalization": "none" if include_tcp else None,
        }

    def predict_action(self, examples=None, unnorm_key=None, **kwargs):
        if unnorm_key is not None:
            kwargs["unnorm_key"] = unnorm_key
        if self.conditioning is None:
            return self.policy.predict_action(examples=examples, **kwargs)
        if not isinstance(examples, list):
            examples = [examples]
        include_state = self.conditioning.get("include_state", False)
        include_tcp = self.conditioning.get("include_tcp", False)
        raw_state = self.conditioning.get("state_input_space", "raw") == "raw"
        normalizer = self._state_normalizer(unnorm_key) if include_state and raw_state else None
        prepared = []
        for index, example in enumerate(examples):
            converted = dict(example)
            if include_state:
                if converted.get("state") is None:
                    raise ValueError(f"OFT example {index} requires state when include_state=True")
                if normalizer is not None:
                    converted["state"] = normalizer.apply_state(converted["state"])
            else:
                converted.pop("state", None)
            if include_tcp:
                converted[TCP_KEY] = _canonical_tcp(converted, index)
            else:
                converted.pop(TCP_KEY, None)
            for alias in TCP_ALIASES:
                converted.pop(alias, None)
            prepared.append(converted)
        return self.policy.predict_action(examples=prepared, **kwargs)
