"""RoboTwin raw-state adapter using VLAct's original training transforms.

This module never touches action predictions or TCP targets. Action decoding
remains with the existing VLAct clients.
"""

from __future__ import annotations

import inspect

import numpy as np

from deployment.model_server.pose_conventions import validate_quaternion_order


class RoboTwinStateNormalizer:
    """Apply the saved joint-state transform or preserve explicit raw TCP state."""

    KEY_DIMS = {
        "state.left_joints": 6,
        "state.right_joints": 6,
        "state.left_gripper": 1,
        "state.right_gripper": 1,
    }
    STATISTICS = ("min", "max", "mean", "std", "q01", "q99")

    def __init__(self, data_cfg, statistics):
        from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
        from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
        from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
        from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata, StateActionMetadata
        from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform

        data_mix = data_cfg.get("data_mix")
        if data_mix not in DATASET_NAMED_MIXTURES:
            raise ValueError("Raw state normalization requires the registered training data_mix")
        robot_types = {entry[2] for entry in DATASET_NAMED_MIXTURES[data_mix]}
        if len(robot_types) != 1:
            raise ValueError("Raw RoboTwin state requires a single robot type in the training mixture")
        robot_type = next(iter(robot_types))
        if not robot_type.startswith("robotwin"):
            raise ValueError("This raw-state adapter supports RoboTwin training configurations only")
        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        representation = data_cfg.get("state_representation", "joint")
        if representation not in ("joint", "tcp_pose"):
            raise ValueError("state_representation must be 'joint' or 'tcp_pose'")
        self.pose_state = representation == "tcp_pose"
        self.quaternion_order = validate_quaternion_order(data_cfg.get("tcp_quaternion_order", "wxyz"))
        if self.pose_state:
            self.KEY_DIMS = {"state.tcp_pose": 14}
        self.state_keys = list(data_config.state_keys)
        if self.state_keys != list(self.KEY_DIMS):
            raise ValueError("Training state keys do not match the selected RoboTwin state representation")

        state_stats = statistics.get("state")
        if not isinstance(state_stats, dict):
            raise ValueError("Checkpoint has no state statistics for raw state inputs")
        arrays = {}
        for key in self.STATISTICS:
            value = np.asarray(state_stats.get(key), dtype=np.float64)
            if value.shape != (14,) or not np.isfinite(value).all():
                raise ValueError(f"Checkpoint state statistic {key!r} must be finite and 14D")
            arrays[key] = value

        by_key, modalities = {}, {}
        start = 0
        for key in self.state_keys:
            width = self.KEY_DIMS[key]
            name = key.split(".", 1)[1]
            by_key[name] = {stat: values[start:start + width].copy() for stat, values in arrays.items()}
            modalities[name] = StateActionMetadata(absolute=True, shape=(width,), continuous=True)
            start += width
        metadata = DatasetMetadata.model_validate({
            "statistics": {"state": by_key, "action": {}},
            "modalities": {"state": modalities, "action": {}, "video": {}},
            "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT,
        })
        parameters = inspect.signature(data_config.transform).parameters
        original = data_config.transform(data_cfg=data_cfg) if "data_cfg" in parameters else data_config.transform()
        state_transforms = []
        for transform in original.transforms:
            keys = list(transform.apply_to)
            if any(key.startswith("state.") for key in keys):
                if not all(key.startswith("state.") for key in keys):
                    raise ValueError("Mixed state/action transforms require a dedicated deployment adapter")
                state_transforms.append(transform)
        if not state_transforms:
            raise ValueError("Training DataConfig contains no state transforms")
        self.transform = ComposedModalityTransform(transforms=state_transforms)
        self.transform.set_metadata(metadata)
        self.transform.eval()

    def apply_state(self, state):
        import torch

        values = np.asarray(state, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape != (1, 14) or not np.isfinite(values).all():
            raise ValueError("Raw RoboTwin state must be a finite vector shaped [14] or [1,14]")
        if self.pose_state:
            quaternions = values[:, [3, 4, 5, 6, 10, 11, 12, 13]].reshape(-1, 4)
            if not np.allclose(np.linalg.norm(quaternions, axis=-1), 1.0, rtol=0, atol=1e-5):
                raise ValueError(f"TCP pose state requires unit {self.quaternion_order} quaternions for both arms")
        parts = {}
        start = 0
        for key in self.state_keys:
            width = self.KEY_DIMS[key]
            parts[key] = values[:, start:start + width].copy()
            start += width
        transformed = self.transform(parts)
        result = []
        for key in self.state_keys:
            value = transformed[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            result.append(np.asarray(value))
        return np.concatenate(result, axis=-1)
