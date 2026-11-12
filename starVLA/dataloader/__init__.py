import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        use_target_tcp = False
        if "include_tcp" in vla_dataset_cfg or "use_target_tcp" in vla_dataset_cfg:
            from starVLA.dataloader.tcp_goal_dataset import resolve_include_tcp

            use_target_tcp = resolve_include_tcp(vla_dataset_cfg)
        include_state = vla_dataset_cfg.get("include_state", False)
        if use_target_tcp:
            if type(include_state) is not bool:
                raise TypeError("datasets.vla_data.include_state must be a boolean.")
            include_time_budget = vla_dataset_cfg.get("include_time_budget", False)
            if type(include_time_budget) is not bool:
                raise TypeError("datasets.vla_data.include_time_budget must be a boolean.")
            if include_time_budget:
                raise ValueError("TCP goal conditioning does not use a time_budget input.")

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        if use_target_tcp:
            from starVLA.dataloader.tcp_goal_dataset import TCPGoalDataset

            vla_dataset = TCPGoalDataset(
                vla_dataset,
                data_root=Path(vla_dataset_cfg.data_root_dir),
                seed=getattr(cfg, "seed", 42),
                cache_size=vla_dataset_cfg.get("tcp_cache_size", 32),
                include_state=include_state,
                expected_target_schema_version=vla_dataset_cfg.get("tcp_target_schema_version"),
            )
        num_workers = vla_dataset_cfg.get("num_workers", 4)
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("datasets.vla_data.num_workers must be a nonnegative integer.")
        pin_memory = vla_dataset_cfg.get("pin_memory", False)
        if type(pin_memory) is not bool:
            raise TypeError("datasets.vla_data.pin_memory must be a boolean.")
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            # shuffle=True
        )        
        if not dist.is_initialized() or dist.get_rank() == 0:
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
