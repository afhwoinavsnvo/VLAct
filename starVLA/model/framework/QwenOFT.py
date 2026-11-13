# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025]. 

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Configured Qwen vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states


Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
  or /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md （adpat a little code)
  
"""
from contextlib import nullcontext
from collections.abc import Mapping
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
AUTO_ANGULAR_JOINT_LOSS_GRIPPER_DIMS = {
    "interna1_split_aloha": 2,
    "robocoin": 2,
    "new_embodiment": 2,
}

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.vlm.image_preprocessing import preprocess_image_batch
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.action_model.oft_conditioning import OFTConditioning
from starVLA.model.modules.action_model.flow_matching_loss import blend_angular_wrap_abs_error
from starVLA.training.trainer_utils.trainer_tools import resize_images

@FRAMEWORK_REGISTRY.register("QwenOFT")
class Qwenvl_OFT(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for image/instruction token embeddings
      - One action-token embedding per predicted action step
      - Optional output projection and numeric state/target-TCP MLP encoders
      - MLPResNet action head for direct continuous-action regression

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        # Opt-in numeric conditions leave released checkpoints unchanged by default.
        condition_cfg = config.framework.get("conditioning", {})
        flags = {name: condition_cfg.get(name, default) for name, default in (
            ("enabled", False), ("include_state", False), ("include_tcp", False), ("freeze_vlm", True),
        )}
        for name, value in flags.items():
            if type(value) is not bool:
                raise ValueError(f"framework.conditioning.{name} must be a boolean")
        self.conditioning = OFTConditioning(
            hidden_dim=config.framework.action_model.action_hidden_dim,
            encoder_hidden_dim=condition_cfg.get("encoder_hidden_dim", 1024),
            state_dim=condition_cfg.get("state_dim", 14),
            tcp_dim=condition_cfg.get("tcp_dim", 14),
            include_state=flags["include_state"], include_tcp=flags["include_tcp"], enabled=flags["enabled"],
        )
        if self.conditioning.enabled:
            data_cfg = config.datasets.vla_data
            for name in ("include_state", "include_tcp"):
                if name in data_cfg:
                    if type(data_cfg[name]) is not bool:
                        raise ValueError(f"datasets.vla_data.{name} must be a boolean")
                    if data_cfg[name] != flags[name]:
                        raise ValueError(f"datasets.vla_data.{name} disagrees with framework.conditioning.{name}")
            if data_cfg.get("include_time_budget", False) or config.framework.action_model.get("use_time_budget", False):
                raise ValueError("Conditioned QwenOFT does not use time_budget; disable both budget switches")
            if flags["freeze_vlm"]:
                self.qwen_vl_interface.requires_grad_(False)
                self.qwen_vl_interface.eval()
                if hasattr(self.qwen_vl_interface.model, "gradient_checkpointing_disable"):
                    self.qwen_vl_interface.model.gradient_checkpointing_disable()

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        # self.hidden_dim = config.framework.action_model.action_hidden_dim
        
        self.action_token = "🔍" # TODO also can add spacail token to Qwen, but too complex
        token_ids = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"]
        if self.conditioning.enabled and len(token_ids) != 1:
            raise ValueError("OFT action placeholder must tokenize to one ID; use the checkpoint's matching processor")
        self.action_token_id = token_ids[0]

        # L1 损失
        self.l1_loss = nn.L1Loss()
        trainer_cfg = getattr(self.config, "trainer", None)
        shortest_angular_joint_loss = (
            trainer_cfg.get("shortest_angular_joint_loss", False) if trainer_cfg is not None else False
        )
        self.use_shortest_angular_joint_loss = str(shortest_angular_joint_loss).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._logged_first_batch_image_sizes = False

    def train(self, mode: bool = True):
        """Train adapters/head while keeping a frozen checkpoint VLM in eval mode."""
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.qwen_vl_interface.parameters()):
            self.qwen_vl_interface.eval()
        return self

    def _last_hidden(self, inputs):
        frozen = not any(parameter.requires_grad for parameter in self.qwen_vl_interface.parameters())
        # no_grad produces ordinary features that the trainable projection can
        # save for backward; inference_mode would prevent that training path.
        with (torch.no_grad() if frozen else nullcontext()):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encode_last = getattr(self.qwen_vl_interface, "encode_last_hidden_state", None)
                if self.conditioning.enabled and callable(encode_last):
                    return encode_last(**inputs)
                outputs = self.qwen_vl_interface(
                    **inputs, output_attentions=False, output_hidden_states=True, return_dict=True,
                )
                return outputs.hidden_states[-1]

    def _batch_condition(self, examples, key, dimension, parameter, *, aliases=()):
        values = []
        for index, example in enumerate(examples):
            keys = [name for name in (key, *aliases) if name in example]
            if not keys:
                raise ValueError(f"QwenOFT requires '{key}' for every example (missing at batch index {index})")
            tensors = []
            for name in keys:
                value = torch.as_tensor(example[name])
                if value.shape == (dimension,):
                    value = value.unsqueeze(0)
                if value.shape != (1, dimension):
                    raise ValueError(f"{name} at batch index {index} must have shape (1, {dimension}) or ({dimension},)")
                tensors.append(value)
            if any(not torch.allclose(tensors[0], value.to(tensors[0]), rtol=1e-6, atol=1e-6)
                   for value in tensors[1:]):
                raise ValueError(f"Conflicting {key} and {aliases} at batch index {index}")
            values.append(tensors[0])
        return torch.stack(values).to(device=parameter.device, dtype=parameter.dtype)

    def _condition_action_queries(self, action_queries, examples):
        if not self.conditioning.enabled:
            return action_queries
        parameter = next(self.conditioning.parameters())
        action_queries = action_queries.to(device=parameter.device, dtype=parameter.dtype)
        state = self._batch_condition(
            examples, "state", self.conditioning.state_dim, parameter,
        ) if self.conditioning.include_state else None
        tcp = self._batch_condition(
            examples, "target_tcp_pose", self.conditioning.tcp_dim, parameter, aliases=("tcp_pose", "target_pose"),
        ) if self.conditioning.include_tcp else None
        return self.conditioning(action_queries, state=state, tcp=tcp)

    def _prepare_vlm_inputs(self, examples, *, inference=False):
        """Accept worker-prepared tensors or retain the original image path."""
        if isinstance(examples, Mapping) and "qwen_inputs" in examples:
            batch = examples.get("examples")
            inputs = examples["qwen_inputs"]
            if not isinstance(batch, list) or not batch or not isinstance(inputs, Mapping):
                raise ValueError("Prepared OFT batches require examples and qwen_inputs")
            input_ids = inputs.get("input_ids")
            if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != len(batch):
                raise ValueError("Prepared OFT input_ids must match the example batch")
            device = next(self.qwen_vl_interface.parameters()).device
            inputs = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                      for key, value in inputs.items()}
            return batch, inputs

        if inference and type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) if inference else example["image"]
                        for example in examples]
        instructions = [example["lang"] for example in examples]
        if inference:
            self._log_batch_image_sizes_once(batch_images, stage="infer_recv")
        image_size_buckets = getattr(self.config.datasets.vla_data, "image_size_buckets", None)
        image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if self.conditioning.enabled:
            batch_images = preprocess_image_batch(
                batch_images, image_size_buckets=image_size_buckets, image_size=image_size,
            )
        elif image_size_buckets is None and image_size:
            batch_images = resize_images(batch_images, target_size=image_size)
        if not inference:
            self._log_batch_image_sizes_once(batch_images, stage="train")
        action_tokens = self.action_token * self.chunk_len
        suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=[instruction + suffix for instruction in instructions],
        )
        return examples, inputs

    def _log_batch_image_sizes_once(self, batch_images: List[List[Image.Image]], stage: str = "train") -> None:
        if self._logged_first_batch_image_sizes:
            return

        formatted_sizes = []
        for sample_idx, sample_images in enumerate(batch_images):
            sample_sizes = []
            for image_idx, image in enumerate(sample_images):
                size = getattr(image, "size", None)
                if size is None:
                    sample_sizes.append(f"img{image_idx}=unknown")
                else:
                    width, height = size
                    sample_sizes.append(f"img{image_idx}={width}x{height}")
            formatted_sizes.append(f"sample{sample_idx}: " + ", ".join(sample_sizes))

        logger.info(
            f"[QwenOFT] first {stage} batch image sizes -> " + " | ".join(formatted_sizes)
        )
        self._logged_first_batch_image_sizes = True

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        训练前向：直接回归未来动作（无扩散）。

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar supervised action-regression loss.
        """
        examples, qwen_inputs = self._prepare_vlm_inputs(examples)
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        robot_tags = [example.get("robot_tag") or example.get("robot_type") for example in examples]
        action_layout_tags = [example.get("robot_type", example.get("robot_tag")) for example in examples]
        action_valid_masks = [example.get("action_valid_mask") for example in examples]
        action_dim_valid_masks = [example.get("action_dim_valid_mask") for example in examples]

        last_hidden = self._last_hidden(qwen_inputs)  # [B, L, hidden_dim]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 提取动作 token embedding 作为动作预测查询
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)  # [B, chunk_len, H]
            action_queries = self._condition_action_queries(action_queries, examples)
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

            # 标签对齐：取最后 chunk_len 段
            try:
                actions_array = np.asarray(actions)
            except ValueError:
                for idx, action in enumerate(actions):
                    print(
                        "[QwenOFT action shape debug] "
                        f"idx={idx}, "
                        f"shape={getattr(action, 'shape', None)}, "
                        f"type={type(action)}, "
                        f"robot_tag={robot_tags[idx] if idx < len(robot_tags) else None}, "
                        f"robot_type={action_layout_tags[idx] if idx < len(action_layout_tags) else None}, "
                        f"dataset={examples[idx].get('dataset_name') if idx < len(examples) else None}, "
                        f"subset={examples[idx].get('subset_path') if idx < len(examples) else None}",
                        flush=True,
                    )
                raise
            actions = torch.tensor(
                actions_array, device=pred_actions.device, dtype=pred_actions.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)
            action_valid_mask = self._build_action_valid_mask_tensor(
                action_valid_masks,
                device=pred_actions.device,
                dtype=pred_actions.dtype,
            )
            if action_valid_mask is not None:
                action_valid_mask = action_valid_mask[:, -actions_target.shape[1] :]
            action_dim_valid_mask = self._build_action_dim_valid_mask_tensor(
                action_dim_valid_masks,
                action_dim=actions_target.shape[-1],
                device=pred_actions.device,
                dtype=pred_actions.dtype,
            )

            action_dit_loss = self._compute_action_loss(
                pred_actions,
                actions_target,
                robot_tags,
                action_valid_mask=action_valid_mask,
                action_dim_valid_mask=action_dim_valid_mask,
            )

            total_loss = action_dit_loss

        return {
            "action_loss": total_loss,
            "action_dit_loss": action_dit_loss,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], directly regressed actions.
        """
        examples, qwen_inputs = self._prepare_vlm_inputs(examples, inference=True)
        last_hidden = self._last_hidden(qwen_inputs)  # [B, L, hidden_dim]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 提取动作 token embedding 作为动作预测查询
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)  # [B, chunk_len, H]
            action_queries = self._condition_action_queries(action_queries, examples)
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _compute_action_loss(
        self,
        pred_actions: torch.Tensor,
        actions_target: torch.Tensor,
        robot_tags: Optional[List[Optional[str]]] = None,
        action_valid_mask: Optional[torch.Tensor] = None,
        action_dim_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raw_diff = pred_actions - actions_target
        abs_error = torch.abs(raw_diff)
        angular_mask = self._infer_shortest_angular_joint_mask(robot_tags, pred_actions)
        if angular_mask is not None:
            abs_error = blend_angular_wrap_abs_error(abs_error, raw_diff, angular_mask)
        if action_valid_mask is None and action_dim_valid_mask is None:
            return abs_error.mean()

        loss_mask = torch.ones_like(abs_error)
        if action_valid_mask is not None:
            step_mask = action_valid_mask.to(device=abs_error.device, dtype=abs_error.dtype).unsqueeze(-1)
            loss_mask = loss_mask * step_mask
        if action_dim_valid_mask is not None:
            dim_mask = action_dim_valid_mask.to(device=abs_error.device, dtype=abs_error.dtype).unsqueeze(1)
            loss_mask = loss_mask * dim_mask

        denom = loss_mask.sum()
        if denom.item() <= 0:
            # Differentiable zero so DeepSpeed can still backprop through trainable params.
            return pred_actions.sum() * 0.0
        return (abs_error * loss_mask).sum() / denom

    def _build_action_valid_mask_tensor(
        self,
        action_valid_masks,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not action_valid_masks or all(mask is None for mask in action_valid_masks):
            return None

        normalized_masks = []
        for mask in action_valid_masks:
            if mask is None:
                normalized_masks.append(np.ones((self.chunk_len,), dtype=np.float32))
                continue
            arr = np.asarray(mask, dtype=np.float32).reshape(-1)
            normalized_masks.append(arr)
        return torch.tensor(np.stack(normalized_masks, axis=0), device=device, dtype=dtype)

    def _build_action_dim_valid_mask_tensor(
        self,
        action_dim_valid_masks,
        *,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not action_dim_valid_masks or all(mask is None for mask in action_dim_valid_masks):
            return None

        normalized_masks = []
        for mask in action_dim_valid_masks:
            if mask is None:
                normalized_masks.append(np.ones((action_dim,), dtype=np.float32))
                continue
            arr = np.asarray(mask, dtype=np.float32).reshape(-1)
            if arr.shape[0] != action_dim:
                raise ValueError(f"action_dim_valid_mask has shape {arr.shape}, expected ({action_dim},)")
            normalized_masks.append(arr)
        return torch.tensor(np.stack(normalized_masks, axis=0), device=device, dtype=dtype)

    def _infer_shortest_angular_joint_mask(
        self,
        robot_tags: Optional[List[Optional[str]]],
        pred_actions: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_shortest_angular_joint_loss or not robot_tags:
            return None

        batch_size, _, action_dim = pred_actions.shape
        masks = []
        for idx in range(batch_size):
            tag = str(robot_tags[idx]).lower() if idx < len(robot_tags) and robot_tags[idx] is not None else ""
            mask = torch.zeros(action_dim, dtype=torch.bool, device=pred_actions.device)
            gripper_dims = AUTO_ANGULAR_JOINT_LOSS_GRIPPER_DIMS.get(tag)
            if gripper_dims is not None and action_dim > gripper_dims:
                mask[: action_dim - gripper_dims] = True
            masks.append(mask)

        angular_mask = torch.stack(masks, dim=0)
        if not angular_mask.any():
            return None
        return angular_mask

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,   # [B, L, H]
        input_ids: torch.Tensor,     # [B, L]
        action_token_id=None,        # 可为 int 或 List[int]
    ) -> torch.Tensor:
        """
        向量化批量提取动作 token embedding:
          - 不再逐样本 for 循环
          - 取每个样本里最靠后的 chunk_len 个动作占位 token
        Args:
            last_hidden: [B, L, H]
            input_ids:   [B, L]
            action_token_id: int 或 List[int]
        Returns:
            action_queries: [B, chunk_len, H]
        """
        if action_token_id is None:
            raise ValueError("action_token_id 不能为空")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # 支持多 id（如多个变体）
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            # torch.isin 需要 PyTorch >=1.10
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"以下样本动作 token 数量不足 {self.chunk_len}: {insufficient} | counts={counts.tolist()}"
            )

        # 位置索引
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))   # 非动作位置置 -1

        # 取最后 chunk_len 个（索引大的在序列靠后）
        # 注意: 已确保数量足够，不会出现 -1 被错误选中的问题
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values     # [B, chunk_len] 未排序
        # 时间顺序排序
        selected_pos = topk_pos.sort(dim=-1).values                     # [B, chunk_len]

        # Gather
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)   # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.action_model.action_hidden_dim = 2048

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"
    

    # try get model
    model = Qwenvl_OFT(cfg)
    print(model)

    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # two views
        "lang": "This is a fake instruction for testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # two views
        "lang": "For testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")


    # try forward model
    # can be fake sample， but here get from dataloader for simpler
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # zhe
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        batch
        break

    # try get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model(batch)
    pass
    action = model.predict_action(batch)
