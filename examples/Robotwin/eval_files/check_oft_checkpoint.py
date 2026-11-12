#!/usr/bin/env python3
r"""Check a released RoboTwin OFT checkpoint with one real three-camera sample.

This script does not train a model, change checkpoint files, or execute robot
actions. It checks strict checkpoint loading, action shape, and finite outputs.
It does not measure RoboTwin task success or apply action denormalization.

Example (run from the VLAct repository):
    python examples/Robotwin/eval_files/check_oft_checkpoint.py \
        --checkpoint /scratch/jliu3375/checkpoints/VLAct-OFT-Robotwin-Clean-wrap32/checkpoints/steps_50000_pytorch_model.pt \
        --sample-dir /scratch/jliu3375/datasets/robotwin_official_randomized/adjust_bottle/episode_000000 \
        --hf-cache /scratch/jliu3375/huggingface/hub --offline

Use --validate-only to check local input files without importing the model or
using a GPU. For inference, CUDA_VISIBLE_DEVICES selects the physical GPU;
--device defaults to cuda:0 within that visible set.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Released .pt file inside a checkpoints directory.")
    parser.add_argument(
        "--sample-dir", type=Path, required=True,
        help="Directory containing sample.json and raw_cam_*.png (legacy 03_train_sample.json is also supported).",
    )
    parser.add_argument("--hf-cache", type=Path, help="Hugging Face hub cache containing the matching Action base model.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device within CUDA_VISIBLE_DEVICES (default: cuda:0).")
    parser.add_argument("--offline", action="store_true", help="Load the base model entirely from the local Hugging Face cache.")
    parser.add_argument("--validate-only", action="store_true", help="Validate input files and configuration without loading model weights.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; no report is written by default.")
    return parser.parse_args()


def validate_inputs(checkpoint, sample_dir):
    import yaml

    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint is missing or empty: {checkpoint}")
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("Keep the released layout: RUN/checkpoints/FILE.pt, RUN/config.yaml, RUN/dataset_statistics.json.")

    run_dir = checkpoint.parent.parent
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    statistics = json.loads((run_dir / "dataset_statistics.json").read_text())
    if config["framework"]["name"] != "QwenOFT":
        raise ValueError("This checker supports QwenOFT checkpoints only.")
    if not isinstance(statistics, dict) or not statistics:
        raise ValueError("dataset_statistics.json must contain a nonempty statistics mapping.")

    action_cfg = config["framework"]["action_model"]
    horizon = int(action_cfg["past_action_window_size"]) + 1 + int(action_cfg["future_action_window_size"])
    action_dim = int(action_cfg["action_dim"])
    if horizon <= 0 or action_dim <= 0:
        raise ValueError("The checkpoint must declare a positive action horizon and action dimension.")

    sample_path = sample_dir / "sample.json"
    if not sample_path.is_file():
        sample_path = sample_dir / "03_train_sample.json"
    sample = json.loads(sample_path.read_text())
    instruction = sample.get("lang")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{sample_path.name} must contain a nonempty 'lang' string.")
    for camera in CAMERAS:
        image_path = sample_dir / f"raw_{camera}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Sample image is missing: {image_path}")

    provenance_path = sample_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
    return config, instruction, (1, horizon, action_dim), list(statistics), sample_path, provenance


def load_images(sample_dir, data_cfg):
    import cv2
    import numpy as np
    from PIL import Image

    # The upstream RoboTwin client treats bucket pairs as (width, height).
    buckets = [tuple(map(int, pair)) for pair in data_cfg.get("image_size_buckets", [])]
    fixed_size = data_cfg.get("image_size")
    if any(len(pair) != 2 or min(pair) <= 0 for pair in buckets):
        raise ValueError("Image buckets must contain positive (width, height) pairs.")

    images = []
    for camera in CAMERAS:
        with Image.open(sample_dir / f"raw_{camera}.png") as image:
            pixels = np.asarray(image.convert("RGB"))
        height, width = pixels.shape[:2]
        if buckets:
            target = min(buckets, key=lambda pair: abs(pair[0] / pair[1] - width / height))
        elif fixed_size:
            target = tuple(map(int, fixed_size))
        else:
            target = (width, height)
        pixels = cv2.resize(pixels, target, interpolation=cv2.INTER_AREA)
        images.append(Image.fromarray(pixels))
    return images


def main():
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    sample_dir = args.sample_dir.expanduser().resolve()

    # Set cache options before importing Transformers or Hugging Face Hub.
    if args.hf_cache:
        os.environ["HF_HUB_CACHE"] = str(args.hf_cache.expanduser().resolve())
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Prefer this checkout over another editable starVLA installation.
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    config, instruction, expected_shape, statistics_keys, sample_path, provenance = validate_inputs(
        checkpoint, sample_dir
    )
    report = {
        "checkpoint": str(checkpoint),
        "sample_dir": str(sample_dir),
        "sample_file": str(sample_path),
        "instruction": instruction,
        "sample_provenance": provenance,
        "scope": "strict_checkpoint_loading_and_single_recorded_sample_inference",
        "base_model": config["framework"]["qwenvl"]["base_vlm"],
        "expected_shape": list(expected_shape),
        "statistics_keys": statistics_keys,
        "state_input": False,
        "status": "inputs_validated",
    }
    print(f"Input files validated. Expected action shape: {expected_shape}", flush=True)

    if not args.validate_only:
        import numpy as np
        import torch
        from accelerate import PartialState
        from starVLA.model.framework.base_framework import baseframework

        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Inference requires CUDA: the upstream VLM uses FlashAttention 2. Use --validate-only for a CPU check.")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        PartialState()
        torch.cuda.set_device(device)

        images = load_images(sample_dir, config["datasets"]["vla_data"])
        print("Loading the checkpoint with strict parameter matching...", flush=True)
        start = time.perf_counter()
        model = baseframework.from_pretrained(str(checkpoint))
        model = model.to(dtype=torch.bfloat16, device=device).eval()
        torch.cuda.synchronize(device)
        report["load_seconds"] = time.perf_counter() - start

        # Upstream predict_action converts the MLP output directly to NumPy,
        # which cannot represent BF16 tensors. Cast only the final MLP output;
        # preserve the upstream BF16 computation and all checkpoint parameters.
        hook = model.action_model.model.register_forward_hook(
            lambda module, inputs, output: output.float()
        )
        try:
            print("Running one three-camera inference without a state input...", flush=True)
            start = time.perf_counter()
            with torch.inference_mode():
                result = model.predict_action(examples=[{"image": images, "lang": instruction}])
            torch.cuda.synchronize(device)
            report["first_inference_seconds"] = time.perf_counter() - start
        finally:
            hook.remove()

        actions = np.asarray(result["normalized_actions"])
        if actions.shape != expected_shape:
            raise RuntimeError(f"Action shape mismatch: expected {expected_shape}, received {actions.shape}.")
        if not np.isfinite(actions).all():
            raise RuntimeError("Model output contains NaN or infinite values.")
        report.update(
            status="passed",
            device=str(device),
            action_shape=list(actions.shape),
            first_action=actions[0, 0].tolist(),
        )
        print(f"PASS: strict checkpoint loading and one inference succeeded: {actions.shape}", flush=True)
        print("This check does not measure RoboTwin task success or training throughput.", flush=True)
    else:
        print("PASS: input validation only; model weights were not loaded and no GPU was used.", flush=True)

    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Report saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
