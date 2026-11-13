"""Prepare OFT's unchanged Qwen inputs in CPU DataLoader workers.

The opt-in collator loads a processor, never a model. Its output is a plain
mapping so Accelerate can pin/transfer the tensors using its normal loader
path. Sampling and target generation remain the dataset's responsibility.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import random

import numpy as np
import torch

from starVLA.model.modules.vlm.image_preprocessing import preprocess_image_batch


@contextmanager
def _preserve_cpu_rng():
    """Processor imports/initialization must not perturb worker sampling RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def _resolve_local_processor_id(model_id: str) -> str:
    if os.path.isabs(model_id):
        return model_id
    roots = []
    if os.environ.get("STARVLA_ROOT"):
        roots.append(Path(os.environ["STARVLA_ROOT"]))
    roots.extend((Path.cwd(), Path(__file__).resolve().parents[2]))
    for root in roots:
        candidate = (root / model_id).resolve()
        if candidate.exists():
            return str(candidate)
    return model_id


def _load_cpu_processor(model_id: str):
    # Import lazily in the worker. No VLM, checkpoint weights, CUDA context or
    # network lookup is needed for this already-cached training model.
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        _resolve_local_processor_id(model_id), local_files_only=True,
    )
    processor.tokenizer.padding_side = "left"
    return processor


def prepare_qwen_oft_cpu_inputs(
    processor,
    examples,
    *,
    image_size_buckets=None,
    image_size=None,
    chunk_len: int = 32,
    cot_prompt: str | None = None,
):
    """Reproduce conditioned QwenOFT's CPU image/prompt/processor operations.

    The examples are neither modified nor resampled. The exact existing
    processor defaults, image order, chat template and action-token suffix
    are retained. CPU RNG state is restored even if a processor raises.
    """
    if not isinstance(examples, list) or not examples:
        raise ValueError("OFT CPU preprocessing requires a nonempty list of examples")
    if type(chunk_len) is not int or chunk_len <= 0:
        raise ValueError("chunk_len must be a positive integer")
    if cot_prompt is not None and not isinstance(cot_prompt, str):
        raise TypeError("CoT_prompt must be a string when configured")

    with _preserve_cpu_rng():
        images = preprocess_image_batch(
            [example["image"] for example in examples],
            image_size_buckets=image_size_buckets,
            image_size=image_size,
        )
        action_tokens = "🔍" * chunk_len
        suffix = f" Please predict the next {chunk_len} robot actions: <action>{action_tokens}<action>."
        messages = []
        for views, example in zip(images, examples):
            instruction = example["lang"] + suffix
            prompt = cot_prompt.replace("{instruction}", instruction) if cot_prompt is not None else instruction
            content = [{"type": "image", "image": image} for image in views]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])
        inputs = dict(processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ))
        if "input_ids" not in inputs:
            raise ValueError("Qwen processor did not return input_ids")
        if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in inputs.values()):
            raise ValueError("OFT worker preprocessing must return CPU tensors only")
        # Accelerate finds the real batch dimension before encountering the
        # unbatched per-example state tensors or flattened image patches.
        return {"input_ids": inputs.pop("input_ids"), **inputs}


def _plain_size(value):
    if value is None or isinstance(value, (str, int)):
        return value
    return tuple(_plain_size(item) for item in value)


class QwenOFTCPUCollator:
    """Picklable, process-local processor for opt-in conditioned OFT training."""

    def __init__(self, config):
        unwrap = getattr(config, "unwrap", None)
        if callable(unwrap):
            config = unwrap()
        framework = config["framework"]
        conditioning = framework.get("conditioning", {})
        if framework.get("name") != "QwenOFT" or conditioning.get("enabled") is not True:
            raise ValueError("qwen_cpu_preprocessing requires conditioned QwenOFT")
        data = config["datasets"]["vla_data"]
        action = framework["action_model"]
        self.model_id = framework["qwenvl"]["base_vlm"]
        self.image_size_buckets = _plain_size(data.get("image_size_buckets"))
        self.image_size = _plain_size(data.get("image_size"))
        self.chunk_len = action.get("past_action_window_size", 0) + 1 + action["future_action_window_size"]
        self.cot_prompt = data.get("CoT_prompt")
        self._processor = None
        self._processor_pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_processor"] = None
        state["_processor_pid"] = None
        return state

    def __call__(self, examples):
        with _preserve_cpu_rng():
            pid = os.getpid()
            if self._processor is None or self._processor_pid != pid:
                self._processor = _load_cpu_processor(self.model_id)
                self._processor_pid = pid
            qwen_inputs = prepare_qwen_oft_cpu_inputs(
                self._processor,
                examples,
                image_size_buckets=self.image_size_buckets,
                image_size=self.image_size,
                chunk_len=self.chunk_len,
                cot_prompt=self.cot_prompt,
            )
        return {"qwen_inputs": qwen_inputs, "examples": examples}
