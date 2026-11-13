"""CPU preprocessing keeps OFT inputs and worker sampling RNGs unchanged."""

import ast
from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from starVLA.dataloader import oft_cpu_preprocessing as prep
from starVLA.model.modules.vlm.image_preprocessing import preprocess_image_batch


def configuration():
    return OmegaConf.create({
        "framework": {
            "name": "QwenOFT",
            "conditioning": {"enabled": True},
            "qwenvl": {"base_vlm": "cached-test-processor"},
            "action_model": {"past_action_window_size": 0, "future_action_window_size": 31},
        },
        "datasets": {"vla_data": {
            "image_size_buckets": [[32, 16], [24, 24]],
            "image_size": [24, 24],
            "CoT_prompt": "Robot prompt: {instruction}. End.",
        }},
    })


def examples():
    rng = np.random.default_rng(728)
    return [{
        "image": [Image.fromarray(rng.integers(0, 256, (17 + i, 31 - i, 3), dtype=np.uint8)) for i in range(3)],
        "lang": text,
        "state": torch.arange(14, dtype=torch.float32).reshape(1, 14),
        "target_tcp_pose": rng.standard_normal(14).astype(np.float32),
        "action": rng.standard_normal((32, 14)).astype(np.float32),
        "trajectory_id": idx,
    } for idx, text in enumerate(("Pick the red cup", "Place the blue bottle beside it"))]


def consume_cpu_rng():
    random.random()
    np.random.random()
    torch.rand(1)


class RecordingProcessor:
    def __init__(self, consume_rng=False, fail=False):
        self.consume_rng = consume_rng
        self.fail = fail
        self.calls = []
        self.tokenizer = SimpleNamespace(padding_side="left")

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.consume_rng:
            consume_cpu_rng()
        if self.fail:
            raise RuntimeError("processor failed")
        # Deliberately put flattened image features first; the helper must
        # restore input_ids-first order for Accelerate's batch-size inference.
        return {
            "pixel_values": torch.ones(7, 4),
            "input_ids": torch.arange(len(messages) * 35).reshape(len(messages), 35),
            "attention_mask": torch.ones(len(messages), 35, dtype=torch.int64),
        }


class WorkerRandomSamples(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {
            "image": [Image.new("RGB", (8, 6), (index, 10, 20))],
            "lang": f"Instruction {index}",
            "index": index,
            "rng_draws": (random.random(), float(np.random.random()), float(torch.rand(()))),
            "target_tcp_pose": np.random.random(14).astype(np.float32),
            "worker_id": torch.utils.data.get_worker_info().id,
        }


def identity_collate(batch):
    return batch


class QwenOFTCPUPreprocessingTests(unittest.TestCase):
    def assert_rng_equal(self, before):
        python_rng, numpy_rng, torch_rng = before
        self.assertEqual(random.getstate(), python_rng)
        after = np.random.get_state()
        self.assertEqual(after[0], numpy_rng[0])
        np.testing.assert_array_equal(after[1], numpy_rng[1])
        self.assertEqual(after[2:], numpy_rng[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))

    @staticmethod
    def rng_states():
        return random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()

    def test_exact_suffix_prompt_image_order_and_examples_preserved(self):
        batch = examples()
        original_images = [[np.asarray(image).copy() for image in sample["image"]] for sample in batch]
        processor = RecordingProcessor()
        cfg = configuration()
        collator = prep.QwenOFTCPUCollator(cfg)
        with patch.object(prep, "_load_cpu_processor", return_value=processor) as load:
            self.assertEqual(load.call_count, 0)
            prepared = collator(batch)
        self.assertIs(prepared["examples"], batch)
        self.assertEqual(list(prepared), ["qwen_inputs", "examples"])
        self.assertEqual(next(iter(prepared["qwen_inputs"])), "input_ids")
        load.assert_called_once_with("cached-test-processor")
        messages, kwargs = processor.calls[0]
        self.assertEqual(kwargs, dict(tokenize=True, padding=True, add_generation_prompt=True,
                                     return_dict=True, return_tensors="pt"))
        suffix = " Please predict the next 32 robot actions: <action>" + "🔍" * 32 + "<action>."
        for idx, (sample, message) in enumerate(zip(batch, messages)):
            content = message[0]["content"]
            self.assertEqual(message[0]["role"], "user")
            self.assertEqual(content[-1], {"type": "text", "text": "Robot prompt: " + sample["lang"] + suffix + ". End."})
            expected = preprocess_image_batch(sample["image"], image_size_buckets=[[32, 16], [24, 24]], image_size=[24, 24])
            for view_idx, (actual, target) in enumerate(zip(content[:-1], expected)):
                self.assertEqual(actual["type"], "image")
                np.testing.assert_array_equal(np.asarray(actual["image"]), np.asarray(target))
                np.testing.assert_array_equal(np.asarray(sample["image"][view_idx]), original_images[idx][view_idx])

    def test_first_initialization_and_every_call_preserve_all_cpu_rngs(self):
        processor = RecordingProcessor(consume_rng=True)

        def load(_):
            consume_cpu_rng()
            return processor

        batch = examples()
        collator = prep.QwenOFTCPUCollator(configuration())
        before = self.rng_states()
        with patch.object(prep, "_load_cpu_processor", side_effect=load) as loader:
            collator(batch)
            self.assert_rng_equal(before)
            collator(batch)
            self.assert_rng_equal(before)
            self.assertEqual(loader.call_count, 1)

    def test_processor_failure_also_restores_worker_rngs(self):
        batch = examples()
        processor = RecordingProcessor(consume_rng=True, fail=True)
        before = self.rng_states()
        with self.assertRaisesRegex(RuntimeError, "processor failed"):
            prep.prepare_qwen_oft_cpu_inputs(processor, batch)
        self.assert_rng_equal(before)

    def test_pickling_and_pid_changes_reset_processor(self):
        collator = prep.QwenOFTCPUCollator(configuration())
        batch = examples()
        with patch.object(prep, "_load_cpu_processor", side_effect=lambda _: RecordingProcessor()) as load:
            collator(batch)
            first = collator._processor
            restored = pickle.loads(pickle.dumps(collator))
            self.assertIsNone(restored._processor)
            restored(batch)
            self.assertEqual(load.call_count, 2)
            with patch.object(prep.os, "getpid", return_value=os.getpid() + 1):
                collator(batch)
            self.assertIsNot(collator._processor, first)
            self.assertEqual(load.call_count, 3)

    def test_unsupported_framework_and_empty_batch_fail_clearly(self):
        cfg = configuration()
        cfg.framework.name = "QwenPI"
        with self.assertRaisesRegex(ValueError, "conditioned QwenOFT"):
            prep.QwenOFTCPUCollator(cfg)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            prep.prepare_qwen_oft_cpu_inputs(RecordingProcessor(), [])

    def test_access_tracked_config_is_snapshotted_without_worker_wrapper(self):
        from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig

        cfg = AccessTrackedConfig(configuration())
        collator = prep.QwenOFTCPUCollator(cfg)
        self.assertEqual(collator.image_size_buckets, ((32, 16), (24, 24)))
        self.assertEqual(collator.model_id, "cached-test-processor")
        self.assertEqual(collator.chunk_len, 32)

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "Requires Linux fork workers")
    def test_two_worker_sampling_and_target_rng_sequence_matches_original_collate(self):
        def collect(collate):
            loader = torch.utils.data.DataLoader(
                WorkerRandomSamples(), batch_size=2, num_workers=2,
                collate_fn=collate, multiprocessing_context="fork",
                generator=torch.Generator().manual_seed(684),
            )
            result = []
            for batch in loader:
                result.extend(batch["examples"] if isinstance(batch, dict) else batch)
            return result

        baseline = collect(identity_collate)

        def noisy_processor_load(_):
            consume_cpu_rng()
            return RecordingProcessor(consume_rng=True)

        with patch.object(prep, "_load_cpu_processor", side_effect=noisy_processor_load):
            optimized = collect(prep.QwenOFTCPUCollator(configuration()))
        self.assertEqual({sample["worker_id"] for sample in optimized}, {0, 1})
        self.assertEqual(len(baseline), len(optimized))
        for old, new in zip(baseline, optimized):
            self.assertEqual(old["index"], new["index"])
            self.assertEqual(old["worker_id"], new["worker_id"])
            self.assertEqual(old["rng_draws"], new["rng_draws"])
            np.testing.assert_array_equal(old["target_tcp_pose"], new["target_tcp_pose"])

    def test_accelerate_infers_batch_size_and_transfers_mapping(self):
        from accelerate.utils import find_batch_size, send_to_device

        batch = examples()
        collator = prep.QwenOFTCPUCollator(configuration())
        with patch.object(prep, "_load_cpu_processor", return_value=RecordingProcessor()):
            prepared = collator(batch)
        self.assertEqual(find_batch_size(prepared), len(batch))
        transferred = send_to_device(prepared, torch.device("cpu"), non_blocking=True)
        self.assertTrue(torch.equal(transferred["qwen_inputs"]["input_ids"], prepared["qwen_inputs"]["input_ids"]))
        self.assertEqual(transferred["examples"][0]["lang"], batch[0]["lang"])
        np.testing.assert_array_equal(transferred["examples"][0]["target_tcp_pose"], batch[0]["target_tcp_pose"])

    def test_real_cached_processor_matches_original_method_bitwise(self):
        candidates = list(Path("/scratch/jliu3375/huggingface/hub/models--StarVLA--Qwen3-VL-4B-Instruct-Action/snapshots").glob("*/preprocessor_config.json"))
        configured = os.environ.get("VLACT_TEST_PROCESSOR_PATH")
        if not configured and not candidates:
            self.skipTest("No local Qwen processor cache; set VLACT_TEST_PROCESSOR_PATH")
        processor_path = Path(configured) if configured else candidates[0].parent
        batch = examples()
        cfg = configuration()
        cfg.framework.qwenvl.base_vlm = str(processor_path)
        collator = prep.QwenOFTCPUCollator(cfg)
        before = self.rng_states()
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU preprocessing initialized CUDA")):
            prepared = collator(batch)["qwen_inputs"]
        self.assert_rng_equal(before)

        # Compile only the original processor method to avoid constructing or
        # importing a VLM in this CPU equivalence test.
        source = Path(__file__).resolve().parents[1] / "starVLA/model/modules/vlm/QWen3.py"
        module = ast.parse(source.read_text())
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "_QWen3_VL_Interface")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "build_qwenvl_inputs")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        reference = SimpleNamespace(processor=collator._processor, config=cfg,
                                    model=SimpleNamespace(device=torch.device("cpu")))
        images = preprocess_image_batch([sample["image"] for sample in batch],
                                        image_size_buckets=cfg.datasets.vla_data.image_size_buckets,
                                        image_size=cfg.datasets.vla_data.image_size)
        suffix = " Please predict the next 32 robot actions: <action>" + "🔍" * 32 + "<action>."
        expected = namespace["build_qwenvl_inputs"](reference, images=images,
                                                    instructions=[sample["lang"] + suffix for sample in batch])
        self.assertEqual(set(expected), set(prepared))
        for key in expected:
            self.assertTrue(torch.equal(expected[key], prepared[key]), key)


if __name__ == "__main__":
    unittest.main()
