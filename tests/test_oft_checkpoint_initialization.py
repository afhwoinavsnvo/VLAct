"""Restore only the checkpoint VLM and optimize only fresh OFT modules."""

from pathlib import Path
import json
import tempfile
import unittest

from omegaconf import OmegaConf
import torch
from torch import nn

from starVLA.model.modules.action_model.oft_conditioning import OFTConditioning
from starVLA.model.framework.share_tools import read_mode_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups


class SmallOFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_vl_interface = nn.Linear(4, 4)
        self.action_model = nn.Linear(4, 2)
        self.conditioning = OFTConditioning(
            hidden_dim=4, encoder_hidden_dim=8,
            include_state=True, include_tcp=True, enabled=True,
        )


class OFTCheckpointInitializationTests(unittest.TestCase):
    def test_backbone_only_reload_preserves_fresh_head_and_conditioning_for_both_pt_formats(self):
        source = SmallOFT()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(7.)
        for zip_format in (False, True):
            with self.subTest(zip_format=zip_format), tempfile.TemporaryDirectory() as folder:
                target = SmallOFT()
                target.qwen_vl_interface.requires_grad_(False)
                original = {key: value.clone() for key, value in target.state_dict().items()}
                checkpoint = Path(folder) / "model.pt"
                torch.save(source.state_dict(), checkpoint, _use_new_zipfile_serialization=zip_format)
                result = TrainerUtils.load_pretrained_backbones(
                    target, checkpoint_path=checkpoint, reload_modules="qwen_vl_interface",
                )
                self.assertIs(result, target)
                for key, value in target.state_dict().items():
                    expected = source.state_dict()[key] if key.startswith("qwen_vl_interface.") else original[key]
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                self.assertTrue(all(not p.requires_grad for p in target.qwen_vl_interface.parameters()))

    def test_missing_backbone_prefix_is_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "head_only.pt"
            torch.save({"action_model.weight": torch.zeros(2, 4)}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "absent"):
                TrainerUtils.load_pretrained_backbones(
                    SmallOFT(), checkpoint_path=checkpoint, reload_modules="qwen_vl_interface",
                )

    def test_partial_backbone_reload_requires_all_requested_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "incomplete.pt"
            torch.save({"qwen_vl_interface.weight": torch.zeros(4, 4)}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "bias"):
                TrainerUtils.load_pretrained_backbones(
                    SmallOFT(), checkpoint_path=checkpoint, reload_modules="qwen_vl_interface",
                )

    def test_optimizer_excludes_frozen_backbone_and_deduplicates_overlapping_modules(self):
        model = SmallOFT()
        model.qwen_vl_interface.requires_grad_(False)
        cfg = OmegaConf.create({"trainer": {
            "freeze_modules": "",  # requires_grad alone must exclude the VLM.
            "learning_rate": {
                "base": 1e-4, "qwen_vl_interface": 1e-5,
                "conditioning": 1e-4, "conditioning.output_projection": 1e-4,
                "conditioning.state_encoder": 1e-4, "action_model": 1e-4,
            },
        }})
        groups = build_param_lr_groups(model, cfg)
        actual = [id(parameter) for group in groups for parameter in group["params"]]
        expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertNotIn("qwen_vl_interface", [group["name"] for group in groups])
        optimizer = torch.optim.AdamW(groups)
        initial_vlm = model.qwen_vl_interface.weight.clone()
        initial_head = model.action_model.weight.clone()
        with torch.no_grad():
            features = model.qwen_vl_interface(torch.randn(3, 4)).unsqueeze(1)
        fused = model.conditioning(features, state=torch.randn(3, 14), tcp=torch.randn(3, 14))
        model.action_model(fused).square().mean().backward()
        optimizer.step()
        torch.testing.assert_close(model.qwen_vl_interface.weight, initial_vlm, rtol=0, atol=0)
        self.assertIsNone(model.qwen_vl_interface.weight.grad)
        self.assertFalse(torch.equal(model.action_model.weight, initial_head))

    def test_optimizer_honors_explicit_freeze_path_before_freeze_backbones_runs(self):
        model = SmallOFT()
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        cfg = OmegaConf.create({"trainer": {
            "freeze_modules": "qwen_vl_interface",
            "learning_rate": {"base": 1e-4, "qwen_vl_interface": 1e-5},
        }})
        groups = build_param_lr_groups(model, cfg)
        actual = {id(parameter) for group in groups for parameter in group["params"]}
        frozen = {id(parameter) for parameter in model.qwen_vl_interface.parameters()}
        expected = {id(parameter) for parameter in model.parameters()} - frozen
        self.assertEqual(actual, expected)

    def test_reset_dataloader_advances_dataset_and_sampler_epochs_in_real_samples(self):
        class RecordingDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.epoch = 0
                self.epochs = []

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return {"index": index, "epoch": self.epoch}

            def set_epoch(self, epoch):
                self.epoch = epoch
                self.epochs.append(epoch)

        class RecordingSampler(torch.utils.data.Sampler):
            def __init__(self, dataset):
                self.dataset = dataset
                self.epochs = []

            def __iter__(self):
                return iter(range(len(self.dataset)))

            def __len__(self):
                return len(self.dataset)

            def set_epoch(self, epoch):
                self.epochs.append(epoch)

        dataset = RecordingDataset()
        sampler = RecordingSampler(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)
        self.assertEqual(next(iter(loader))["epoch"].item(), 0)
        epoch = 0
        for expected in (1, 2):
            iterator, epoch = TrainerUtils._reset_dataloader(loader, epoch)
            self.assertEqual(epoch, expected)
            self.assertEqual(next(iterator)["epoch"].item(), expected)
        self.assertEqual(dataset.epochs, [1, 2])
        self.assertEqual(sampler.epochs, [1, 2])

    def test_inference_config_merges_full_snapshot_with_accessed_values_for_both_save_locations(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            full = OmegaConf.create({
                "framework": {"name": "QwenOFT", "conditioning": {
                    "enabled": True, "include_state": True, "include_tcp": True,
                    "freeze_vlm": True, "encoder_hidden_dim": 1024, "state_dim": 14, "tcp_dim": 14,
                }},
                "datasets": {"vla_data": {"image_size_buckets": "[(288,288),(384,288)]"}},
            })
            # The access tracker may omit defaults that were never read.
            sparse = OmegaConf.create({"framework": {"conditioning": {"include_state": False}}})
            OmegaConf.save(full, run / "config.full.yaml")
            OmegaConf.save(sparse, run / "config.yaml")
            statistics = {"robotwin_wrap_32": {"action": {"mean": [0.] * 14}}}
            (run / "dataset_statistics.json").write_text(json.dumps(statistics))
            for directory in ("checkpoints", "final_model"):
                with self.subTest(directory=directory):
                    checkpoint = run / directory / "model.pt"
                    checkpoint.parent.mkdir()
                    checkpoint.touch()  # Config discovery does not deserialize weights.
                    config, actual_statistics = read_mode_config(checkpoint)
                    condition = config["framework"]["conditioning"]
                    self.assertFalse(condition["include_state"])
                    self.assertTrue(condition["enabled"] and condition["include_tcp"] and condition["freeze_vlm"])
                    self.assertEqual(condition["state_dim"], 14)
                    self.assertEqual(condition["tcp_dim"], 14)
                    self.assertEqual(condition["encoder_hidden_dim"], 1024)
                    self.assertEqual(config["framework"]["name"], "QwenOFT")
                    self.assertEqual(actual_statistics, statistics)


if __name__ == "__main__":
    unittest.main()
