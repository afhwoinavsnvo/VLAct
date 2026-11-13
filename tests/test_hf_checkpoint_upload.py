"""Checkpoint uploads preserve training isolation and recoverable local artifacts."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf
import torch

from starVLA.training import train_starvla as training
from starVLA.training.trainer_utils import hf_checkpoint_upload as uploads


class HFCheckpointUploadTests(unittest.TestCase):
    def uploader(self, folder, **kwargs):
        result = uploads.HFCheckpointUploader("research/test-model", folder, **kwargs)
        self.addCleanup(result.close, wait=True)
        return result

    def checkpoint(self, folder, step=10000, *, final=False):
        relative = ("final_model/pytorch_model.pt" if final
                    else f"checkpoints/steps_{step}_pytorch_model.pt")
        result = Path(folder) / relative
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_bytes(b"saved model weights")
        return result

    def queued_job(self, folder, **kwargs):
        uploader = self.uploader(folder, **kwargs)
        with patch.object(uploader, "_run_background_job"):
            manifest = uploader.enqueue(self.checkpoint(folder), 10000)
            uploader.close(wait=True)
        return manifest

    def test_prepare_keeps_parent_offline_and_auth_out_of_command_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            uploader = self.uploader(folder)
            with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "0,1,6,7",
                                         "HF_TOKEN": "test-token-never-in-manifest"}), \
                    patch.object(uploads.subprocess, "run") as run:
                uploader.prepare()
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1,6,7")
                child_env = run.call_args.kwargs["env"]
                self.assertEqual(child_env["HF_HUB_OFFLINE"], "0")
                self.assertEqual(child_env["HF_ENDPOINT"], "https://huggingface.co")
                self.assertEqual(child_env["CUDA_VISIBLE_DEVICES"], "")
                arguments = run.call_args.args[0]
                self.assertNotIn("--private", arguments)
                self.assertNotIn("test-token-never-in-manifest", " ".join(arguments))

    def test_metadata_is_snapshotted_and_only_release_files_are_uploaded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.yaml").write_text("batch: 32\n")
            (root / "dataset_statistics.json").write_text('{"mean": [0]}')
            (root / "token").write_text("do-not-upload")
            (root / "wandb").mkdir()
            (root / "wandb" / "debug.log").write_text("private runtime log")
            uploader = self.uploader(folder)
            with patch.dict(os.environ, {"HF_TOKEN": "test-token-never-in-manifest"}), \
                    patch.object(uploader, "_run_background_job"):
                manifest = uploader.enqueue(self.checkpoint(folder), 10000)
                uploader.close(wait=True)
            (root / "config.yaml").write_text("batch: 64\n")
            job = json.loads(manifest.read_text())
            self.assertEqual(job["checkpoint_repo_path"], "checkpoints/steps_10000_pytorch_model.pt")
            self.assertEqual({item["path_in_repo"] for item in job["metadata"]},
                             {"config.yaml", "dataset_statistics.json"})
            snapshot = next(item for item in job["metadata"] if item["path_in_repo"] == "config.yaml")
            self.assertEqual(Path(snapshot["source_path"]).read_text(), "batch: 32\n")
            self.assertNotIn("test-token-never-in-manifest", manifest.read_text())
            self.assertEqual(Path(job["checkpoint_path"]).read_bytes(), b"saved model weights")

    def test_final_checkpoint_has_distinct_job_and_remote_path(self):
        with tempfile.TemporaryDirectory() as folder:
            uploader = self.uploader(folder)
            with patch.object(uploader, "_run_background_job"):
                periodic = uploader.enqueue(self.checkpoint(folder), 10000)
                final = uploader.enqueue(self.checkpoint(folder, final=True), 10000, final=True)
                uploader.close(wait=True)
            self.assertNotEqual(periodic, final)
            job = json.loads(final.read_text())
            self.assertTrue(job["final"])
            self.assertEqual(job["checkpoint_repo_path"], "final_model/pytorch_model.pt")

    def test_restarting_same_step_keeps_both_immutable_upload_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            uploader = self.uploader(folder)
            source = self.checkpoint(folder)
            with patch.object(uploader, "_run_background_job"):
                first = uploader.enqueue(source, 10000)
                replacement = source.with_suffix(".tmp")
                replacement.write_bytes(b"new weights after restarting training")
                replacement.replace(source)
                second = uploader.enqueue(source, 10000)
                uploader.close(wait=True)
            self.assertNotEqual(first, second)
            first_job = json.loads(first.read_text())
            second_job = json.loads(second.read_text())
            self.assertEqual(Path(first_job["checkpoint_path"]).read_bytes(), b"saved model weights")
            self.assertEqual(Path(second_job["checkpoint_path"]).read_bytes(),
                             b"new weights after restarting training")
            self.assertEqual(first_job["checkpoint_repo_path"], second_job["checkpoint_repo_path"])

    def test_worker_commits_weights_and_matching_metadata_atomically_after_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "config.yaml").write_text("batch: 32\n")
            manifest = self.queued_job(folder, max_retries=2)
            api = MagicMock()
            api.create_commit.side_effect = [RuntimeError("temporary network error"),
                                            SimpleNamespace(commit_url="https://huggingface.co/research/test-model/commit/abc")]
            with patch("huggingface_hub.HfApi", return_value=api), \
                    patch.object(uploads.time, "sleep"):
                uploads._upload_job(manifest)
            self.assertEqual(api.create_commit.call_count, 2)
            kwargs = api.create_commit.call_args.kwargs
            self.assertEqual(kwargs["repo_id"], "research/test-model")
            self.assertEqual({operation.path_in_repo for operation in kwargs["operations"]},
                             {"checkpoints/steps_10000_pytorch_model.pt", "config.yaml"})
            status = json.loads(manifest.with_name("status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["attempt"], 2)
            self.assertTrue(status["commit_url"].endswith("/abc"))

    def test_exhausted_retries_leave_local_checkpoint_and_retry_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = self.queued_job(folder, max_retries=1)
            api = MagicMock()
            api.create_commit.side_effect = RuntimeError("network unavailable")
            with patch("huggingface_hub.HfApi", return_value=api), \
                    patch.object(uploads.time, "sleep"), \
                    self.assertRaisesRegex(RuntimeError, "network unavailable"):
                uploads._upload_job(manifest)
            self.assertEqual(api.create_commit.call_count, 2)
            status = json.loads(manifest.with_name("status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["attempt"], 2)
            job = json.loads(manifest.read_text())
            self.assertEqual(Path(job["checkpoint_path"]).read_bytes(), b"saved model weights")

    def test_upload_queue_is_nonblocking_and_serial(self):
        with tempfile.TemporaryDirectory() as folder:
            uploader = self.uploader(folder)
            entered = threading.Event()
            release = threading.Event()
            started = []

            def child(arguments, **kwargs):
                job = json.loads(Path(arguments[-1]).read_text())
                started.append(job["step"])
                if len(started) == 1:
                    entered.set()
                    if not release.wait(timeout=5):
                        raise RuntimeError("test did not release upload worker")
                return subprocess.CompletedProcess(arguments, 0)

            with patch.object(uploads.subprocess, "run", side_effect=child):
                try:
                    uploader.enqueue(self.checkpoint(folder, 10000), 10000)
                    self.assertTrue(entered.wait(timeout=5))
                    uploader.enqueue(self.checkpoint(folder, 20000), 20000)
                    self.assertEqual(started, [10000])
                finally:
                    release.set()
                    uploader.close(wait=True)
            self.assertEqual(started, [10000, 20000])

    def test_existing_private_repository_is_not_silently_published(self):
        api = MagicMock()
        api.repo_info.return_value = SimpleNamespace(private=True)
        with patch("huggingface_hub.HfApi", return_value=api), \
                self.assertRaisesRegex(RuntimeError, "visibility"):
            uploads._prepare_repo("research/test-model", private=False)
        self.assertFalse(api.create_repo.call_args.kwargs["private"])
        api.update_repo_settings.assert_not_called()


class TrainerCheckpointUploadTests(unittest.TestCase):
    def trainer(self, folder, *, main=True, save_format="pt"):
        trainer = training.VLATrainer.__new__(training.VLATrainer)
        trainer.config = OmegaConf.create({"output_dir": folder, "trainer": {"save_format": save_format}})
        trainer.model = object()
        trainer.checkpoint_dir = str(Path(folder) / "checkpoints")
        Path(trainer.checkpoint_dir).mkdir()
        trainer.completed_steps = 10000
        trainer.accelerator = SimpleNamespace(is_main_process=main, print=MagicMock(),
                                             wait_for_everyone=MagicMock())
        trainer._hf_uploader = MagicMock()
        return trainer

    def test_periodic_upload_is_queued_after_checkpoint_is_fully_saved(self):
        for save_format in ("pt", "safetensors"):
            with self.subTest(save_format=save_format), tempfile.TemporaryDirectory() as folder:
                trainer = self.trainer(folder, save_format=save_format)
                observed = []

                def queued(path, step):
                    observed.append((Path(path), step, Path(path).is_file()))

                trainer._hf_uploader.enqueue.side_effect = queued
                with patch.object(training, "_get_state_dict", return_value={"weight": torch.tensor([1.0])}):
                    trainer._save_checkpoint()
                self.assertEqual(len(observed), 1)
                path, step, exists = observed[0]
                self.assertEqual(step, 10000)
                self.assertTrue(exists)
                self.assertEqual(path.parent, Path(trainer.checkpoint_dir))
                self.assertTrue(path.name.endswith(".pt" if save_format == "pt" else ".safetensors"))
                trainer.accelerator.wait_for_everyone.assert_called_once()

    def test_nonmain_rank_neither_saves_nor_uploads(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.trainer(folder, main=False)
            with patch.object(training, "_get_state_dict") as state:
                trainer._save_checkpoint()
                trainer._finalize_training()
            state.assert_not_called()
            trainer._hf_uploader.enqueue.assert_not_called()
            trainer._hf_uploader.close.assert_not_called()

    def test_final_upload_starts_without_waiting_on_network_before_ddp_barrier(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.trainer(folder)
            events = []
            trainer._hf_uploader.enqueue.side_effect = lambda *args, **kwargs: events.append("enqueue")
            trainer._hf_uploader.close.side_effect = lambda **kwargs: events.append(("close", kwargs["wait"]))
            trainer.accelerator.wait_for_everyone.side_effect = lambda: events.append("barrier")
            with patch.object(training, "_get_state_dict", return_value={"weight": torch.tensor([1.0])}), \
                    patch.object(training, "logger"):
                trainer._finalize_training()
            trainer._hf_uploader.enqueue.assert_called_once_with(
                str(Path(folder) / "final_model" / "pytorch_model.pt"), 10000, final=True)
            self.assertEqual(events, ["enqueue", ("close", False), "barrier"])

    def test_disabled_config_preserves_existing_training_without_hf_setup(self):
        config = OmegaConf.create({"trainer": {}})
        accelerator = SimpleNamespace(is_main_process=True, wait_for_everyone=MagicMock())
        with patch.object(training, "HFCheckpointUploader") as constructor:
            self.assertIsNone(training.initialize_hf_upload(config, accelerator))
        constructor.assert_not_called()

    def test_setup_failure_is_reported_before_training_and_closes_uploader(self):
        with tempfile.TemporaryDirectory() as folder:
            config = OmegaConf.create({"output_dir": folder, "trainer": {"hf_upload": {
                "enabled": True, "repo_id": "research/test-model", "private": False,
            }}})
            accelerator = SimpleNamespace(is_main_process=True, wait_for_everyone=MagicMock())
            uploader = MagicMock()
            uploader.prepare.side_effect = RuntimeError("token cannot write")
            with patch.object(training, "HFCheckpointUploader", return_value=uploader), \
                    patch.object(training.dist, "is_initialized", return_value=False), \
                    self.assertRaisesRegex(RuntimeError, "before model loading"):
                training.initialize_hf_upload(config, accelerator)
            uploader.close.assert_called_once_with(wait=False)
            accelerator.wait_for_everyone.assert_not_called()


if __name__ == "__main__":
    unittest.main()
