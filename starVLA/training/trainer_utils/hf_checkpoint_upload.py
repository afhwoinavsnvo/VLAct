"""Upload saved checkpoints without changing the training process's offline mode."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


logger = logging.getLogger(__name__)
_METADATA_FILES = (
    "config.yaml", "config.full.yaml", "dataset_statistics.json",
    "tcp_pose_statistics.json", "target_tcp_contract.json", "state_statistics.json",
)


def _child_env():
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="0", HF_ENDPOINT="https://huggingface.co", CUDA_VISIBLE_DEVICES="")
    return env


def _write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _status(manifest_path, state, **fields):
    _write_json(Path(manifest_path).with_name("status.json"), {
        "status": state, "updated_at": datetime.now(timezone.utc).isoformat(), **fields,
    })


def _prepare_repo(repo_id, private):
    # Import only in a fresh child: huggingface_hub reads offline mode at import.
    from huggingface_hub import HfApi

    api = HfApi(endpoint="https://huggingface.co", token=True)
    api.whoami()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if bool(info.private) != private:
        raise RuntimeError(f"Repository {repo_id} visibility differs from private={private}.")
    print(f"HF upload destination ready: https://huggingface.co/{repo_id}", flush=True)


def _upload_job(manifest_path):
    from huggingface_hub import CommitOperationAdd, HfApi

    manifest_path = Path(manifest_path).resolve()
    job = json.loads(manifest_path.read_text(encoding="utf-8"))
    api = HfApi(endpoint="https://huggingface.co", token=True)
    attempts = int(job["max_retries"]) + 1
    for attempt in range(1, attempts + 1):
        _status(manifest_path, "uploading", attempt=attempt)
        try:
            files = [{"source_path": job["checkpoint_path"],
                      "path_in_repo": job["checkpoint_repo_path"]}, *job["metadata"]]
            operations = [CommitOperationAdd(path_in_repo=item["path_in_repo"],
                                             path_or_fileobj=item["source_path"])
                          for item in files]
            commit = api.create_commit(
                repo_id=job["repo_id"], repo_type="model", operations=operations,
                commit_message=f"Upload step {job['step']}" + (" (final)" if job["final"] else ""),
            )
            commit_url = str(getattr(commit, "commit_url", commit))
            _status(manifest_path, "complete", attempt=attempt, commit_url=commit_url)
            print(f"Checkpoint upload complete: {commit_url}", flush=True)
            return
        except Exception as exc:
            last = attempt == attempts
            _status(manifest_path, "failed" if last else "retrying",
                    attempt=attempt, error=str(exc))
            print(f"Checkpoint upload attempt {attempt}/{attempts} failed: {exc}", flush=True)
            if last:
                raise
            time.sleep(min(30, 2 ** attempt))


class HFCheckpointUploader:
    """Queue immutable local weights and whitelisted metadata for serial upload."""

    def __init__(self, repo_id, output_dir, private=False, max_retries=3):
        if int(max_retries) < 0:
            raise ValueError("max_retries must be nonnegative")
        self.repo_id = repo_id
        self.output_dir = Path(output_dir).resolve()
        self.private = bool(private)
        self.max_retries = int(max_retries)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-checkpoint")
        self._closed = False

    def prepare(self):
        """Authenticate and ensure destination visibility before starting training."""
        args = [sys.executable, str(Path(__file__).resolve()), "--prepare", "--repo-id", self.repo_id]
        if self.private:
            args.append("--private")
        subprocess.run(args, env=_child_env(), check=True)

    def enqueue(self, checkpoint_path, step, final=False):
        """Snapshot metadata; return its retry manifest, or log a local failure."""
        try:
            if self._closed:
                raise RuntimeError("Checkpoint uploader is closed")
            step = int(step)
            if step < 0:
                raise ValueError("step must be nonnegative")
            checkpoint_path = Path(checkpoint_path).resolve()
            relative_path = checkpoint_path.relative_to(self.output_dir)
            if relative_path.parts[0] not in ("checkpoints", "final_model") or not checkpoint_path.is_file():
                raise ValueError("Checkpoint must be a saved file in output_dir/checkpoints or final_model")
            upload_dir = self.output_dir / "hf_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            job_dir = Path(tempfile.mkdtemp(
                prefix=f"steps_{step}{'_final' if final else ''}_", dir=upload_dir,
            ))
            manifest_path = job_dir / "job.json"
            # The trainer saves with os.replace, so later saves cannot mutate
            # the inode referenced by this link; retry jobs retain their weights.
            snapshot_weight = job_dir / checkpoint_path.name
            os.link(checkpoint_path, snapshot_weight)
            metadata = []
            for name in _METADATA_FILES:
                source = self.output_dir / name
                if source.is_file() and source.resolve().parent == self.output_dir:
                    snapshot = job_dir / name
                    shutil.copy2(source, snapshot)
                    metadata.append({"source_path": str(snapshot), "path_in_repo": name})
            _write_json(manifest_path, {
                "repo_id": self.repo_id, "step": step, "final": bool(final),
                "checkpoint_path": str(snapshot_weight), "checkpoint_repo_path": relative_path.as_posix(),
                "metadata": metadata, "max_retries": self.max_retries,
            })
            _status(manifest_path, "queued", attempt=0)
            self._executor.submit(self._run_background_job, manifest_path)
            logger.info("HF checkpoint upload queued; status and logs: %s", job_dir)
            return manifest_path
        except Exception:
            logger.exception("Could not queue HF checkpoint upload; local checkpoint retained")
            return None

    def _run_background_job(self, manifest_path):
        try:
            with manifest_path.with_name("upload.log").open("a", encoding="utf-8") as log:
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--job", str(manifest_path)],
                    env=_child_env(), stdout=log, stderr=subprocess.STDOUT, check=True,
                )
            logger.info("HF checkpoint upload complete: %s", manifest_path)
        except Exception as exc:
            status_path = manifest_path.with_name("status.json")
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("status") != "failed":
                    _status(manifest_path, "failed", error=str(exc))
            except Exception:
                logger.exception("Could not persist HF upload failure status: %s", manifest_path)
            logger.exception("HF upload failed; checkpoint retained. Retry: %s %s --job %s",
                             sys.executable, Path(__file__).resolve(), manifest_path)

    def close(self, wait=False):
        """Stop accepting jobs; queued uploads drain and Python joins them at exit."""
        self._closed = True
        self._executor.shutdown(wait=wait)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--job", type=Path, help="Retry an upload manifest without starting training")
    parser.add_argument("--repo-id")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    # The CLI can also be invoked directly to retry a job from an offline shell.
    os.environ.update(HF_HUB_OFFLINE="0", HF_ENDPOINT="https://huggingface.co", CUDA_VISIBLE_DEVICES="")
    if args.prepare:
        if not args.repo_id:
            parser.error("--prepare requires --repo-id")
        _prepare_repo(args.repo_id, args.private)
    else:
        _upload_job(args.job)


if __name__ == "__main__":
    main()
