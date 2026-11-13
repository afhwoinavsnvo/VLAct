# 🚀 RoboTwin 2.0 Evaluation

This document provides instructions for reproducing our **experimental results** with [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin).  

## Local VLAct OFT with state and target TCP

The original joint-state experiment uses this VLAct checkout (`feat/oft-state-tcp`),
`train_files/starvla_oft_state_tcp.yaml`, and the actual TCP-enriched data root:

```text
/scratch/jliu3375/datasets/RoboTwin-Clean-TCP
```

This is the selected **Clean-last** TCP variant, with 50 tasks, 2,500 episodes
and 549,787 original training frames. The `robotwin_wrap_32` mix name selects
the original task list and wrapped-angle action schema; the configured data
root selects the TCP dataset. It does not redirect training to plain Clean.
The existing manifest's schema-2 goal policy is preserved: sample from
`[t+64, t+96)` when the complete window fits, otherwise use the last Clean
observation. The additional RAW terminal frame is not a target in this variant.

The model path is:

```text
frozen checkpoint VLM -> 32 action tokens -> trainable output projection
                                             + State Encoder(current state)
                                             + TCP Encoder(target_tcp_pose)
                                             -> fresh OFT MLP action head
```

Set `datasets.vla_data.include_state` and `include_tcp` independently; the
corresponding model flags interpolate these values. The VLM remains frozen
and in eval mode. State is a numeric encoder input, not instruction text.
Only the VLM subtree is restored from the released OFT checkpoint, leaving
the new projection/encoders/head trainable. There is no time-budget input or
scheduled VLM unfreezing. Switches define the saved architecture at model
initialization; restore checkpoints with matching switches.

The current training launcher preserves the original FlashAttention2 backend,
the extra action-diagnostic batch every 100 updates, and the original seed
timing in `prepare_training`, after model construction. It does not set a seed
before constructing the model. A configured seed therefore does not by itself
control the new modules' initial weights, matching the original training flow.

The previously released 50,000-step model used SDPA, disabled the extra action
diagnostic, and set the seed before model construction. Its saved configuration
and pinned code revision describe that completed experiment; the current
launcher settings apply to subsequent runs.

The output projection is an additional layer after action-token extraction;
it does not unfreeze or replace the VLM's final Transformer layer. For the
selected Qwen3-VL-4B recipe, initialization and training are:

| Module | Initialization | Trained |
| --- | --- | --- |
| Entire VLM, including its final layer | Released OFT checkpoint | No |
| Output projection, `Linear(2560, 2560)` | Identity weights, zero bias | Yes |
| State encoder, `14 -> 1024 -> 2560` with GELU | Random | Yes |
| Target TCP encoder, `14 -> 1024 -> 2560` with GELU | Random | Yes |
| OFT MLPResNet action head | Random; released head weights are skipped | Yes |

The two encoders are independent MLPs. Their single-token outputs are added
to the projected action embeddings by broadcasting over the 32 action slots.
The original QwenOFT also uses an MLP action head. Its old `DiT-B` configuration
label does not select a DiT, and `action_dit_loss` is a compatibility alias
for the same action-regression objective.

`target_tcp_pose` is the existing loader's future world-frame TCP in meters
with wxyz quaternions, ordered left pose then right pose. It uses the dataset's
TCP definition `p_TCP = p_EE + R(q) @ [0.12, 0, 0]`, without joint-state
normalization. The policy server normalizes raw state using the original
training transform; direct model calls take transformed state. Set
`framework.conditioning.state_input_space: normalized` only for clients that
already perform that transform. The server continues returning
`normalized_actions`; the RoboTwin client applies the original action mapping.

Run checks from this VLAct repository:

```bash
VLACT_PYTHON=/var/tmp/jliu3375/miniconda3/envs/starVLA/bin/python \
  bash examples/Robotwin/train_files/run_oft_state_tcp.sh --config-only

VLACT_PYTHON=/var/tmp/jliu3375/miniconda3/envs/starVLA/bin/python \
  bash examples/Robotwin/train_files/run_oft_state_tcp.sh --validate-only
```

Omit the check flag to train after validation passes. Edit `TRAIN_GPU_IDS`
and `TRAIN_NUM_PROCESSES` at the top of `run_oft_state_tcp.sh` before training
(for example, `"0,1"` and `2` for two selected GPUs). The launcher includes the
selected local GPU/team settings; choose your own GPUs and W&B team before
using it on another account or machine. Validation does not require GPUs. External
`CUDA_VISIBLE_DEVICES`/`NUM_PROCESSES` override those defaults. Multiple
processes use native DDP. The launcher selects the prepared
local Hugging Face cache and defaults to offline loading and disabled W&B.
The original checkpoint remains the directly evaluated baseline.

For live W&B curves, log in once in the training environment, then add
`--wandb` (or `--wandb-mode online`) to the launcher:

The W&B setup and logging paths were verified with W&B SDK 0.31.0.

```bash
python -m wandb login
WANDB_PROJECT=VLAct_Robotwin_TCP \
  bash examples/Robotwin/train_files/run_oft_state_tcp.sh --wandb
```

Set `WANDB_ENTITY` to select a team/workspace; otherwise the logged-in account
default or only writable team is checked before loading the VLM. A login
username is never guessed as an entity. A server-provided default can be
read-only and is then skipped.
The preflight reads entity/project permission metadata without creating a run;
the final W&B initialization still determines whether a new project/run can
be created. The actual `wandb.init` is therefore also called before loading
the model, so authentication/run-creation failures do not load the 4B VLM.
You may instead edit `TRAIN_WANDB_ENTITY` at the top of the launcher.
To create a team at launch, leave that field empty and set
`TRAIN_WANDB_CREATE_TEAM` to your chosen new team name. An existing team in
your account is reused. New team creation requires the account's Default API
organization in W&B user settings and permission to create teams there;
the script does not create an organization. Configuration/data validation
never creates teams. `WANDB_NAME` overrides the displayed run name. W&B receives
the resolved training configuration, action loss, per-module learning rates,
epoch and data/model times every `trainer.logging_frequency` updates (100 by
default). For this wrapped-angle OFT experiment, monitor `action_loss`.
The original `mse_score` auxiliary diagnostic is also recorded every 100 updates,
using an extra batch from the training loader on all ranks. Despite its name,
it is the masked Euclidean norm divided by the valid element count, not mean
squared error, held-out validation loss or a wrapped-angle metric. It does not
update model parameters. The official joint objective blends half wrapped L1 with
half ordinary L1; it is not pure shortest-angle MAE. Benchmark success is
measured separately by closed-loop evaluation. For an offline node,
use `--wandb-mode offline` and later `wandb sync /path/to/offline-run-directory`.
Logging remains disabled when no W&B mode is requested.

With the supplied recipe, model weights are saved every 10,000 updates to
`<run_root_dir>/<run_id>/checkpoints/steps_<step>_pytorch_model.pt`.
Normal completion also writes `final_model/pytorch_model.pt` in that run
directory. Both contain the complete model weights, including the frozen
VLM and the trainable modules; optimizer, scheduler and RNG state are not
saved. Keep the run's configuration and dataset statistics with its weights.

These checks establish code/data integration, not benchmark success. The
enhanced closed-loop benchmark still needs the same-scene expert provider to
put future `target_tcp_pose` values into each policy request; a current TCP
observation must not be substituted for that future target.

中文：修改位于 VLAct，训练目录为 `RoboTwin-Clean-TCP`，已选择回退到 Clean
最后观测帧的版本。原图像、动作和全部训练帧保留；state/TCP 独立开关，
VLM 全程冻结，只训练输出 projection、两个 encoder 和随机初始化 OFT 头。
完整 50 项闭环 benchmark 与训练入口分开，CPU/数据检查不代表已完成 benchmark。

### Current TCP pose state with XYZW quaternions

An independent dataset variant is available at:

```text
/scratch/jliu3375/datasets/RoboTwin-Clean-TCP-PoseState-XYZW
```

Both current `observation.state` / `observation.tcp_pose` and future
`target_tcp_pose` use raw float32 TCP poses: left `[x,y,z,qx,qy,qz,qw]`
followed by right `[x,y,z,qx,qy,qz,qw]`, in world coordinates and meters.
The current state is from the same observation frame. The original joint/gripper
observation is backed up as `observation.joint_state`; joint/gripper actions,
language, frame/episode IDs and video correspondence remain unchanged.
Video directories share the original Clean files through links.

This directory is an exactly reversible component permutation of
`RoboTwin-Clean-TCP-PoseState`, which remains unchanged with its original WXYZ
order. TCP positions, quaternion signs and physical orientations are preserved.
The canonical future sidecar array is `tcp_pose_world_xyzw`. Its original EE
provenance array, `ee_pose_world_wxyz`, remains unchanged and explicitly WXYZ;
`array_quaternion_orders` records the separate orders. State/TCP pose contracts
use schema 2 and declare XYZW, independently of the unchanged goal-sampling schema.

Use `robotwin_tcp_state_wrap_32`, explicit `state_representation: tcp_pose`,
and `tcp_quaternion_order: xyzw`.
This opt-in reader preserves state values and float32 precision, with no
joint normalization, gripper thresholding, angle wrapping or state reordering.
The action transforms remain the original joint-wrap32 transforms. Future
`target_tcp_pose` continues to use Clean-last schema 2 independently of the
current pose state. Its two input switches remain independent.

The separate recipe `train_files/starvla_oft_tcp_pose_state.yaml` computes new
state statistics and has its own run ID. It keeps FlashAttention2, the original
100-step extra-batch diagnostic and the original seed timing. The earlier
released checkpoint expects joint state; the pose-state variant requires
newly trained state/TCP encoders and action head with the matching XYZW inputs.

The independent full audit passed for all 50 tasks, 2,500 episodes, 549,787
Clean observation rows, 552,287 full RAW pose frames and 7,500 shared videos.
It verified exact forward/inverse permutations, unchanged Clean actions and
metadata correspondence, statistics and sidecar SHA provenance. The maximum
EE-to-TCP geometry error was `5.96e-8` meters. This is a dataset audit; it does
not establish model or closed-loop benchmark performance.

```bash
# If needed, first create the WXYZ pose-state source from Clean-TCP.
python examples/Robotwin/train_files/prepare_tcp_state_dataset.py

# Create the separate XYZW folder; existing output folders are never overwritten.
python examples/Robotwin/train_files/prepare_tcp_xyzw_dataset.py

# Full task/episode/frame correspondence, including all sidecars and videos.
python examples/Robotwin/train_files/check_tcp_xyzw_correspondence.py \
  --report /scratch/jliu3375/logs/tcp_xyzw_clean_correspondence.json

# Validate this recipe before starting a new run.
bash examples/Robotwin/train_files/run_oft_state_tcp.sh \
  --config examples/Robotwin/train_files/starvla_oft_tcp_pose_state.yaml --validate-only

# Start the independent pose-state run when ready.
bash examples/Robotwin/train_files/run_oft_state_tcp.sh \
  --config examples/Robotwin/train_files/starvla_oft_tcp_pose_state.yaml --wandb
```

### Four-GPU fast training and checkpoint uploads

`run_oft_state_tcp.sh` now defaults to the XYZW pose-state recipe above, using
GPUs `0,1,6,7` and four native DDP processes. With batch size 32 per device
and accumulation 1, the global batch size is 128. From this repository, run:

```bash
bash examples/Robotwin/train_files/run_oft_state_tcp.sh --wandb
```

`CUDA_VISIBLE_DEVICES` and `NUM_PROCESSES` can override the script defaults.
The launcher defaults `NCCL_NVLS_ENABLE=0` for the local H100 node's NVLS
multicast initialization error (CUDA 401), retaining ordinary NVLink/P2P
communication. Explicit environment overrides remain supported.

The recipe prepares the same Qwen processor inputs inside CPU data-loader
workers and transfers those tensors without splitting flattened image patches
across examples. PyAV decoding uses one thread per codec to avoid excessive
decoder threads during random-frame access. Images, prompts, sampling RNGs,
action targets and model outputs retain their existing contracts.

W&B and terminal logs include `grad_norm` at the configured logging interval.
This is the total gradient norm **before** clipping; values above the clipping
threshold of 1.0 are expected. The existing extra-batch `mse_score` diagnostic
and its historical formula described above are unchanged.

The latest recipe enables `trainer.allow_tf32: true` and
`trainer.optimizer.fused: true` for subsequent runs. TF32 permits reduced
mantissa precision for FP32 matrix multiplication without changing stored
weight dtypes; fused AdamW uses PyTorch's fused optimizer implementation.
Other configurations retain their defaults when these options are absent.
The completed 50k-step fast run did not enable either option; use its saved
`config.full.yaml` when reproducing that run's numerical settings.

`trainer.hf_upload` enables model uploads to the configured Hugging Face
repository (`afhwoinavsnvo/VLAct-OFT-TCP-PoseState` in this local recipe).
Authentication uses the standard HF token environment or login cache. Change
the repository ID and visibility in the recipe to select another destination.
The destination is checked before loading the VLM. Every 10k-step save and
the final model are queued for serial background upload with matching configs,
dataset statistics and TCP contracts. Upload subprocesses can connect to HF
while the training process keeps offline model loading.

Checkpoint saves use atomic replacement. Each upload keeps an immutable local
hardlink and metadata snapshot, so retries are safe even when a later run
reuses the same output filenames. Each job records `job.json`, `status.json`
and `upload.log` under `<output_dir>/hf_uploads/`; failures retain the local
weights and have bounded retries. To retry a job without starting training:

```bash
python starVLA/training/trainer_utils/hf_checkpoint_upload.py \
  --job /path/to/run/hf_uploads/steps_10000_<id>/job.json
```

## Upstream baseline workflow and results

The workflow and published scores below are retained from the upstream
baseline documentation. Evaluation results for the state/TCP-conditioned
model above have not yet been measured.

The evaluation process consists of two main parts:  

1. Setting up the `robotwin` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `robotwin` environments.  

We have verified that this workflow runs successfully on **NVIDIA 4090** GPUs.  

# Results


<details close>
<summary><b>RoboTwin 2.0 Benchmark Results over 48 Tasks </b></summary>


| Task Name | RDT Easy | RDT Hard | Pi0 Easy | Pi0 Hard | ACT Easy | ACT Hard | DP Easy | DP Hard | DP3 Easy | DP3 Hard | Qwen3OFT Easy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Adjust Bottle | 81 | 75 | 90 | 56 | 97 | 23 | 97 | 0 | 99 | 3 | 96 |
| Beat Block Hammer | 77 | 37 | 43 | 21 | 56 | 3 | 42 | 0 | 72 | 8 | 58 |
| Blocks Ranking RGB | 3 | 0 | 19 | 5 | 1 | 0 | 0 | 0 | 3 | 0 | 45 |
| Blocks Ranking Size | 0 | 0 | 7 | 1 | 0 | 0 | 1 | 0 | 2 | 0 | 27 |
| Click Alarmclock | 61 | 12 | 63 | 11 | 32 | 4 | 61 | 5 | 77 | 14 | 91 |
| Click Bell | 80 | 9 | 44 | 3 | 58 | 3 | 54 | 0 | 90 | 0 | 94 |
| Dump Bin Bigbin | 64 | 32 | 83 | 24 | 68 | 1 | 49 | 0 | 85 | 53 | 68 |
| Grab Roller | 74 | 43 | 96 | 80 | 94 | 25 | 98 | 0 | 98 | 2 | 93 |
| Handover Block | 45 | 14 | 45 | 8 | 42 | 0 | 10 | 0 | 70 | 0 | 0 |
| Handover Mic | 90 | 31 | 98 | 13 | 85 | 0 | 53 | 0 | 100 | 3 | 39 |
| Hanging Mug | 23 | 16 | 11 | 3 | 7 | 0 | 8 | 0 | 17 | 1 | 15 |
| Lift Pot | 72 | 9 | 84 | 36 | 88 | 0 | 39 | 0 | 97 | 0 | 0 |
| Move Can Pot | 25 | 12 | 58 | 21 | 22 | 4 | 39 | 0 | 70 | 6 | 50 |
| Move Pillbottle Pad | 8 | 0 | 21 | 1 | 0 | 0 | 1 | 0 | 41 | 0 | 54 |
| Move Playingcard Away | 43 | 11 | 53 | 22 | 36 | 0 | 47 | 0 | 68 | 3 | 69 |
| Move Stapler Pad | 2 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 12 | 0 | 12 |
| Open Laptop | 59 | 32 | 85 | 46 | 56 | 0 | 49 | 0 | 82 | 7 | 31 |
| Open Microwave | 37 | 20 | 80 | 50 | 86 | 0 | 5 | 0 | 61 | 22 | -- |
| Pick Diverse Bottles | 2 | 0 | 27 | 6 | 7 | 0 | 6 | 0 | 52 | 1 | 30 |
| Pick Dual Bottles | 42 | 13 | 57 | 12 | 31 | 0 | 24 | 0 | 60 | 1 | 43 |
| Place A2B Left | 3 | 1 | 31 | 1 | 1 | 0 | 2 | 0 | 46 | 2 | 20 |
| Place A2B Right | 1 | 1 | 27 | 6 | 0 | 0 | 13 | 0 | 49 | 0 | 22 |
| Place Bread Basket | 10 | 2 | 17 | 4 | 6 | 0 | 14 | 0 | 26 | 1 | 52 |
| Place Bread Skillet | 5 | 1 | 23 | 1 | 7 | 0 | 11 | 0 | 19 | 0 | 56 |
| Place Burger Fries | 50 | 27 | 80 | 4 | 49 | 0 | 72 | 0 | 72 | 18 | 96 |
| Place Can Basket | 19 | 6 | 41 | 5 | 1 | 0 | 18 | 0 | 67 | 2 | 63 |
| Place Cans Plasticbox | 6 | 5 | 34 | 2 | 16 | 0 | 40 | 0 | 48 | 3 | 81 |
| Place Container Plate | 78 | 17 | 88 | 45 | 72 | 1 | 41 | 0 | 86 | 1 | 99 |
| Place Dual Shoes | 4 | 4 | 15 | 0 | 9 | 0 | 8 | 0 | 13 | 0 | 28 |
| Place Empty Cup | 56 | 7 | 37 | 11 | 61 | 0 | 37 | 0 | 65 | 1 | 72 |
| Place Fan | 12 | 2 | 20 | 10 | 1 | 0 | 3 | 0 | 36 | 1 | 28 |
| Place Mouse Pad | 1 | 0 | 7 | 1 | 0 | 0 | 0 | 0 | 4 | 1 | 9 |
| Place Object Basket | 33 | 17 | 16 | 2 | 15 | 0 | 15 | 0 | 65 | 0 | 40 |
| Place Object Scale | 1 | 0 | 10 | 0 | 0 | 0 | 1 | 0 | 15 | 0 | 19 |
| Place Object Stand | 15 | 5 | 36 | 11 | 1 | 0 | 22 | 0 | 60 | 0 | 48 |
| Place Phone Stand | 15 | 6 | 35 | 7 | 2 | 0 | 13 | 0 | 44 | 2 | 24 |
| Place Shoe | 35 | 7 | 28 | 6 | 5 | 0 | 23 | 0 | 58 | 2 | 63 |
| Press Stapler | 41 | 24 | 62 | 29 | 31 | 6 | 6 | 0 | 69 | 3 | 60 |
| Put Bottles Dustbin | 21 | 4 | 54 | 13 | 27 | 1 | 22 | 0 | 60 | 21 | -- |
| Put Object Cabinet | 33 | 18 | 68 | 18 | 15 | 0 | 42 | 0 | 72 | 1 | 35 |
| Rotate QRcode | 50 | 5 | 68 | 15 | 1 | 0 | 13 | 0 | 74 | 1 | 50 |
| Scan Object | 4 | 1 | 18 | 1 | 2 | 0 | 9 | 0 | 31 | 1 | 13 |
| Shake Bottle Horizontally | 84 | 51 | 99 | 51 | 63 | 4 | 59 | 18 | 100 | 25 | 98 |
| Shake Bottle | 74 | 45 | 97 | 60 | 74 | 10 | 65 | 8 | 98 | 19 | 98 |
| Stack Blocks Three | 2 | 0 | 17 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 41 |
| Stack Blocks Two | 21 | 2 | 42 | 1 | 25 | 0 | 7 | 0 | 24 | 0 | 83 |
| Stack Bowls Three | 51 | 17 | 66 | 24 | 48 | 0 | 63 | 0 | 57 | 5 | 62 |
| Stack Bowls Two | 76 | 30 | 91 | 41 | 82 | 0 | 61 | 0 | 83 | 6 | 90 |
| Stamp Seal | 1 | 0 | 3 | 4 | 2 | 0 | 2 | 0 | 18 | 0 | 27 |
| Turn Switch | 35 | 15 | 27 | 23 | 5 | 2 | 36 | 1 | 46 | 8 | 26 |
| **Average** | **34.50** | **13.72** | **46.42** | **16.34** | **29.74** | **1.74** | **28.04** | **0.64** | **55.24** | **4.96** | **50.38** |

*Note: All 50 tasks are trained within a single model, using 50 demonstrations per task (50×50 total demonstrations). Checkpoints can be downloaded at [Qwen3-VL-OFT-Robotwin2](https://huggingface.co/StarVLA/Qwen3-VL-OFT-Robotwin2)*.

</details>


<details open>
<summary><b>RoboTwin 2.0 data-scaling settings </b></summary>

### Training Dataset

The model is trained using the official **RobotWin 2.0 dataset**.

* Clean Demonstrations: 50 tasks × 50 trajectories per task
* Randomized Demonstrations: 50 tasks × 500 trajectories per task

### StarVLA-OFT

| Task                      | Easy       | Hard       |
| ------------------------- | ---------- | ---------- |
| stack_blocks_two          | 1.0000     | 1.0000     |
| place_cans_plasticbox     | 1.0000     | 0.9900     |
| grab_roller               | 1.0000     | 1.0000     |
| place_empty_cup           | 1.0000     | 1.0000     |
| shake_bottle_horizontally | 1.0000     | 1.0000     |
| lift_pot                  | 1.0000     | 1.0000     |
| place_burger_fries        | 1.0000     | 1.0000     |
| move_playingcard_away     | 1.0000     | 0.9800     |
| adjust_bottle             | 1.0000     | 0.9900     |
| shake_bottle              | 1.0000     | 1.0000     |
| blocks_ranking_rgb        | 0.9900     | 0.9800     |
| stack_bowls_two           | 0.9900     | 1.0000     |
| place_container_plate     | 0.9900     | 0.9900     |
| press_stapler             | 0.9900     | 0.9600     |
| place_object_stand        | 0.9900     | 0.9800     |
| open_laptop               | 0.9800     | 1.0000     |
| handover_mic              | 0.9800     | 0.9600     |
| move_pillbottle_pad       | 0.9800     | 1.0000     |
| handover_block            | 0.9700     | 0.9300     |
| place_shoe                | 0.9600     | 1.0000     |
| stack_bowls_three         | 0.9500     | 0.9100     |
| place_fan                 | 0.9400     | 0.9500     |
| scan_object               | 0.9400     | 0.9100     |
| stack_blocks_three        | 0.9400     | 0.8600     |
| place_object_basket       | 0.9300     | 0.9400     |
| beat_block_hammer         | 0.9300     | 0.9200     |
| place_object_scale        | 0.9300     | 0.9300     |
| place_dual_shoes          | 0.9100     | 0.8900     |
| pick_dual_bottles         | 0.9100     | 0.9300     |
| place_bread_basket        | 0.9100     | 0.7800     |
| dump_bin_bigbin           | 0.9100     | 0.9400     |
| move_can_pot              | 0.9100     | 0.9000     |
| put_bottles_dustbin       | 0.9000     | 0.8500     |
| place_a2b_left            | 0.9000     | 0.9500     |
| place_bread_skillet       | 0.8900     | 0.8000     |
| put_object_cabinet        | 0.8900     | 0.9100     |
| place_a2b_right           | 0.8800     | 0.9500     |
| rotate_qrcode             | 0.8800     | 0.9000     |
| pick_diverse_bottles      | 0.8700     | 0.8600     |
| place_mouse_pad           | 0.8700     | 0.9400     |
| stamp_seal                | 0.8600     | 0.9000     |
| place_phone_stand         | 0.8600     | 0.9500     |
| blocks_ranking_size       | 0.7900     | 0.8000     |
| place_can_basket          | 0.7500     | 0.7500     |
| move_stapler_pad          | 0.7400     | 0.9000     |
| turn_switch               | 0.6500     | 0.6200     |
| click_alarmclock          | 0.5800     | 0.5100     |
| hanging_mug               | 0.3400     | 0.2900     |
| open_microwave            | 0.2800     | 0.3900     |
| click_bell                | 0.2300     | 0.2700     |
| **Average**               | **0.8818** | **0.8832** |

*Note: All 50 tasks are trained within a single model, using 50 + 500 demonstrations per task (50×550 total demonstrations). Checkpoints can be downloaded at [Qwen3-VL-OFT-Robotwin2-All](https://huggingface.co/StarVLA/Qwen3-VL-OFT-RoboTwin2-All)*.


</details>

---



# Evaluation

## 📦 1. Environment Setup

To set up the environment, please first follow the [official RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) to install the base `robotwin` environment.  

than pip install additional requirements

```bash
pip install -r examples/Robotwin/eval_files/requirements.txt
```

and edit `ROBOTWIN_PATH` in `examples/Robotwin/eval_files/eval.sh`.

## 🚀 2. Evaluation Workflow

### Step 1. Start the server (starVLA environment)

In the first terminal, activate the `starVLA` conda environment and run:  

```bash
bash examples/Robotwin/eval_files/run_policy_server.sh
```

Edit your checkpoint path in `examples/Robotwin/eval_files/deploy_policy.yml` and `examples/Robotwin/eval_files/run_policy_server.sh`.

---

### Step 2. Start the simulation (robotwin environment)

In the second terminal, activate the `robotwin` conda environment and run:  

```bash
conda activate robotwin
cd examples/Robotwin/eval_files
bash eval.sh task_name demo_clean my_test_v1 0 0
```

all tasks in RoboTwin 2.0 include:

```txt
adjust_bottle
beat_block_hammer
blocks_ranking_rgb
blocks_ranking_size
click_alarmclock
click_bell
dump_bin_bigbin
grab_roller
handover_block
handover_mic
hanging_mug
lift_pot
move_can_pot
move_pillbottle_pad
move_playingcard_away
move_stapler_pad
open_laptop
open_microwave
pick_diverse_bottles
pick_dual_bottles
place_a2b_left
place_a2b_right
place_bread_basket
place_bread_skillet
place_burger_fries
place_can_basket
place_cans_plasticbox
place_container_plate
place_dual_shoes
place_empty_cup
place_fan
place_mouse_pad
place_object_basket
place_object_scale
place_object_stand
place_phone_stand
place_shoe
press_stapler
put_bottles_dustbin
put_object_cabinet
rotate_qrcode
scan_object
shake_bottle_horizontally
shake_bottle
stack_blocks_three
stack_blocks_two
stack_bowls_three
stack_bowls_two
stamp_seal
turn_switch
```

and all modes include `demo_clean` and `demo_randomized`.
