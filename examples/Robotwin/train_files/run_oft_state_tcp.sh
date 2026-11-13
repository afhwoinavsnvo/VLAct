#!/usr/bin/env bash
# Four-GPU XYZW TCP-pose-state training with the released Clean OFT VLM frozen.
set -euo pipefail

TRAIN_GPU_IDS="0,1,6,7"
TRAIN_NUM_PROCESSES=4
TRAIN_WANDB_ENTITY="vlact-ljy666"
TRAIN_WANDB_CREATE_TEAM=""

# This H100 node fails NVLS multicast binding with CUDA error 401.
# Use the regular NVLink/P2P collectives; allow an explicit override.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
python_bin="${VLACT_PYTHON:-${STARVLA_PYTHON:-python}}"
config_path="$script_dir/starvla_oft_tcp_pose_state.yaml"
validate_only=false
config_only=false
wandb_mode="${WANDB_MODE:-disabled}"
while (( $# )); do
  case "$1" in
    --config)
      if (( $# < 2 )); then echo '--config requires a YAML path' >&2; exit 2; fi
      config_path="$2"; shift 2 ;;
    --validate-only) validate_only=true; shift ;;
    --config-only) config_only=true; shift ;;
    --wandb) wandb_mode=online; shift ;;
    --wandb-mode)
      if (( $# < 2 )); then echo '--wandb-mode requires online, offline or disabled' >&2; exit 2; fi
      wandb_mode="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export STARVLA_ROOT="$repo_root"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/scratch/jliu3375/huggingface/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
if [[ -n "${WANDB_ENTITY:-$TRAIN_WANDB_ENTITY}" ]]; then
  export WANDB_ENTITY="${WANDB_ENTITY:-$TRAIN_WANDB_ENTITY}"
fi
if [[ -n "${WANDB_CREATE_TEAM:-$TRAIN_WANDB_CREATE_TEAM}" ]]; then
  export WANDB_CREATE_TEAM="${WANDB_CREATE_TEAM:-$TRAIN_WANDB_CREATE_TEAM}"
fi
case "$wandb_mode" in
  online|offline|disabled) export WANDB_MODE="$wandb_mode" ;;
  *) echo '--wandb-mode must be online, offline or disabled' >&2; exit 2 ;;
esac
check_args=(--config_yaml "$config_path")
if [[ "$config_only" == true ]]; then check_args+=(--config-only); fi
"$python_bin" "$script_dir/check_oft_setup.py" "${check_args[@]}"
if [[ "$config_only" == true || "$validate_only" == true ]]; then exit 0; fi

# Frozen-backbone training can run with native Accelerate/DDP, without a
# missing or machine-specific DeepSpeed configuration.
export STARVLA_DISABLE_DEEPSPEED=1
gpu_ids="${CUDA_VISIBLE_DEVICES:-$TRAIN_GPU_IDS}"
num_processes="${NUM_PROCESSES:-$TRAIN_NUM_PROCESSES}"
if [[ -z "$gpu_ids" ]]; then
  echo '请修改脚本顶部 TRAIN_GPU_IDS，例如 "0" 或 "0,1"。' >&2; exit 2
fi
if [[ ! "$num_processes" =~ ^[1-9][0-9]*$ ]]; then
  echo 'TRAIN_NUM_PROCESSES / NUM_PROCESSES must be a positive integer' >&2; exit 2
fi
IFS=',' read -r -a gpu_devices <<< "$gpu_ids"
if (( num_processes > ${#gpu_devices[@]} )); then
  echo '进程数不能超过 TRAIN_GPU_IDS / CUDA_VISIBLE_DEVICES 中的 GPU 数量。' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="$gpu_ids"
export NUM_PROCESSES="$num_processes"
if [[ "$wandb_mode" == online ]]; then
  # Resolve/authenticate before allocating the 4B VLM on a GPU. Run once in
  # this parent process so DDP workers cannot create duplicate teams.
  WANDB_ENTITY="$("$python_bin" -m starVLA.training.trainer_utils.wandb_setup --config "$config_path")"
  export WANDB_ENTITY
  unset WANDB_CREATE_TEAM
  echo "W&B entity: $WANDB_ENTITY"
fi
echo "Training GPUs: $CUDA_VISIBLE_DEVICES; processes: $NUM_PROCESSES"
if (( num_processes == 1 )); then
  exec "$python_bin" starVLA/training/train_starvla.py --config_yaml "$config_path"
fi
launch_args=(--multi_gpu --num_processes "$num_processes" --num_machines 1 --mixed_precision bf16)
if [[ -n "${MASTER_PORT:-}" ]]; then launch_args+=(--main_process_port "$MASTER_PORT"); fi
exec "$python_bin" -m accelerate.commands.launch "${launch_args[@]}" \
  starVLA/training/train_starvla.py --config_yaml "$config_path"
