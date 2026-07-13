#!/usr/bin/env bash
#SBATCH --job-name=webarea-debug-tmax-pi-1n
#SBATCH --account=nemotron_omni_vision
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.err
#SBATCH --export=ALL

# One-node PI + Qwen3.5-4B TMax smoke/debug run.
# Uses 4 GPUs for Megatron training and 4 GPUs for SGLang rollout.
set -euo pipefail

project_root="${project_root:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server}"
script_dir="${project_root}/examples/tmax_slime"
train_sqsh="${train_sqsh:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/docker/polr_swegym_qwen35_torch211_te2161_fa4b19_numpy126_scipy117_tebindcu130_20260707.sqsh}"
container_mounts="${container_mounts:-/lustre/fs1:/lustre/fs1,/lustre/fsw:/lustre/fsw}"
tmax_dataset_dir="${tmax_dataset_dir:-/lustre/fsw/portfolios/nvr/users/songyangh/bjin_works/agent_world_model/tmax15k/dataset}"
tmax_image_dir="${tmax_image_dir:-/lustre/fsw/portfolios/nvr/users/songyangh/bjin_works/agent_world_model/tmax15k/sif}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[ "${SLURM_JOB_NUM_NODES:-0}" = "1" ] || die "this script requires exactly one allocated node"
case "${SLURM_JOB_NAME:-}" in
    webarea-debug*) ;;
    *) die "job name must start with webarea-debug, got ${SLURM_JOB_NAME:-unknown}" ;;
esac
[ -x "${script_dir}/run_tmax_pi_apptainer_train.sh" ] || die "missing inner launcher: ${script_dir}/run_tmax_pi_apptainer_train.sh"
[ -f "${train_sqsh}" ] || die "missing training image: ${train_sqsh}"
[ -d "${tmax_dataset_dir}" ] || die "missing TMax dataset dir: ${tmax_dataset_dir}"
[ -d "${tmax_image_dir}" ] || die "missing TMax SIF dir: ${tmax_image_dir}"
mkdir -p "${project_root}/logs/slurm"

job_account="${SLURM_JOB_ACCOUNT:-}"
if [ -z "${job_account}" ]; then
    job_account="$(scontrol show job -o "${SLURM_JOB_ID}" | sed -n 's/.* Account=\([^ ]*\).*/\1/p')"
fi
case "${job_account}" in
    nemotron_omni_vision) ;;
    *) die "account ${job_account:-unknown} is not allowed; use nemotron_omni_vision" ;;
esac

if [ -z "${WANDB_API_KEY:-}" ]; then
    WANDB_API_KEY="$(python3 - <<'PY'
import netrc

try:
    auth = netrc.netrc().authenticators("api.wandb.ai")
except Exception:
    auth = None
print(auth[2] if auth else "")
PY
)"
    export WANDB_API_KEY
fi

run_id="${run_id:-webarea-debug_tmax_pi_1n_${SLURM_JOB_ID}}"
run_label="${run_label:-1n-tmax-smoke}"

# Use job-specific ports so repeated debug submissions do not collide with
# stale host-network services from earlier allocations.
port_slot=$((SLURM_JOB_ID % 1000))
ray_port="${ray_port:-$((30000 + port_slot))}"
ray_dashboard_port="${ray_dashboard_port:-$((28000 + port_slot))}"
rollout_port="${rollout_port:-$((18000 + port_slot))}"
gateway_port="${gateway_port:-$((20000 + port_slot))}"
sglang_router_port="${sglang_router_port:-$((26000 + port_slot))}"
slime_sglang_base_port="${slime_sglang_base_port:-$((34000 + port_slot))}"

export project_root run_id run_label train_sqsh
export tmax_dataset_dir tmax_image_dir
export num_nodes=1
export gpus_per_node=8
export total_gpus=8
export train_num_gpus=4
export actor_num_nodes=1
export actor_num_gpus_per_node=4
export rollout_num_gpus=4
export rollout_num_gpus_per_engine=1
export tensor_model_parallel_size=4
export ray_head_num_gpus=8
export ray_expected_num_gpus=8
export ray_num_cpus=128

# Small TMax smoke: scan a prefix for ready SIFs, train on two prompt groups
# with two trajectories each, and stop after one rollout.
export tmax_scan_tasks="${tmax_scan_tasks:-512}"
export tmax_only_ready="${tmax_only_ready:-1}"
export smoke_rows="${smoke_rows:-2}"
export rollout_batch_size="${rollout_batch_size:-2}"
export n_samples_per_prompt="${n_samples_per_prompt:-2}"
export num_rollout="${num_rollout:-1}"
export target_num_rollout="${target_num_rollout:-1}"
export save_interval="${save_interval:-1}"

export qwen_gdn_backend=fla
export attention_backend=flash
export pi_fail_on_context_limit=0
export pi_context_window="${pi_context_window:-24000}"
export pi_max_tokens="${pi_max_tokens:-512}"
export fetch_trajectory_retry_times=3
export max_tokens_per_gpu="${max_tokens_per_gpu:-50000}"
export sglang_context_length="${sglang_context_length:-50000}"
export rollout_max_prompt_len="${rollout_max_prompt_len:-8192}"
export rollout_max_response_len="${rollout_max_response_len:-4096}"
export polar_max_async_level="${polar_max_async_level:-2}"
export polar_min_complete_accept_fraction="${polar_min_complete_accept_fraction:-0.0}"
export polar_task_timeout_from_metadata=1
export polar_request_timeout="${polar_request_timeout:-1800}"

export ray_port ray_dashboard_port rollout_port gateway_port
export sglang_router_port slime_sglang_base_port
export ray_tmpdir="${ray_tmpdir:-/tmp/wdtr-${SLURM_JOB_ID}}"
export job_cache_root="${job_cache_root:-/tmp/wdtp-${SLURM_JOB_ID}}"
export POLAR_APPTAINER_DIRECT_EXEC="${POLAR_APPTAINER_DIRECT_EXEC:-1}"
export POLAR_APPTAINER_EXEC_MODE="${POLAR_APPTAINER_EXEC_MODE:-direct}"

export use_wandb="${use_wandb:-1}"
export wandb_mode="${wandb_mode:-online}"
export wandb_entity="${wandb_entity:-hwinf_dcm}"
export wandb_project="${wandb_project:-harness-tma}"
export wandb_group="${wandb_group:-${run_id}}"
export wandb_run_id="${wandb_run_id:-${run_id}}"

printf 'job_id=%s\n' "${SLURM_JOB_ID}"
printf 'job_name=%s\n' "${SLURM_JOB_NAME}"
printf 'node_list=%s\n' "${SLURM_NODELIST}"
printf 'run_id=%s\n' "${run_id}"
date -u +%FT%TZ

srun --kill-on-bad-exit=1 \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=128 \
    --container-image="${train_sqsh}" \
    --container-mounts="${container_mounts}" \
    --container-workdir="${project_root}" \
    --container-writable --no-container-mount-home \
    bash -lc '
set -euo pipefail
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
bash examples/tmax_slime/run_tmax_pi_apptainer_train.sh
'
