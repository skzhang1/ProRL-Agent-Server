#!/usr/bin/env bash
#SBATCH --job-name=webarea-debug-tmax-pi-2n-validgrad
#SBATCH --account=nemotron_omni_vision
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.err
#SBATCH --export=ALL

# Two-node TMax PI valid-gradient smoke.
# Rank 0 owns Ray head, Polar rollout/gateway, and Slime training. Rank 1 joins
# Ray as a worker so SGLang rollout engines can use the second node's GPUs.
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

[ "${SLURM_JOB_NUM_NODES:-0}" = "2" ] || die "this script requires exactly two allocated nodes"
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

run_id="${run_id:-webarea-debug_tmax_pi_2n_validgrad_${SLURM_JOB_ID}}"
run_label="${run_label:-2n-tmax-validgrad}"
run_dir="${run_dir:-${project_root}/tmp/${run_id}}"
run_log_dir="${run_log_dir:-${run_dir}/logs}"
rollout_save_dir="${rollout_save_dir:-${run_dir}/rollout_results}"
save_dir="${save_dir:-${project_root}/tmp/ckpt/${run_id}}"
mkdir -p "${run_dir}" "${run_log_dir}" "${rollout_save_dir}" "${save_dir}"

stop_file="${run_dir}/ray_workers.stop"
worker_script="${run_dir}/ray_cluster_rank.sh"
rm -f "${stop_file}"

mapfile -t slurm_nodes < <(scontrol show hostnames "${SLURM_NODELIST}")
[ "${#slurm_nodes[@]}" -eq 2 ] || die "expected two Slurm hosts, got ${#slurm_nodes[@]}"
head_node="${slurm_nodes[0]}"
worker_node="${slurm_nodes[1]}"
ray_head_ip="$(srun --overlap -N1 -n1 -w "${head_node}" hostname -I | awk '{print $1}')"
[ -n "${ray_head_ip}" ] || die "failed to resolve Ray head IP for ${head_node}"
sglang_router_host="${ray_head_ip}"

# Use job-specific ports so repeated debug submissions do not collide with
# stale host-network services from earlier allocations.
port_slot=$((SLURM_JOB_ID % 1000))
ray_port="${ray_port:-$((30000 + port_slot))}"
ray_dashboard_port="${ray_dashboard_port:-$((28000 + port_slot))}"
rollout_port="${rollout_port:-$((18000 + port_slot))}"
gateway_port="${gateway_port:-$((20000 + port_slot))}"
sglang_router_port="${sglang_router_port:-$((26000 + port_slot))}"
slime_sglang_base_port="${slime_sglang_base_port:-$((34000 + port_slot))}"

export project_root script_dir train_sqsh container_mounts
export tmax_dataset_dir tmax_image_dir
export run_id run_label run_dir run_log_dir rollout_save_dir save_dir stop_file
export num_nodes=2
export gpus_per_node=8
export total_gpus=16
export train_num_gpus=8
export actor_num_nodes=1
export actor_num_gpus_per_node=8
export rollout_num_gpus=8
export rollout_num_gpus_per_engine=1
export tensor_model_parallel_size=4
export ray_head_num_gpus=8
export ray_expected_num_gpus=16
export ray_cluster_timeout_seconds="${ray_cluster_timeout_seconds:-600}"
export ray_num_cpus=128

# Keep the same small valid-gradient surface as the 1-node successful run.
export tmax_scan_tasks="${tmax_scan_tasks:-512}"
export tmax_only_ready="${tmax_only_ready:-1}"
export smoke_rows="${smoke_rows:-4}"
export rollout_batch_size="${rollout_batch_size:-4}"
export n_samples_per_prompt="${n_samples_per_prompt:-4}"
export num_rollout="${num_rollout:-2}"
export target_num_rollout="${target_num_rollout:-2}"
export save_interval="${save_interval:-1}"

export qwen_gdn_backend=fla
export attention_backend=flash
export pi_fail_on_context_limit=0
export pi_context_window="${pi_context_window:-32000}"
export pi_max_tokens="${pi_max_tokens:-512}"
export fetch_trajectory_retry_times=3
export max_tokens_per_gpu="${max_tokens_per_gpu:-50000}"
export sglang_context_length="${sglang_context_length:-50000}"
export rollout_max_prompt_len="${rollout_max_prompt_len:-32000}"
export rollout_max_response_len="${rollout_max_response_len:-4096}"
export polar_max_async_level="${polar_max_async_level:-4}"
export polar_min_complete_accept_fraction="${polar_min_complete_accept_fraction:-0.0}"
export polar_task_timeout_from_metadata=1
export polar_request_timeout="${polar_request_timeout:-1800}"

export ray_port ray_dashboard_port rollout_port gateway_port
export sglang_router_port slime_sglang_base_port
export ray_head_ip sglang_router_host
export POLAR_APPTAINER_DIRECT_EXEC="${POLAR_APPTAINER_DIRECT_EXEC:-1}"
export POLAR_APPTAINER_EXEC_MODE="${POLAR_APPTAINER_EXEC_MODE:-direct}"

export use_wandb="${use_wandb:-1}"
export wandb_mode="${wandb_mode:-online}"
export wandb_entity="${wandb_entity:-hwinf_dcm}"
export wandb_project="${wandb_project:-harness-tma}"
export wandb_group="${wandb_group:-${run_id}}"
export wandb_run_id="${wandb_run_id:-${run_id}}"

cat >"${worker_script}" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

rank="${SLURM_PROCID}"
node="$(hostname)"
node_ip="$(hostname -I | awk '{print $1}')"
cache_root="/tmp/wdt2-${SLURM_JOB_ID}-${rank}"
export PATH="/opt/polr_venv/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HOME="${cache_root}/home"
export APPTAINER_CACHEDIR="${cache_root}/apptainer-cache"
export APPTAINER_TMPDIR="${cache_root}/apptainer-tmp"
export APPTAINER_WORKDIR="${cache_root}/apptainer-work"
export TRITON_CACHE_DIR="${cache_root}/triton"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export XDG_CACHE_HOME="${cache_root}/xdg-cache"
export XDG_CONFIG_HOME="${cache_root}/xdg-config"
export XDG_RUNTIME_DIR="${cache_root}/xdg-runtime"
export CUDA_CACHE_PATH="${cache_root}/cuda-cache"
export NUMBA_CACHE_DIR="${cache_root}/numba"
export ray_tmpdir="${cache_root}/ray"
export job_cache_root="${cache_root}/job"
mkdir -p "${HOME}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_WORKDIR}" \
    "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" "${XDG_RUNTIME_DIR}" "${CUDA_CACHE_PATH}" "${NUMBA_CACHE_DIR}" \
    "${ray_tmpdir}" "${job_cache_root}"
chmod 700 "${XDG_RUNTIME_DIR}"

if [ ! -x /opt/polr_venv/bin/python ]; then
    echo "ERROR: ${node} is missing /opt/polr_venv/bin/python" >&2
    exit 86
fi
/opt/polr_venv/bin/python - <<'PY'
import torch

if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 visible GPUs, got {torch.cuda.device_count()}")
print(f"CUDA preflight: torch={torch.__version__}, cuda={torch.version.cuda}, GPUs={torch.cuda.device_count()}")
PY

patch_container_runtime_only=1 bash "${script_dir}/run_tmax_pi_apptainer_train.sh" \
    >"${run_log_dir}/container-patch-rank-${rank}.log" 2>&1

ray stop --force >/dev/null 2>&1 || true
monitor_pid=""
ray_pid=""
cleanup() {
    [ -z "${monitor_pid}" ] || kill "${monitor_pid}" 2>/dev/null || true
    if [ -n "${ray_pid}" ]; then
        kill "${ray_pid}" 2>/dev/null || true
        wait "${ray_pid}" 2>/dev/null || true
    fi
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

(
    while true; do
        printf '%s,%s,%s,' "$(date -u +%FT%TZ)" "${node}" "${rank}"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits | paste -sd ';' -
        sleep 30
    done
) >>"${run_log_dir}/gpu-rank-${rank}.csv" 2>&1 &
monitor_pid="$!"

if [ "${rank}" = "0" ]; then
    ray start --head --node-ip-address="${ray_head_ip}" --port="${ray_port}" \
        --dashboard-host=0.0.0.0 --dashboard-port="${ray_dashboard_port}" \
        --num-cpus="${ray_num_cpus}" --num-gpus="${gpus_per_node}" \
        --temp-dir="${ray_tmpdir}" --disable-usage-stats \
        >"${run_log_dir}/ray-head.log" 2>&1

    finish_workers() {
        touch "${stop_file}"
    }
    trap 'finish_workers; cleanup' EXIT

    ray_use_existing_cluster=1 ray_stop_on_exit=0 \
        bash "${script_dir}/run_tmax_pi_apptainer_train.sh"
else
    python - "${ray_head_ip}" "${ray_port}" <<'PY'
import socket
import sys
import time

host, port = sys.argv[1], int(sys.argv[2])
for _ in range(300):
    try:
        with socket.create_connection((host, port), timeout=2):
            raise SystemExit(0)
    except OSError:
        time.sleep(2)
raise SystemExit(f"Ray head did not open at {host}:{port}")
PY
    ray start --address="${ray_head_ip}:${ray_port}" --node-ip-address="${node_ip}" \
        --num-cpus="${ray_num_cpus}" --num-gpus="${gpus_per_node}" \
        --temp-dir="${ray_tmpdir}" --disable-usage-stats --block \
        >"${run_log_dir}/ray-worker-${rank}.log" 2>&1 &
    ray_pid="$!"
    while [ ! -f "${stop_file}" ]; do
        kill -0 "${ray_pid}" 2>/dev/null || { wait "${ray_pid}"; exit $?; }
        sleep 5
    done
fi
WORKER
chmod +x "${worker_script}"

cat <<SUMMARY
============================================================
webarea TMax PI 2-node validgrad
  job_id:    ${SLURM_JOB_ID}
  job_name:  ${SLURM_JOB_NAME}
  nodes:     ${slurm_nodes[*]}
  ray head:  ${head_node} (${ray_head_ip})
  worker:    ${worker_node}
  run:       ${run_id}
  account:   ${job_account}
  GPUs:      8 train + 8 rollout
  TP/DP:     ${tensor_model_parallel_size}/$((train_num_gpus / tensor_model_parallel_size))
  batch:     ${rollout_batch_size} prompts x ${n_samples_per_prompt} samples = $((rollout_batch_size * n_samples_per_prompt)) trajectories
  boundary:  ${num_rollout}/${target_num_rollout}
  W&B:       ${wandb_entity}/${wandb_project}/${wandb_run_id}
============================================================
SUMMARY
date -u +%FT%TZ

srun --overlap --kill-on-bad-exit=1 \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=128 \
    --container-image="${train_sqsh}" \
    --container-mounts="${container_mounts}" \
    --container-workdir="${project_root}" \
    --container-writable --no-container-mount-home \
    bash "${worker_script}"
