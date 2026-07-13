#!/usr/bin/env bash
#SBATCH --job-name=webarea-debug-tmax-pi-8n-dapo-profile
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=batch_block1
#SBATCH --reservation=sla_res_fw190_d580
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.err
#SBATCH --export=ALL

# 8-node TMax PI DAPO profiling launcher.
#
# This is intentionally a short-run, env-overridable profiler: each submission
# runs one optimizer step, records rollout/train timings, and exits.
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

positive_int() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        0) return 1 ;;
        *) return 0 ;;
    esac
}

nonnegative_int() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

nodes="${nodes:-8}"
[ "${SLURM_JOB_NUM_NODES:-0}" = "${nodes}" ] || die "this script requires exactly ${nodes} allocated nodes"
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
    nvr_lpr_agentic) ;;
    *) die "account ${job_account:-unknown} is not allowed; use nvr_lpr_agentic" ;;
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

profile_id="${profile_id:-b8_s8_t2_r48_a4_gw128}"
run_id="${run_id:-webarea-debug_tmax_pi_8n_dapo_${profile_id}_${SLURM_JOB_ID}}"
run_label="${run_label:-8n-dapo-${profile_id}}"
run_dir="${run_dir:-${project_root}/tmp/${run_id}}"
run_log_dir="${run_log_dir:-${run_dir}/logs}"
rollout_save_dir="${rollout_save_dir:-${run_dir}/rollout_results}"
save_dir="${save_dir:-${project_root}/tmp/ckpt/${run_id}}"
mkdir -p "${run_dir}" "${run_log_dir}" "${rollout_save_dir}" "${save_dir}"

stop_file="${run_dir}/ray_workers.stop"
worker_script="${run_dir}/ray_cluster_rank.sh"
rm -f "${stop_file}"

mapfile -t slurm_nodes < <(scontrol show hostnames "${SLURM_NODELIST}")
[ "${#slurm_nodes[@]}" -eq "${nodes}" ] || die "expected ${nodes} Slurm hosts, got ${#slurm_nodes[@]}"

slurm_node_records=()
ray_head_ip=""
for idx in "${!slurm_nodes[@]}"; do
    node="${slurm_nodes[$idx]}"
    node_ip="$(srun --overlap -N1 -n1 -w "${node}" hostname -I | awk '{print $1}')"
    [ -n "${node_ip}" ] || die "failed to resolve node IP for ${node}"
    slurm_node_records+=("${idx}:${node}:${node_ip}")
    if [ "${idx}" = "0" ]; then
        ray_head_ip="${node_ip}"
    fi
done
[ -n "${ray_head_ip}" ] || die "failed to resolve Ray head IP"
sglang_router_host="${ray_head_ip}"

# Use job-specific ports so repeated debug submissions do not collide with
# stale host-network services from earlier allocations.
port_slot=$((SLURM_JOB_ID % 1000))
sglang_worker_port_block_size="${sglang_worker_port_block_size:-320}"
sglang_worker_port_slot_count="${sglang_worker_port_slot_count:-20}"
sglang_worker_port_floor="${sglang_worker_port_floor:-2048}"
sglang_worker_port_slot=$((SLURM_JOB_ID % sglang_worker_port_slot_count))
ray_port="${ray_port:-$((30000 + port_slot))}"
ray_dashboard_port="${ray_dashboard_port:-$((28000 + port_slot))}"
rollout_port="${rollout_port:-$((18000 + port_slot))}"
gateway_port="${gateway_port:-$((20000 + port_slot))}"
sglang_router_port="${sglang_router_port:-$((26000 + port_slot))}"
slime_sglang_base_port="${slime_sglang_base_port:-$((sglang_worker_port_floor + sglang_worker_port_slot * sglang_worker_port_block_size))}"
cleanup_stale_sglang="${cleanup_stale_sglang:-1}"

export project_root script_dir train_sqsh container_mounts
export tmax_dataset_dir tmax_image_dir
export run_id run_label run_dir run_log_dir rollout_save_dir save_dir stop_file
export num_nodes="${nodes}"
export gpus_per_node="${gpus_per_node:-8}"
export total_gpus="$((num_nodes * gpus_per_node))"
export train_node_count="${train_node_count:-2}"
export train_num_gpus="${train_num_gpus:-$((train_node_count * gpus_per_node))}"
export actor_num_nodes="${actor_num_nodes:-${train_node_count}}"
export actor_num_gpus_per_node="${actor_num_gpus_per_node:-8}"
export rollout_num_gpus="${rollout_num_gpus:-$((total_gpus - train_num_gpus))}"
export rollout_num_gpus_per_engine="${rollout_num_gpus_per_engine:-1}"
export tensor_model_parallel_size="${tensor_model_parallel_size:-4}"
export ray_head_num_gpus="${ray_head_num_gpus:-${gpus_per_node}}"
export ray_expected_num_gpus="${ray_expected_num_gpus:-${total_gpus}}"
export ray_cluster_timeout_seconds="${ray_cluster_timeout_seconds:-900}"
export ray_num_cpus="${ray_num_cpus:-128}"

positive_int "${train_node_count}" || die "train_node_count must be a positive integer"
positive_int "${train_num_gpus}" || die "train_num_gpus must be a positive integer"
positive_int "${actor_num_nodes}" || die "actor_num_nodes must be a positive integer"
positive_int "${rollout_num_gpus}" || die "rollout_num_gpus must be a positive integer"
positive_int "${tensor_model_parallel_size}" || die "tensor_model_parallel_size must be a positive integer"
[ "$((train_num_gpus + rollout_num_gpus))" -le "${total_gpus}" ] || die "train + rollout GPUs exceed total"
[ "$((train_num_gpus % tensor_model_parallel_size))" -eq 0 ] || die "train_num_gpus must be divisible by TP"

# Full ready TMax prompt pool. The rollout boundary keeps profiling jobs short.
export tmax_scan_tasks="${tmax_scan_tasks:-14601}"
export tmax_only_ready="${tmax_only_ready:-1}"
export smoke_rows="${smoke_rows:-0}"
export rollout_batch_size="${rollout_batch_size:-8}"
export n_samples_per_prompt="${n_samples_per_prompt:-8}"
export num_rollout="${num_rollout:-2}"
export target_num_rollout="${target_num_rollout:-${num_rollout}}"
export save_interval="${save_interval:-1}"
export global_batch_size="${global_batch_size:-$((rollout_batch_size * n_samples_per_prompt))}"
positive_int "${tmax_scan_tasks}" || die "tmax_scan_tasks must be positive"
nonnegative_int "${smoke_rows}" || die "smoke_rows must be a non-negative integer"
positive_int "${rollout_batch_size}" || die "rollout_batch_size must be positive"
positive_int "${n_samples_per_prompt}" || die "n_samples_per_prompt must be positive"
positive_int "${num_rollout}" || die "num_rollout must be positive"
positive_int "${global_batch_size}" || die "global_batch_size must be positive"
[ "${global_batch_size}" -eq "$((rollout_batch_size * n_samples_per_prompt))" ] || \
    die "global_batch_size must equal rollout_batch_size*n_samples_per_prompt"

export qwen_gdn_backend=fla
export attention_backend="${attention_backend:-flash}"
export pi_fail_on_context_limit="${pi_fail_on_context_limit:-0}"
export pi_context_window="${pi_context_window:-32000}"
export pi_max_tokens="${pi_max_tokens:-512}"
export fetch_trajectory_retry_times="${fetch_trajectory_retry_times:-3}"
export max_tokens_per_gpu="${max_tokens_per_gpu:-50000}"
export sglang_context_length="${sglang_context_length:-50000}"
export sglang_mem_fraction_static="${sglang_mem_fraction_static:-0.7}"
export sglang_cuda_graph_max_bs="${sglang_cuda_graph_max_bs:-}"
export sglang_disable_custom_all_reduce="${sglang_disable_custom_all_reduce:-0}"
export sglang_disable_cuda_graph="${sglang_disable_cuda_graph:-0}"
export rollout_max_prompt_len="${rollout_max_prompt_len:-32000}"
export rollout_max_response_len="${rollout_max_response_len:-4096}"
export polar_max_async_level="${polar_max_async_level:-4}"
export polar_min_complete_accept_fraction="${polar_min_complete_accept_fraction:-0.5}"
export polar_task_timeout_from_metadata="${polar_task_timeout_from_metadata:-1}"
export polar_request_timeout="${polar_request_timeout:-2400}"
export polar_task_timeout_seconds="${polar_task_timeout_seconds:-1800}"

export polar_multi_gateway="${polar_multi_gateway:-1}"
export polar_gateway_count="${polar_gateway_count:-$((num_nodes - train_node_count))}"
export polar_gateway_max_init_workers="${polar_gateway_max_init_workers:-16}"
export polar_gateway_max_run_workers="${polar_gateway_max_run_workers:-128}"
export polar_gateway_max_postrun_workers="${polar_gateway_max_postrun_workers:-64}"
positive_int "${polar_max_async_level}" || die "polar_max_async_level must be positive"
positive_int "${polar_gateway_count}" || die "polar_gateway_count must be positive"
[ "${polar_gateway_count}" -le "${num_nodes}" ] || die "polar_gateway_count exceeds num_nodes"

if [ -z "${polar_gateway_hosts:-}" ]; then
    start_rank="${gateway_start_rank:-${train_node_count}}"
    [ "${start_rank}" -lt "${num_nodes}" ] || die "gateway_start_rank ${start_rank} leaves no gateway nodes"
    mapfile -t gateway_selection < <(python3 - "${polar_gateway_count}" "${start_rank}" "${slurm_node_records[@]}" <<'PYSEL'
import sys

count = int(sys.argv[1])
start_rank = int(sys.argv[2])
records = []
for raw in sys.argv[3:]:
    rank, node, ip = raw.split(":", 2)
    records.append((int(rank), node, ip))
gateways = [item for item in records if item[0] >= start_rank][:count]
if len(gateways) != count:
    raise SystemExit(f"failed to select {count} gateway host(s) from ranks >= {start_rank}")
print("GATEWAY_HOSTS=" + ",".join(item[1] for item in gateways))
print("GATEWAY_RANKS=" + ",".join(str(item[0]) for item in gateways))
PYSEL
)
    for item in "${gateway_selection[@]}"; do
        case "${item}" in
            GATEWAY_HOSTS=*) polar_gateway_hosts="${item#GATEWAY_HOSTS=}" ;;
            GATEWAY_RANKS=*) polar_gateway_ranks="${item#GATEWAY_RANKS=}" ;;
        esac
    done
elif [ -z "${polar_gateway_ranks:-}" ]; then
    polar_gateway_ranks="$(seq -s, 0 $((polar_gateway_count - 1)))"
fi
export polar_gateway_hosts polar_gateway_ranks

export dynamic_sampling_filter_path="${dynamic_sampling_filter_path:-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
export calculate_per_token_loss="${calculate_per_token_loss:-1}"
export grpo_std_normalization="${grpo_std_normalization:-0}"
export kl_loss_coef="${kl_loss_coef:-0.0}"
export kl_loss_type="${kl_loss_type:-k2}"
export use_tis="${use_tis:-0}"
export eps_clip="${eps_clip:-0.2}"
export eps_clip_high="${eps_clip_high:-0.28}"
export eps_clip_c="${eps_clip_c:-10.0}"
export train_lr="${train_lr:-5e-7}"
export clip_grad="${clip_grad:-1.0}"

export ray_port ray_dashboard_port rollout_port gateway_port
export sglang_router_port slime_sglang_base_port cleanup_stale_sglang
export ray_head_ip sglang_router_host
export POLAR_APPTAINER_DIRECT_EXEC="${POLAR_APPTAINER_DIRECT_EXEC:-1}"
export POLAR_APPTAINER_EXEC_MODE="${POLAR_APPTAINER_EXEC_MODE:-direct}"

export use_wandb="${use_wandb:-1}"
export wandb_mode="${wandb_mode:-online}"
export wandb_entity="${wandb_entity:-hwinf_dcm}"
export wandb_project="${wandb_project:-harness-tma}"
export wandb_group="${wandb_group:-8n-dapo-profile}"
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

rank_is_gateway() {
    [ "${polar_multi_gateway}" = "1" ] || return 1
    local ranks="${polar_gateway_ranks:-}"
    if [ -z "${ranks}" ]; then
        ranks="$(seq -s, 0 $((polar_gateway_count - 1)))"
    fi
    case ",${ranks}," in
        *",${rank},"*) return 0 ;;
        *) return 1 ;;
    esac
}

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

cleanup_stale_sglang_processes() {
    if [ "${cleanup_stale_sglang}" != "1" ]; then
        echo "cleanup_stale_sglang=${cleanup_stale_sglang}; skipping stale SGLang cleanup on ${node}"
        return
    fi
    if ! command -v pgrep >/dev/null 2>&1; then
        echo "pgrep not available; skipping stale SGLang cleanup on ${node}"
        return
    fi
    local user_id pattern pids
    user_id="$(id -u)"
    pattern='SGLangEngine|sglang_router|sglang[.]srt|sglang[.]launch'
    pids="$(pgrep -u "${user_id}" -f "${pattern}" || true)"
    if [ -z "${pids}" ]; then
        echo "No stale SGLang/router processes found on ${node}"
        return
    fi
    echo "Stopping stale SGLang/router processes on ${node}: $(printf '%s' "${pids}" | tr '\n' ' ')"
    kill ${pids} >/dev/null 2>&1 || true
    sleep 2
    pids="$(pgrep -u "${user_id}" -f "${pattern}" || true)"
    if [ -n "${pids}" ]; then
        echo "Force-stopping stale SGLang/router processes on ${node}: $(printf '%s' "${pids}" | tr '\n' ' ')"
        kill -9 ${pids} >/dev/null 2>&1 || true
    fi
}

cleanup_stale_sglang_processes >"${run_log_dir}/stale-sglang-cleanup-rank-${rank}.log" 2>&1
ray stop --force >/dev/null 2>&1 || true
monitor_pid=""
ray_pid=""
gateway_pid=""
cleanup() {
    [ -z "${monitor_pid}" ] || kill "${monitor_pid}" 2>/dev/null || true
    [ -z "${gateway_pid}" ] || kill "${gateway_pid}" 2>/dev/null || true
    [ -z "${gateway_pid}" ] || wait "${gateway_pid}" 2>/dev/null || true
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
for _ in range(450):
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
    if rank_is_gateway; then
        RAY_NODE_RANK="${rank}" pi_multigw_sidecar=1 ray_use_existing_cluster=1 ray_stop_on_exit=0 \
            bash "${script_dir}/run_tmax_pi_apptainer_train.sh" \
            >"${run_log_dir}/gateway-sidecar-rank-${rank}.driver.log" 2>&1 &
        gateway_pid="$!"
    fi
    while [ ! -f "${stop_file}" ]; do
        kill -0 "${ray_pid}" 2>/dev/null || { wait "${ray_pid}"; exit $?; }
        if [ -n "${gateway_pid}" ]; then
            kill -0 "${gateway_pid}" 2>/dev/null || { wait "${gateway_pid}"; exit $?; }
        fi
        sleep 5
    done
fi
WORKER
chmod +x "${worker_script}"

cat <<SUMMARY
============================================================
webarea TMax PI 8-node DAPO profile
  job_id:    ${SLURM_JOB_ID}
  job_name:  ${SLURM_JOB_NAME}
  profile:   ${profile_id}
  nodes:     ${slurm_nodes[*]}
  ray head:  ${slurm_nodes[0]} (${ray_head_ip})
  run:       ${run_id}
  account:   ${job_account}
  partition: ${SLURM_JOB_PARTITION:-unknown}
  reservation: ${SLURM_JOB_RESERVATION:-unknown}
  GPUs:      ${train_num_gpus} train + ${rollout_num_gpus} rollout = ${total_gpus}
  train:     nodes=${train_node_count}, actor_nodes=${actor_num_nodes}, actor_gpus_per_node=${actor_num_gpus_per_node}
  rollout:   engines=$((rollout_num_gpus / rollout_num_gpus_per_engine)), gpus_per_engine=${rollout_num_gpus_per_engine}
  gateways:  ${polar_gateway_count} (${polar_gateway_hosts}), ranks=${polar_gateway_ranks}, workers init/run/post=${polar_gateway_max_init_workers}/${polar_gateway_max_run_workers}/${polar_gateway_max_postrun_workers}
  TP/DP:     ${tensor_model_parallel_size}/$((train_num_gpus / tensor_model_parallel_size))
  batch:     ${rollout_batch_size} prompts x ${n_samples_per_prompt} samples = ${global_batch_size} trajectories
  async:     max_async=${polar_max_async_level}, min_complete_accept_fraction=${polar_min_complete_accept_fraction}
  data:      tmax_scan_tasks=${tmax_scan_tasks}, smoke_rows=${smoke_rows}
  boundary:  ${num_rollout}/${target_num_rollout}
  caps:      max_tokens_per_gpu=${max_tokens_per_gpu}, sglang_context=${sglang_context_length}, response=${rollout_max_response_len}, pi_tokens=${pi_max_tokens}
  sglang:    mem_fraction=${sglang_mem_fraction_static}, router_port=${sglang_router_port}, worker_base_port=${slime_sglang_base_port}, cuda_graph_max_bs=${sglang_cuda_graph_max_bs:-default}, disable_custom_all_reduce=${sglang_disable_custom_all_reduce}, disable_cuda_graph=${sglang_disable_cuda_graph}
  DAPO:      dynamic_sampling=${dynamic_sampling_filter_path}, per_token=${calculate_per_token_loss}, std_norm=${grpo_std_normalization}, KL=${kl_loss_coef}
  W&B:       ${wandb_entity}/${wandb_project}/${wandb_run_id}
============================================================
SUMMARY
date -u +%FT%TZ

srun --overlap --kill-on-bad-exit=1 \
    --nodes="${num_nodes}" --ntasks="${num_nodes}" --ntasks-per-node=1 \
    --gres="gpu:${gpus_per_node}" --cpus-per-task="${ray_num_cpus}" \
    --container-image="${train_sqsh}" \
    --container-mounts="${container_mounts}" \
    --container-workdir="${project_root}" \
    --container-writable --no-container-mount-home \
    bash "${worker_script}"
