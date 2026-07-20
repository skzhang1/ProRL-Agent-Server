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
# Same-name resubmissions run serially and auto-resume from the latest
# checkpoint, so an externally cancelled job (e.g. the cluster's idle-GPU
# reaper) only loses the step in progress.
#SBATCH --dependency=singleton
# pool1-00187 externally SIGKILLs Ray processes ~20 min into every job that
# lands on it (jobs 5229386, 5229849, 5230998, 5243843); no OOM, no error.
# pool1-00120's SGLang engines stop answering mid-run (jobs 5229658, 5243844),
# and engine recreation on the sick node hangs update_weights until the idle
# reaper kills the job.
# pool1-00202: raylet SIGKILL at 21 min (5243847), same class as 00187.
#SBATCH --exclude=pool1-[00001-00224],pool1-00266,pool0-[00126,04476,04843,04892,04982,05158,05432],pool0-[05800-05984]
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

# Defaults reproduce the first fully successful 5-step run (job 5249486,
# 2026-07-15, ~19 min/step): 1 train node (TP4/DP2, 8 train + 56 rollout GPUs),
# gateway workers 24/96/48, SGLang cuda_graph_max_bs=32. Change profile_id to
# start a fresh lineage; identical profile_id resumes its checkpoints.
profile_id="${profile_id:-b8_s16_a4_tn1_gw96_cg32_t480_mcf05}"
# The run identity is intentionally job-id-free: chained singleton jobs share
# the same checkpoints, prompt data, and W&B run, and resume automatically.
run_id="${run_id:-webarea-debug_tmax_pi_8n_dapo_${profile_id}}"
run_label="${run_label:-8n-dapo-${profile_id}}"
run_dir="${run_dir:-${project_root}/tmp/${run_id}}"
run_log_dir="${run_log_dir:-${run_dir}/logs/job-${SLURM_JOB_ID}}"
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
    # This probe only needs a tiny step; keep it from inheriting the training
    # job's large CPU/memory request, which can make Slurm reject the step.
    node_ip="$(srun --overlap --cpus-per-task=1 --mem=1G -N1 -n1 -w "${node}" hostname -I | awk '{print $1}')"
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
export train_node_count="${train_node_count:-1}"
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
# Raylet reads this at `ray start`; the inner trainer sets it too late to
# affect the host-memory monitor. 0.99 avoids false SIGKILLs of SGLang
# actors from CUDA/page-cache memory accounting on rollout nodes.
export ray_memory_usage_threshold="${ray_memory_usage_threshold:-0.99}"
# Bound collective hangs: a dead SGLang engine inside the update_weights
# broadcast (seen in 5243844) otherwise blocks for the Megatron default 180
# minutes and the job dies to the idle reaper instead of failing fast and
# letting the singleton chain resume from the checkpoint.
export distributed_timeout_minutes="${distributed_timeout_minutes:-15}"

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
export n_samples_per_prompt="${n_samples_per_prompt:-16}"
# Single global boundary. Each job resumes from the latest checkpoint and
# trains toward this rollout count: it exits cleanly when the target is
# reached, or gets cut by walltime and the next singleton job continues.
# No per-job step budget is needed. (num_rollout is accepted as a legacy
# alias and forced equal internally for the inner launcher.)
export target_num_rollout="${target_num_rollout:-${num_rollout:-5}}"
export num_rollout="${target_num_rollout}"
# Ask Slime to stop at a rollout boundary once this many minutes have passed
# since initialization, leaving headroom inside the 4h walltime for the final
# synchronous checkpoint instead of losing the in-flight step to SIGTERM.
export exit_duration_minutes="${exit_duration_minutes:-210}"
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

# Singleton auto-resume: Slime writes checkpoint iteration N only after
# rollout N has trained, so completed rollouts = N + 1. A fresh save
# directory starts at rollout zero; a finished run exits immediately.
start_rollout_id="${start_rollout_id:-}"
if [ -z "${start_rollout_id}" ]; then
    completed_rollouts=0
    latest_file="${save_dir}/latest_checkpointed_iteration.txt"
    if [ -s "${latest_file}" ]; then
        latest_iteration="$(tr -d '[:space:]' <"${latest_file}")"
        nonnegative_int "${latest_iteration}" || die "invalid checkpoint iteration in ${latest_file}: ${latest_iteration}"
        completed_rollouts=$((latest_iteration + 1))
    fi
    if [ "${completed_rollouts}" -ge "${target_num_rollout}" ]; then
        printf 'Training already complete: %s/%s rollouts in %s\n' \
            "${completed_rollouts}" "${target_num_rollout}" "${save_dir}"
        exit 0
    fi
    start_rollout_id="${completed_rollouts}"
fi
export start_rollout_id

export qwen_gdn_backend=fla
export attention_backend="${attention_backend:-flash}"
export pi_fail_on_context_limit="${pi_fail_on_context_limit:-0}"
export pi_context_window="${pi_context_window:-32000}"
export pi_max_tokens="${pi_max_tokens:-512}"
export fetch_trajectory_retry_times="${fetch_trajectory_retry_times:-3}"
export max_tokens_per_gpu="${max_tokens_per_gpu:-50000}"
export sglang_context_length="${sglang_context_length:-50000}"
export sglang_mem_fraction_static="${sglang_mem_fraction_static:-0.7}"
export sglang_cuda_graph_max_bs="${sglang_cuda_graph_max_bs:-32}"
export sglang_disable_custom_all_reduce="${sglang_disable_custom_all_reduce:-0}"
export sglang_disable_cuda_graph="${sglang_disable_cuda_graph:-0}"
export rollout_max_prompt_len="${rollout_max_prompt_len:-32000}"
export rollout_max_response_len="${rollout_max_response_len:-4096}"
export polar_max_async_level="${polar_max_async_level:-4}"
export polar_min_complete_accept_fraction="${polar_min_complete_accept_fraction:-0.5}"
export polar_task_timeout_from_metadata="${polar_task_timeout_from_metadata:-0}"
export polar_request_timeout="${polar_request_timeout:-1200}"
export polar_task_timeout_seconds="${polar_task_timeout_seconds:-480}"

export polar_multi_gateway="${polar_multi_gateway:-1}"
export polar_gateway_count="${polar_gateway_count:-$((num_nodes - train_node_count))}"
export polar_gateway_max_init_workers="${polar_gateway_max_init_workers:-24}"
export polar_gateway_max_run_workers="${polar_gateway_max_run_workers:-96}"
export polar_gateway_max_postrun_workers="${polar_gateway_max_postrun_workers:-48}"
export polar_gateway_max_restarts="${polar_gateway_max_restarts:-20}"
export ray_worker_max_restarts="${ray_worker_max_restarts:-3}"
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
trace_log="${run_log_dir}/worker-rank-${rank}.trace.log"
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
# Must be visible to `ray start` below; the inner trainer exports it too
# late to affect raylet's host-memory monitor (false kills otherwise).
export RAY_MEMORY_USAGE_THRESHOLD="${ray_memory_usage_threshold:-0.99}"
export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}"
mkdir -p "${HOME}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_WORKDIR}" \
    "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" "${XDG_RUNTIME_DIR}" "${CUDA_CACHE_PATH}" "${NUMBA_CACHE_DIR}" \
    "${ray_tmpdir}" "${job_cache_root}"
chmod 700 "${XDG_RUNTIME_DIR}"

trace() {
    printf '%s rank=%s node=%s pid=%s %s\n' "$(date -u +%FT%TZ)" "${rank}" "${node}" "$$" "$*" >>"${trace_log}" 2>/dev/null || true
}

stop_file_state() {
    if [ -f "${stop_file}" ]; then
        stat -c 'present mtime=%y size=%s' "${stop_file}" 2>/dev/null || printf 'present'
    else
        printf 'absent'
    fi
}

trace "worker start node_ip=${node_ip} cache_root=${cache_root} stop_file=$(stop_file_state)"

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
        echo "cleanup_stale_sglang=${cleanup_stale_sglang}; skipping stale runtime cleanup on ${node}"
        return
    fi
    if ! command -v pgrep >/dev/null 2>&1; then
        echo "pgrep not available; skipping stale runtime cleanup on ${node}"
        return
    fi
    local user_id pattern pids
    user_id="$(id -u)"
    pattern='SGLangEngine|sglang_router|sglang[.]srt|sglang[.]launch|raylet|gcs_server|ray::|ray-dashboard|log_monitor|train_async[.]py|polar serve_(rollout|gateway)|run_tmax_pi_apptainer_train[.]sh|wandb-core|wandb-xpu|squashfuse_ll|fuse-overlayfs'
    pids="$(pgrep -u "${user_id}" -f "${pattern}" || true)"
    if [ -z "${pids}" ]; then
        echo "No stale Ray/Polar/SGLang runtime processes found on ${node}"
        return
    fi
    echo "Stopping stale Ray/Polar/SGLang runtime processes on ${node}: $(printf '%s' "${pids}" | tr '\n' ' ')"
    kill ${pids} >/dev/null 2>&1 || true
    sleep 2
    pids="$(pgrep -u "${user_id}" -f "${pattern}" || true)"
    if [ -n "${pids}" ]; then
        echo "Force-stopping stale Ray/Polar/SGLang runtime processes on ${node}: $(printf '%s' "${pids}" | tr '\n' ' ')"
        kill -9 ${pids} >/dev/null 2>&1 || true
    fi
}

cleanup_stale_sglang_processes >"${run_log_dir}/stale-sglang-cleanup-rank-${rank}.log" 2>&1
ray stop --force >/dev/null 2>&1 || true
monitor_pid=""
resource_monitor_pid=""
ray_pid=""
gateway_pid=""
dump_cgroup_memory() {
    local cg path f
    cg="$(awk -F: '$1 == "0" {print $3; exit}' /proc/self/cgroup 2>/dev/null || true)"
    [ -n "${cg}" ] || cg="/"
    path="/sys/fs/cgroup${cg}"
    printf 'cgroup=%s path=%s\n' "${cg}" "${path}"
    if [ -d "${path}" ]; then
        for f in memory.current memory.peak memory.max memory.high memory.events memory.swap.current memory.swap.max; do
            if [ -f "${path}/${f}" ]; then
                printf '%s=' "${f}"
                cat "${path}/${f}" 2>/dev/null || true
            fi
        done
    fi
}
resource_snapshot() {
    printf '===== %s node=%s rank=%s stop_file=%s =====\n' "$(date -u +%FT%TZ)" "${node}" "${rank}" "$(stop_file_state)"
    dump_cgroup_memory
    free -m 2>/dev/null || true
    df -h /tmp "${cache_root}" 2>/dev/null || true
    ps -u "$(id -u)" -o pid,ppid,stat,comm,rss,vsz,etime,args --sort=-rss 2>/dev/null \
        | awk 'NR == 1 {print; next} /raylet|log_monitor|SGLangEngine|sglang|python|uvicorn|polar|apptainer|codex/ {print; count++; if (count >= 80) exit}'
}
archive_ray_local_logs() {
    local dst files f rel
    dst="${run_log_dir}/ray-local-rank-${rank}"
    files="${dst}/files.txt"
    mkdir -p "${dst}"
    {
        printf 'timestamp=%s\n' "$(date -u +%FT%TZ)"
        printf 'node=%s\nrank=%s\nnode_ip=%s\nray_tmpdir=%s\n' "${node}" "${rank}" "${node_ip}" "${ray_tmpdir}"
    } >"${dst}/manifest.txt"
    if [ ! -d "${ray_tmpdir}" ]; then
        printf 'missing ray_tmpdir\n' >>"${dst}/manifest.txt"
        return
    fi
    find "${ray_tmpdir}" -maxdepth 4 -type f \
        \( -name 'ray_process_exit.log' \
           -o -name 'raylet.err' \
           -o -name 'raylet.out' \
           -o -name 'gcs_server.err' \
           -o -name 'gcs_server.out' \
           -o -name 'dashboard_agent.log' \
           -o -name 'runtime_env_agent.log' \
           -o -name 'worker-*.err' \) \
        -print >"${files}" 2>/dev/null || true
    while IFS= read -r f; do
        [ -n "${f}" ] || continue
        rel="${f#${ray_tmpdir}/}"
        mkdir -p "${dst}/$(dirname "${rel}")"
        cp -p "${f}" "${dst}/${rel}" 2>/dev/null || true
    done <"${files}"
}
cleanup() {
    local cleanup_rc="$?"
    trace "cleanup enter rc=${cleanup_rc} monitor_pid=${monitor_pid:-} ray_pid=${ray_pid:-} gateway_pid=${gateway_pid:-} stop_file=$(stop_file_state)"
    resource_snapshot >>"${run_log_dir}/resource-rank-${rank}.log" 2>&1 || true
    # Kernel-side evidence for the external SIGKILL class: an OOM or kernel
    # kill shows up here; silence implicates a userspace agent instead.
    dmesg -T 2>/dev/null | tail -120 >"${run_log_dir}/dmesg-rank-${rank}.log" || true
    archive_ray_local_logs || true
    [ -z "${monitor_pid}" ] || kill "${monitor_pid}" 2>/dev/null || true
    [ -z "${resource_monitor_pid}" ] || kill "${resource_monitor_pid}" 2>/dev/null || true
    [ -z "${gateway_pid}" ] || kill "${gateway_pid}" 2>/dev/null || true
    [ -z "${gateway_pid}" ] || wait "${gateway_pid}" 2>/dev/null || true
    if [ -n "${ray_pid}" ]; then
        kill "${ray_pid}" 2>/dev/null || true
        wait "${ray_pid}" 2>/dev/null || true
    fi
    ray stop --force >/dev/null 2>&1 || true
    trace "cleanup done rc=${cleanup_rc} stop_file=$(stop_file_state)"
}
trap cleanup EXIT
trap 'trace "signal SIGTERM received stop_file=$(stop_file_state)"; exit 143' TERM
trap 'trace "signal SIGINT received stop_file=$(stop_file_state)"; exit 130' INT

(
    while true; do
        printf '%s,%s,%s,' "$(date -u +%FT%TZ)" "${node}" "${rank}"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits | paste -sd ';' -
        sleep 30
    done
) >>"${run_log_dir}/gpu-rank-${rank}.csv" 2>&1 &
monitor_pid="$!"

(
    while true; do
        resource_snapshot
        sleep 15
    done
) >>"${run_log_dir}/resource-rank-${rank}.log" 2>&1 &
resource_monitor_pid="$!"

if [ "${rank}" = "0" ]; then
    trace "starting ray head ${ray_head_ip}:${ray_port}"
    ray start --head --node-ip-address="${ray_head_ip}" --port="${ray_port}" \
        --dashboard-host=0.0.0.0 --dashboard-port="${ray_dashboard_port}" \
        --num-cpus="${ray_num_cpus}" --num-gpus="${gpus_per_node}" \
        --temp-dir="${ray_tmpdir}" --disable-usage-stats \
        >"${run_log_dir}/ray-head.log" 2>&1

    finish_workers() {
        trace "touch stop_file"
        touch "${stop_file}"
    }
    trap 'finish_workers; cleanup' EXIT

    trace "starting train driver"
    set +e
    ray_use_existing_cluster=1 ray_stop_on_exit=0 \
        bash "${script_dir}/run_tmax_pi_apptainer_train.sh"
    train_rc="$?"
    set -e
    trace "train driver exited rc=${train_rc}"
    exit "${train_rc}"
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
    # Ray's log monitor tails every worker/SGLang log on the node and has
    # repeatedly been the first process SIGKILLed on rollout ranks; its death
    # makes `ray start --block` exit and takes the whole job down. Disable it
    # off the head node; all logs remain on disk and in run_log_dir archives.
    #
    # Something node-local also SIGKILLs `ray start --block` itself on one
    # rollout rank in most runs (5243843/44/47, 5244304/05, on five different
    # nodes). Restart the Ray worker in place: the node re-registers with the
    # head and Slime's rollout fault tolerance recreates the lost engines.
    ray_worker_restarts=0
    start_ray_worker_bg() {
        ray start --address="${ray_head_ip}:${ray_port}" --node-ip-address="${node_ip}" \
            --num-cpus="${ray_num_cpus}" --num-gpus="${gpus_per_node}" \
            --temp-dir="${ray_tmpdir}" --disable-usage-stats \
            --include-log-monitor=false --block \
            >>"${run_log_dir}/ray-worker-${rank}.log" 2>&1 &
        ray_pid="$!"
        trace "started ray worker pid=${ray_pid} restart=${ray_worker_restarts}"
    }
    start_ray_worker_bg
    gateway_restarts=0
    start_gateway_sidecar_bg() {
        RAY_NODE_RANK="${rank}" pi_multigw_sidecar=1 ray_use_existing_cluster=1 ray_stop_on_exit=0 \
            bash "${script_dir}/run_tmax_pi_apptainer_train.sh" \
            >>"${run_log_dir}/gateway-sidecar-rank-${rank}.driver.log" 2>&1 &
        gateway_pid="$!"
        trace "started gateway sidecar pid=${gateway_pid} restart=${gateway_restarts}"
    }
    if rank_is_gateway; then
        start_gateway_sidecar_bg
    fi
    while [ ! -f "${stop_file}" ]; do
        if ! kill -0 "${ray_pid}" 2>/dev/null; then
            set +e
            wait "${ray_pid}"
            child_rc="$?"
            set -e
            trace "ray worker pid=${ray_pid} exited rc=${child_rc} restart=${ray_worker_restarts} stop_file=$(stop_file_state)"
            ray_worker_restarts=$((ray_worker_restarts + 1))
            if [ "${ray_worker_restarts}" -gt "${ray_worker_max_restarts}" ]; then
                trace "ray worker exceeded restart limit ${ray_worker_max_restarts}"
                exit "${child_rc}"
            fi
            ray stop --force >/dev/null 2>&1 || true
            sleep 5
            start_ray_worker_bg
        fi
        if [ -n "${gateway_pid}" ]; then
            if ! kill -0 "${gateway_pid}" 2>/dev/null; then
                set +e
                wait "${gateway_pid}"
                child_rc="$?"
                set -e
                trace "gateway sidecar pid=${gateway_pid} exited rc=${child_rc} restart=${gateway_restarts} stop_file=$(stop_file_state)"
                gateway_restarts=$((gateway_restarts + 1))
                if [ "${gateway_restarts}" -gt "${polar_gateway_max_restarts}" ]; then
                    trace "gateway sidecar exceeded restart limit ${polar_gateway_max_restarts}"
                    exit "${child_rc}"
                fi
                sleep 5
                start_gateway_sidecar_bg
            fi
        fi
        sleep 5
    done
    trace "stop_file observed; worker exiting normally stop_file=$(stop_file_state)"
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
  boundary:  target=${target_num_rollout} (start=${start_rollout_id}), graceful_exit=${exit_duration_minutes}min
  resilience: ray_mem_threshold=${ray_memory_usage_threshold}, gateway_restarts<=${polar_gateway_max_restarts}, fault_tolerance=${use_fault_tolerance:-1}
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
