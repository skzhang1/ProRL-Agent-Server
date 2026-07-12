#!/usr/bin/env bash
#SBATCH --job-name=webarea-distill-pi-q35-4n
#SBATCH --account=nemotron_omni_vision
#SBATCH --partition=batch_block1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
# Repeated `sbatch` calls with this same job name run serially.  A later job
# starts even if the previous one failed, then resumes the last valid checkpoint.
#SBATCH --dependency=singleton
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --exclude=pool1-[00001-00224],pool0-[00126,04476,04843,04892,04982,05158,05432],pool0-[05800-05984]
#SBATCH --output=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.err
#SBATCH --export=ALL

# Repeatable four-node PI + Qwen3.5-4B SWE-Gym GRPO training.
# Node 0 hosts 8 Megatron training GPUs; nodes 1-3 host 24 SGLang rollout GPUs.
# Submit this same file multiple times; each allocation discovers the newest
# checkpoint and runs toward the global target until the inner trainer reaches
# its graceful wall-clock budget.  No per-job rollout boundary is required.
set -euo pipefail

project_root="${project_root:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server}"
script_dir="${project_root}/examples/swegym_slime_grpo"
train_sqsh="${train_sqsh:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/docker/polr_swegym_qwen35_torch211_te2161_fa4b19_numpy126_scipy117_tebindcu130_20260707.sqsh}"
container_mounts="${container_mounts:-/lustre/fs1:/lustre/fs1,/lustre/fsw:/lustre/fsw}"
# Apptainer 1.5.2 cannot re-enter an instance namespace from this Pyxis image.
# Direct exec reuses the same host overlay and session bind across rollout stages.
POLAR_APPTAINER_DIRECT_EXEC="${POLAR_APPTAINER_DIRECT_EXEC:-1}"
# Use clean pinned Slime/Megatron checkouts so earlier diagnostic edits cannot
# leak into the production path.
slime_dir="${slime_dir:-${project_root}/tmp/swegym_deps/slime_v030_fla04_clean}"
megatron_dir="${megatron_dir:-${project_root}/tmp/swegym_deps/Megatron-LM_2604_clean}"

# Experiment shape. Keep the 4 x 16 group fixed so GRPO sees 64 samples/step.
gpus_per_node=8
num_nodes=4
total_gpus=32
train_num_gpus=8
actor_num_nodes=1
actor_num_gpus_per_node=8
rollout_num_gpus=24
rollout_num_gpus_per_engine=1
tensor_model_parallel_size=2
qwen_gdn_backend=fla
attention_backend=flash
qkv_format=thd
use_dynamic_batch_size=1
use_sequence_parallel=1
micro_batch_size=1
global_batch_size=64
load_debug_rollout_data=""
load_debug_rollout_data_subsample=""
rollout_batch_size=4
n_samples_per_prompt=16
num_epoch=1
# A ceil epoch is 74 optimizer updates. Slime numbers the first saved update as
# iteration 1 after rollout 0, so the exclusive rollout boundary must be 75 to
# produce iter_0000074. Singleton allocations discover the latest checkpoint
# and keep that global boundary. An explicit num_rollout keeps manual mode.
target_num_rollout="${target_num_rollout:-75}"
num_rollout="${num_rollout:-}"
start_rollout_id="${start_rollout_id:-}"
smoke_rows="${smoke_rows:-0}"
# Count this from the end of model/Ray initialization. Checking only at a safe
# rollout boundary and then completing one final prefetched rollout normally
# leaves roughly 10-20 minutes for a synchronous checkpoint and cleanup inside
# the four-hour Slurm allocation.
exit_duration_minutes="${exit_duration_minutes:-190}"

# Compact PI history before the 50k inference limit, then prefix-merge within
# each segment. TP=2/DP=4 matches the launch_e2e-style training topology.
max_tokens_per_gpu="${max_tokens_per_gpu:-50000}"
rollout_max_response_len=16000
rollout_max_prompt_len=32000
sglang_context_length=50000
sglang_mem_fraction_static=0.8
distributed_timeout_minutes=180
save_interval=1

# Long tool trajectories can overflow the exponential ratios in low_var_kl and
# built-in TIS even with a 30k token cap. Use the bounded k2 form and a lower LR.
train_lr=5e-7
clip_grad=0.5
kl_loss_coef=0.001
kl_loss_type=k2
use_tis=0
eps_clip=0.2
eps_clip_high=0.28
eps_clip_c=10.0

# PI/Polar settings.
agent_harness=pi
agent_label=pi
pi_api_type=openai-completions
# Trigger PI compaction early enough to absorb a large tool result without
# overshooting SGLang's hard 50k request limit.
pi_context_window=24000
pi_max_tokens=512
# Compaction creates a clean prefix break; prefix_merging starts a new trace at
# that boundary instead of turning the session into a context-limit failure.
pi_fail_on_context_limit=0
polar_builder_strategy=prefix_merging
polar_max_async_level=4
polar_min_complete_accept_fraction=0.6
polar_multi_gateway=1
polar_gateway_count=4
polar_gateway_max_init_workers=24
polar_gateway_max_run_workers=192
polar_gateway_max_postrun_workers=96
# Current Polar correctly rejects memory limits for the Apptainer backend.
polar_runtime_memory_mb=""
polar_task_timeout_seconds=900
polar_request_timeout=900

# Stable identity shared by checkpoints and W&B.  Like MODEL_NAME in the SFT
# singleton launcher, edit/override experiment_name once when starting a new
# experiment; repeated plain `sbatch` calls then resume this same run.
run_label="4n32g-train8-rollout24-tp2dp4-fa4b19-fla04-pmerge-k2"
experiment_name="${experiment_name:-webarea-distill_pi_q35_4n_full293_1ep_$(date -u +%Y%m%dT%H%M%SZ)}"
run_id="${run_id:-${experiment_name}}"
run_dir="${run_dir:-${project_root}/tmp/${run_id}}"
run_log_dir="${run_log_dir:-${run_dir}/logs/job-${SLURM_JOB_ID}}"
save_dir="${save_dir:-${project_root}/tmp/ckpt/${run_id}}"
rollout_save_dir="${rollout_save_dir:-${run_dir}/rollout_results}"

# Production invariant: every training allocation must report to the user's
# approved W&B destination.  Keep these fixed instead of allowing inherited
# submit-shell variables to silently redirect or disable tracking.
use_wandb=1
wandb_mode=online
wandb_entity=hwinf_dcm
wandb_project=harnessgen
wandb_group="${wandb_group:-${run_id}}"
wandb_run_id="${wandb_run_id:-${run_id}}"
wandb_random_suffix=0
wandb_api_key="${wandb_api_key:-${WANDB_API_KEY:-}}"

dry_run="${dry_run:-0}"
# Host networking can expose stale Polar listeners from an earlier allocation.
# Give each Slurm job its own control-plane ports instead of fixed 18080/18100.
port_slot=$((SLURM_JOB_ID % 1000))
rollout_port=$((18000 + port_slot))
gateway_port=$((20000 + port_slot))
ray_port="${ray_port:-6379}"
ray_dashboard_port="${ray_dashboard_port:-28265}"
ray_num_cpus=128
ray_expected_num_gpus=32
ray_cluster_timeout_seconds=600

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[ "${SLURM_JOB_NUM_NODES:-0}" = "4" ] || die "this script requires exactly four allocated nodes"
job_account="${SLURM_JOB_ACCOUNT:-}"
if [ -z "${job_account}" ]; then
    job_account="$(scontrol show job -o "${SLURM_JOB_ID}" | sed -n 's/.* Account=\([^ ]*\).*/\1/p')"
fi
case "${job_account}" in
    nemotron_omni_vision) ;;
    *) die "account ${job_account:-unknown} is not allowed; use nemotron_omni_vision" ;;
esac
[ -x "${script_dir}/run_pi_apptainer_train.sh" ] || die "missing inner launcher"
[ -f "${train_sqsh}" ] || die "missing training image: ${train_sqsh}"
[ -f "${script_dir}/swegym_train_293.jsonl" ] || die "missing SWE-Gym training data"
# The inner launcher creates the converted checkpoint and PI CLI when they are
# absent. This keeps a fresh durable worktree self-contained after cleanup.
case "${target_num_rollout}" in
    ''|*[!0-9]*) die "target_num_rollout must be a positive integer" ;;
esac
case "${exit_duration_minutes}" in
    ''|*[!0-9]*) die "exit_duration_minutes must be a positive integer" ;;
esac
[ "${target_num_rollout}" -eq 75 ] || die "target_num_rollout must remain 75 to produce checkpoint iteration 74"
[ "${exit_duration_minutes}" -ge 1 ] || die "exit_duration_minutes must be a positive integer"

# Automatic singleton resume.  Slime writes checkpoint iteration N only after
# rollout N has trained successfully, so the completed rollout count is N + 1.
# A fresh save directory starts at rollout zero.  The final queued singleton
# jobs exit immediately once the target is already complete.
resume_mode=manual
if [ -z "${num_rollout}" ]; then
    resume_mode=singleton-auto
    completed_rollouts=0
    latest_file="${save_dir}/latest_checkpointed_iteration.txt"
    if [ -s "${latest_file}" ]; then
        latest_iteration="$(<"${latest_file}")"
        case "${latest_iteration}" in
            ''|*[!0-9]*) die "invalid checkpoint iteration in ${latest_file}: ${latest_iteration}" ;;
        esac
        checkpoint_dir="${save_dir}/iter_$(printf '%07d' "${latest_iteration}")"
        [ -d "${checkpoint_dir}" ] || die "latest checkpoint directory is missing: ${checkpoint_dir}"
        completed_rollouts=$((latest_iteration + 1))
    fi

    if [ "${completed_rollouts}" -ge "${target_num_rollout}" ]; then
        printf 'Training already complete: %s/%s rollouts in %s\n' \
            "${completed_rollouts}" "${target_num_rollout}" "${save_dir}"
        exit 0
    fi

    start_rollout_id="${completed_rollouts}"
    num_rollout="${target_num_rollout}"
fi

[ "${num_rollout}" -ge 1 ] && [ "${num_rollout}" -le "${target_num_rollout}" ] || \
    die "num_rollout must be an absolute boundary in [1, ${target_num_rollout}]"

# Keep credentials out of source and process command lines. Before Pyxis
# starts, read the submit host's private netrc and forward only the environment
# value into the container.
if [ "${use_wandb}" = "1" ] && [ "${wandb_mode}" = "online" ] && [ -z "${wandb_api_key}" ]; then
    wandb_api_key="$(python3 - <<'PY'
import netrc

auth = netrc.netrc().authenticators("api.wandb.ai")
print(auth[2] if auth else "")
PY
)"
fi
if [ "${use_wandb}" = "1" ] && [ "${wandb_mode}" = "online" ] && [ -z "${wandb_api_key}" ]; then
    die "online W&B requires WANDB_API_KEY or an api.wandb.ai entry in ~/.netrc"
fi

# A release checkpoint at iteration zero has not consumed rollout zero. Only
# the first allocation needs this override; later allocations infer the next
# rollout id from the numbered checkpoint in save_dir.
if [ -z "${start_rollout_id}" ] && [ ! -s "${save_dir}/latest_checkpointed_iteration.txt" ]; then
    start_rollout_id=0
fi

mkdir -p "${project_root}/logs/slurm" "${run_log_dir}" "${save_dir}" "${rollout_save_dir}"
stop_file="${run_dir}/ray_workers.stop"
worker_script="${run_dir}/ray_cluster_rank.sh"
rm -f "${stop_file}"

mapfile -t slurm_nodes < <(scontrol show hostnames "${SLURM_NODELIST}")
[ "${#slurm_nodes[@]}" -eq 4 ] || die "expected four Slurm hosts"
head_node="${slurm_nodes[0]}"
ray_head_ip="$(srun --overlap -N1 -n1 -w "${head_node}" hostname -I | awk '{print $1}')"
[ -n "${ray_head_ip}" ] || die "failed to resolve Ray head IP"
sglang_router_host="${ray_head_ip}"
polar_gateway_hosts="$(IFS=,; printf '%s' "${slurm_nodes[*]}")"

# Lower-case variables are the inner launcher's public settings.
export project_root script_dir train_sqsh container_mounts slime_dir megatron_dir
export POLAR_APPTAINER_DIRECT_EXEC
export gpus_per_node num_nodes total_gpus train_num_gpus actor_num_nodes actor_num_gpus_per_node
export rollout_num_gpus rollout_num_gpus_per_engine tensor_model_parallel_size qwen_gdn_backend
export attention_backend qkv_format use_dynamic_batch_size use_sequence_parallel micro_batch_size global_batch_size
export load_debug_rollout_data load_debug_rollout_data_subsample
export rollout_batch_size n_samples_per_prompt num_epoch target_num_rollout num_rollout start_rollout_id smoke_rows
export exit_duration_minutes
export max_tokens_per_gpu rollout_max_response_len rollout_max_prompt_len
export sglang_context_length sglang_mem_fraction_static distributed_timeout_minutes save_interval
export train_lr clip_grad kl_loss_coef kl_loss_type use_tis eps_clip eps_clip_high eps_clip_c
export agent_harness agent_label pi_api_type pi_context_window pi_max_tokens pi_fail_on_context_limit
export polar_builder_strategy polar_max_async_level polar_min_complete_accept_fraction
export polar_multi_gateway polar_gateway_count polar_gateway_hosts
export polar_gateway_max_init_workers polar_gateway_max_run_workers polar_gateway_max_postrun_workers
export polar_runtime_memory_mb polar_task_timeout_seconds polar_request_timeout
export rollout_port gateway_port
export run_id run_label run_dir run_log_dir save_dir rollout_save_dir
export use_wandb wandb_mode wandb_entity wandb_project wandb_group wandb_run_id wandb_random_suffix wandb_api_key
export dry_run ray_port ray_dashboard_port ray_num_cpus ray_expected_num_gpus ray_cluster_timeout_seconds
export ray_head_ip sglang_router_host stop_file

cat >"${worker_script}" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

rank="${SLURM_PROCID}"
node="$(hostname)"
node_ip="$(hostname -I | awk '{print $1}')"
cache_root="/tmp/webarea-pi-${SLURM_JOB_ID}-${rank}"
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
mkdir -p "${HOME}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_WORKDIR}" \
    "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" "${XDG_RUNTIME_DIR}" "${CUDA_CACHE_PATH}" "${NUMBA_CACHE_DIR}" "${ray_tmpdir}"
chmod 700 "${XDG_RUNTIME_DIR}"

# Fail before Ray/model initialization if Slurm places us on a node without
# the shared training environment or a working CUDA runtime.
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

# The image is writable per node, so apply only package-local compatibility
# patches on every rank. Shared Slime/Megatron patches remain in the head path.
patch_container_runtime_only=1 bash "${script_dir}/run_pi_apptainer_train.sh" \
    >"${run_log_dir}/container-patch-rank-${rank}.log" 2>&1

ray stop --force >/dev/null 2>&1 || true
monitor_pid=""
gateway_pid=""
preserve_ray_logs() {
    local source_dir="${ray_tmpdir}/session_latest/logs"
    local target_dir="${run_log_dir}/ray-rank-${rank}"
    [ -d "${source_dir}" ] || return 0
    mkdir -p "${target_dir}"
    find -L "${source_dir}" -maxdepth 1 -type f \
        \( -name 'gcs_server.*' -o -name 'raylet.*' \
           -o -name 'ray_process_exit.log' -o -name 'dashboard*.log' \) \
        -exec cp -f {} "${target_dir}/" \; 2>/dev/null || true
}
cleanup() {
    [ -z "${monitor_pid}" ] || kill "${monitor_pid}" 2>/dev/null || true
    [ -z "${gateway_pid}" ] || kill "${gateway_pid}" 2>/dev/null || true
    [ -z "${gateway_pid}" ] || wait "${gateway_pid}" 2>/dev/null || true
    preserve_ray_logs
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Lightweight per-node GPU evidence; W&B records the learning curve itself.
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
        bash "${script_dir}/run_pi_apptainer_train.sh"
else
    python - "${ray_head_ip}" "${ray_port}" <<'PY'
import socket, sys, time
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
    if [ "${polar_multi_gateway}" = "1" ]; then
        RAY_NODE_RANK="${rank}" pi_multigw_sidecar=1 ray_use_existing_cluster=1 ray_stop_on_exit=0 \
            bash "${script_dir}/run_pi_apptainer_train.sh" >"${run_log_dir}/gateway-sidecar-rank-${rank}.driver.log" 2>&1 &
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
webarea PI SWE-Gym GRPO
  run:       ${run_id}
  nodes:     ${slurm_nodes[*]}
  ray head:  ${ray_head_ip}
  account:   ${job_account}
  GPUs:      8 train + 24 rollout
  gateways:  ${polar_gateway_count} (${polar_gateway_hosts}), per-gateway workers init/run/post=${polar_gateway_max_init_workers}/${polar_gateway_max_run_workers}/${polar_gateway_max_postrun_workers}
  CPUs:      ${ray_num_cpus} per node
  TP/DP:     ${tensor_model_parallel_size}/$((train_num_gpus / tensor_model_parallel_size))
  batch:     4 prompts x 16 samples = 64 trajectories
  scheduling:${resume_mode}, graceful budget=${exit_duration_minutes} min, singleton job name=${SLURM_JOB_NAME}
  boundary:  ${num_rollout}/${target_num_rollout} (start=${start_rollout_id:-checkpoint})
  stability: lr=${train_lr}, KL=${kl_loss_coef}, clip=${clip_grad}
  W&B:       ${wandb_entity}/${wandb_project}/${wandb_run_id}
============================================================
SUMMARY

srun --overlap --kill-on-bad-exit=1 --nodes="${num_nodes}" --ntasks="${num_nodes}" --ntasks-per-node=1 --gres="gpu:${gpus_per_node}" \
    --cpus-per-task="${ray_num_cpus}" \
    --container-image="${train_sqsh}" \
    --container-mounts="${container_mounts}" \
    --container-workdir="${project_root}" \
    --container-writable --no-container-mount-home \
    bash "${worker_script}"
