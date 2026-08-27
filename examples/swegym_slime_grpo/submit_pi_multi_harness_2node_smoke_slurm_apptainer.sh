#!/usr/bin/env bash
#SBATCH --job-name=webarea-mh-q35-2n-smoke2
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --exclude=pool1-[00001-00224],pool0-[00126,04476,04843,04892,04982,05158,05432],pool0-[05800-05984]
#SBATCH --output=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.err
#SBATCH --export=ALL

# Isolated two-node, two-step smoke run for multi-harness Qwen3.5-4B SWE-Gym
# GRPO. Node 0 hosts 8 Megatron training GPUs and node 1 hosts 8 rollout GPUs.
# This run has its own checkpoint and W&B identity and cannot resume or
# overwrite the production four-node experiment.
set -euo pipefail

project_root="${project_root:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server}"
script_dir="${project_root}/examples/swegym_slime_grpo"
inner_launcher="${script_dir}/run_pi_multi_harness_apptainer_train.sh"
train_sqsh="${train_sqsh:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/docker/polr_swegym_qwen35_torch211_te2161_fa4b19_numpy126_scipy117_tebindcu130_20260707.sqsh}"
container_mounts="${container_mounts:-/lustre/fs1:/lustre/fs1,/lustre/fsw:/lustre/fsw}"
# Apptainer 1.5.2 cannot re-enter an instance namespace from this Pyxis image.
# Direct exec reuses the same host overlay and session bind across rollout stages.
POLAR_APPTAINER_DIRECT_EXEC="${POLAR_APPTAINER_DIRECT_EXEC:-1}"
# Use the friend Slime checkout validated by Tmax for native train-side idle pulse, and the Megatron checkout validated by the SWE runs.
slime_source_dir="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/example/slime"
slime_ref="db7ac8bdbcf1a10b3147e862779d03dc2df31802"
slime_repo="${slime_source_dir}"
slime_dir="${script_dir}/tmp/slime-${slime_ref:0:8}"
megatron_dir="${megatron_dir:-${project_root}/tmp/swegym_deps/Megatron-LM}"

# Smoke shape: retain the production model/TP/DP path while reducing rollout
# work enough to finish two optimizer steps in an interactive allocation.
gpus_per_node=8
num_nodes=2
total_gpus=16
train_num_gpus=8
actor_num_nodes=1
actor_num_gpus_per_node=8
rollout_num_gpus=8
rollout_num_gpus_per_engine=1
tensor_model_parallel_size="${tensor_model_parallel_size:-4}"
qwen_gdn_backend=fla
attention_backend=flash
qkv_format=thd
use_dynamic_batch_size=1
use_sequence_parallel=1
micro_batch_size=1
global_batch_size="${global_batch_size:-8}"
load_debug_rollout_data=""
load_debug_rollout_data_subsample=""
rollout_batch_size="${rollout_batch_size:-4}"
n_samples_per_prompt="${n_samples_per_prompt:-2}"
num_epoch=2
# Slime saves checkpoint iteration 0 after rollout 0.  The exclusive boundary
# of 2 therefore exercises two complete rollout/train/save cycles and should
# produce iter_0000001.
target_num_rollout="${target_num_rollout:-2}"
num_rollout="${num_rollout:-}"
start_rollout_id="${start_rollout_id:-}"
smoke_rows="${smoke_rows:-0}"
# Count this from the end of model/Ray initialization and leave enough time for
# both smoke steps plus synchronous checkpoint cleanup.
exit_duration_minutes="${exit_duration_minutes:-210}"
train_idle_pulse_after_seconds=1680
train_idle_pulse_duration_seconds=120
train_idle_pulse_matrix_size=2048

# Compact PI history before the 60k inference limit, then prefix-merge within
# each segment. TP=4/DP=2 is the four-node smoke-test validated topology.
max_tokens_per_gpu="${max_tokens_per_gpu:-60000}"
log_probs_chunk_size="${log_probs_chunk_size:-64}"
rollout_max_response_len="${rollout_max_response_len:-16000}"
rollout_max_prompt_len="${rollout_max_prompt_len:-32000}"
sglang_context_length="${sglang_context_length:-60000}"
sglang_mem_fraction_static="${sglang_mem_fraction_static:-0.7}"
distributed_timeout_minutes=180
save_interval=1

# Long tool trajectories can overflow the exponential ratios in low_var_kl and
# built-in TIS even with a long token cap. Use the bounded k2 form and a lower LR.
train_lr="${train_lr:-5e-7}"
clip_grad=0.5
kl_loss_coef=0.001
kl_loss_type=k2
use_tis=0
eps_clip=0.2
eps_clip_high=0.28
eps_clip_c=10.0

# PI/Polar settings.
agent_harness="${agent_harness:-pi}"
agent_label="${agent_label:-multi_harness}"
harness_pool="${harness_pool:-pi,codex,claude_code,qwen_code}"
harness_seed="${harness_seed:-87}"
harness_sampling_strategy="${harness_sampling_strategy:-uniform}"
hapo_epsilon="${hapo_epsilon:-0.30}"
hapo_learning_rate="${hapo_learning_rate:-0.10}"
hapo_correct_threshold="${hapo_correct_threshold:-0.50}"
agent_cli_dir="${agent_cli_dir:-}"
anthropic_max_tokens=2048
qwen_code_max_output_tokens=4096
pi_api_type=openai-completions
# Trigger PI compaction early enough to absorb a large tool result without
# overshooting SGLang's hard 60k request limit.
pi_context_window=24000
pi_max_tokens=512
# Compaction creates a clean prefix break; prefix_merging starts a new trace at
# that boundary instead of turning the session into a context-limit failure.
pi_fail_on_context_limit=0
polar_builder_strategy=prefix_merging
polar_max_async_level="${polar_max_async_level:-4}"
polar_min_complete_accept_fraction=0.6
polar_multi_gateway="${polar_multi_gateway:-1}"
polar_gateway_count="${polar_gateway_count:-1}"
polar_gateway_ranks="${polar_gateway_ranks:-}"
polar_gateway_max_init_workers="${polar_gateway_max_init_workers:-24}"
polar_gateway_max_run_workers="${polar_gateway_max_run_workers:-96}"
polar_gateway_max_postrun_workers="${polar_gateway_max_postrun_workers:-64}"
polar_gateway_max_restarts="${polar_gateway_max_restarts:-20}"
# Current Polar correctly rejects memory limits for the Apptainer backend.
polar_runtime_memory_mb=""
polar_task_timeout_seconds=900
polar_request_timeout=900
# Let Slime detect and recreate a rollout SGLang server whose HTTP process dies
# while the Ray actor remains alive. This avoids failing later in update_weights.
use_fault_tolerance="${use_fault_tolerance:-1}"
rollout_health_check_interval="${rollout_health_check_interval:-30}"
rollout_health_check_timeout="${rollout_health_check_timeout:-30}"
rollout_health_check_first_wait="${rollout_health_check_first_wait:-0}"

# Fixed identity shared by checkpoints and W&B. Inherited submit-shell
# variables cannot redirect this task into another run or checkpoint tree.
# Use a new script identity when intentionally starting a different experiment.
run_label="${run_label:-2n16g-train8-rollout8-tp4dp2-4x2-60k-1gw-smoke2-fa4b19-pmerge-k2-multiharness}"
experiment_name="${experiment_name:-webarea-distill_mh_q35_2n_smoke2_tp4dp2_4x2_60k_1gw_20260805}"
run_id="${run_id:-${experiment_name}}"
run_generation="${run_generation:-20260805-swegym-grpo-2n-smoke2-multiharness-v1}"
run_dir="${project_root}/tmp/${run_id}"
run_log_dir="${run_dir}/logs/job-${SLURM_JOB_ID}"
save_dir="${project_root}/tmp/ckpt/${run_id}"
rollout_save_dir="${run_dir}/rollout_results"

# Production invariant: every training allocation must report to the user's
# approved W&B destination.  Keep these fixed instead of allowing inherited
# submit-shell variables to silently redirect or disable tracking.
use_wandb=1
wandb_mode=online
wandb_entity=hwinf_dcm
wandb_project=harnessgen
wandb_group="${run_id}"
wandb_run_id="${run_id}"
wandb_single_owner="${wandb_single_owner:-0}"
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
ray_expected_num_gpus=16
ray_cluster_timeout_seconds=600
# Ray must see this before `ray start`; the inner trainer starts too late to
# affect raylet's host-memory monitor. 0.99 avoids false kills from SGLang/CUDA
# mappings while still leaving the OS cgroup as the final guardrail.
ray_memory_usage_threshold="${ray_memory_usage_threshold:-0.99}"
ray_memory_monitor_refresh_ms="${ray_memory_monitor_refresh_ms:-}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

allocated_nodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}"
[ "${allocated_nodes}" = "2" ] || die "this script requires exactly two allocated nodes"
job_account="${SLURM_JOB_ACCOUNT:-}"
if [ -z "${job_account}" ]; then
    job_account="$(scontrol show job -o "${SLURM_JOB_ID}" | sed -n 's/.* Account=\([^ ]*\).*/\1/p')"
fi
case "${job_account}" in
    nvr_lpr_agentic) ;;
    *) die "account ${job_account:-unknown} is not allowed; use nvr_lpr_agentic" ;;
esac
[ -x "${inner_launcher}" ] || die "missing inner launcher"
[ -f "${train_sqsh}" ] || die "missing training image: ${train_sqsh}"
[ -f "${script_dir}/swegym_train_293.jsonl" ] || die "missing SWE-Gym training data"
[ -f "${slime_source_dir}/train_async.py" ] || die "missing friend Slime: ${slime_source_dir}"
[ "$(git -C "${slime_source_dir}" rev-parse HEAD)" = "${slime_ref}" ] || die "friend Slime HEAD does not match pinned ref ${slime_ref}"
[ -f "${slime_source_dir}/slime/backends/megatron_utils/train_idle_pulse.py" ] || die "friend Slime lacks train idle pulse support"
[ "$((rollout_batch_size * n_samples_per_prompt))" = "${global_batch_size}" ] || die "global batch must equal rollout_batch_size * n_samples_per_prompt"
# The inner launcher creates the converted checkpoint and PI CLI when they are
# absent. This keeps a fresh durable worktree self-contained after cleanup.
case "${target_num_rollout}" in
    ''|*[!0-9]*) die "target_num_rollout must be a positive integer" ;;
esac
case "${exit_duration_minutes}" in
    ''|*[!0-9]*) die "exit_duration_minutes must be a non-negative integer" ;;
esac
[ "${target_num_rollout}" -eq 2 ] || \
    die "target_num_rollout must remain 2 for the isolated two-step smoke run"

# Strict isolated resume. This smoke run may resume only checkpoints produced
# by this exact task, and a checkpoint is accepted only when all eight model
# shards and the matching rollout cursor are present.
run_manifest="${save_dir}/run_generation.txt"
expected_manifest="run_generation=${run_generation}
project_root=${project_root}
slime_source_dir=${slime_source_dir}
slime_ref=${slime_ref}
slime_dir=${slime_dir}
rollout_batch_size=${rollout_batch_size}
n_samples_per_prompt=${n_samples_per_prompt}
global_batch_size=${global_batch_size}
target_num_rollout=${target_num_rollout}
num_epoch=${num_epoch}
algorithm=grpo
train_lr=${train_lr}
clip_grad=${clip_grad}
kl_loss_coef=${kl_loss_coef}
kl_loss_type=${kl_loss_type}
use_tis=${use_tis}
harness_pool=${harness_pool}
harness_seed=${harness_seed}
harness_sampling_strategy=${harness_sampling_strategy}
hapo_epsilon=${hapo_epsilon}
hapo_learning_rate=${hapo_learning_rate}
hapo_correct_threshold=${hapo_correct_threshold}
agent_cli_dir=${agent_cli_dir:-inner-default}
pi_max_output_tokens=${pi_max_tokens}
codex_max_output_tokens=cli-default
claude_code_max_output_tokens=${anthropic_max_tokens}
qwen_code_max_output_tokens=${qwen_code_max_output_tokens}
train_idle_pulse_after_seconds=${train_idle_pulse_after_seconds}
train_idle_pulse_duration_seconds=${train_idle_pulse_duration_seconds}
train_idle_pulse_matrix_size=${train_idle_pulse_matrix_size}
"
mkdir -p "${project_root}/logs/slurm" "${run_log_dir}" "${save_dir}" "${rollout_save_dir}"
if [ -s "${run_manifest}" ]; then
    [ "$(<"${run_manifest}")" = "${expected_manifest%?}" ] || \
        die "run manifest mismatch; refusing checkpoint directory ${save_dir}"
else
    existing_checkpoint=0
    [ ! -e "${save_dir}/latest_checkpointed_iteration.txt" ] || existing_checkpoint=1
    if compgen -G "${save_dir}/iter_*" >/dev/null || \
       compgen -G "${save_dir}/rollout/global_dataset_state_dict_*.pt" >/dev/null; then
        existing_checkpoint=1
    fi
    [ "${existing_checkpoint}" = "0" ] || \
        die "checkpoint directory predates this run; refusing to load ${save_dir}"
    manifest_tmp="${run_manifest}.tmp.${SLURM_JOB_ID}"
    printf "%s" "${expected_manifest%?}" >"${manifest_tmp}"
    mv "${manifest_tmp}" "${run_manifest}"
fi

resume_mode=fresh
completed_rollouts=0
latest_file="${save_dir}/latest_checkpointed_iteration.txt"
if [ -s "${latest_file}" ]; then
    latest_iteration="$(<"${latest_file}")"
    case "${latest_iteration}" in
        ""|*[!0-9]*) die "invalid checkpoint iteration in ${latest_file}: ${latest_iteration}" ;;
    esac
    checkpoint_dir="${save_dir}/iter_$(printf "%07d" "${latest_iteration}")"
    [ -s "${checkpoint_dir}/.metadata" ] || die "checkpoint metadata missing: ${checkpoint_dir}/.metadata"
    [ -s "${checkpoint_dir}/common.pt" ] || die "checkpoint common state missing: ${checkpoint_dir}/common.pt"
    shopt -s nullglob
    checkpoint_shards=("${checkpoint_dir}"/__*_0.distcp)
    shopt -u nullglob
    [ "${#checkpoint_shards[@]}" -eq "${train_num_gpus}" ] || \
        die "checkpoint ${latest_iteration} has ${#checkpoint_shards[@]} shards; expected ${train_num_gpus}"
    for shard in "${checkpoint_shards[@]}"; do
        [ -s "${shard}" ] || die "empty checkpoint shard: ${shard}"
    done
    cursor_file="${save_dir}/rollout/global_dataset_state_dict_${latest_iteration}.pt"
    [ -s "${cursor_file}" ] || die "matching rollout cursor missing: ${cursor_file}"
    completed_rollouts="${latest_iteration}"
    ((completed_rollouts += 1))
    resume_mode="singleton-resume-checkpoint-${latest_iteration}"
fi

if [ "${completed_rollouts}" -ge "${target_num_rollout}" ]; then
    printf "Training already complete: %s/%s rollouts in %s\n" \
        "${completed_rollouts}" "${target_num_rollout}" "${save_dir}"
    exit 0
fi
[ -z "${start_rollout_id}" ] || die "start_rollout_id is managed by checkpoint state; do not override it"
start_rollout_id="${completed_rollouts}"
num_rollout="${target_num_rollout}"

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

mkdir -p "${project_root}/logs/slurm" "${run_log_dir}" "${save_dir}" "${rollout_save_dir}"
stop_file="${run_dir}/ray_workers.stop"
worker_script="${run_dir}/ray_cluster_rank.sh"
rm -f "${stop_file}"

mapfile -t slurm_nodes < <(scontrol show hostnames "${SLURM_NODELIST}")
[ "${#slurm_nodes[@]}" -eq 2 ] || die "expected two Slurm hosts"
head_node="${slurm_nodes[0]}"
slime_train_node=""
slime_train_rank=""
slime_train_ip=""
slurm_node_records=()
for idx in "${!slurm_nodes[@]}"; do
    node="${slurm_nodes[$idx]}"
    node_ip="$(srun --mpi=none --overlap --exact -N1 -n1 -w "${node}" \
        --cpus-per-task=1 --mem=64M --gres=none hostname -I | awk '{print $1}')"
    [ -n "${node_ip}" ] || die "failed to resolve node IP for ${node}"
    slurm_node_records+=("${idx}:${node}:${node_ip}")
    if [ "${idx}" = "0" ]; then
        ray_head_ip="${node_ip}"
    fi
done
[ -n "${ray_head_ip:-}" ] || die "failed to resolve Ray head IP"
sglang_router_host="${ray_head_ip}"
if [ -z "${polar_gateway_hosts:-}" ]; then
    if [ "${polar_multi_gateway}" = "1" ] && [ "${polar_gateway_count}" -lt "${num_nodes}" ]; then
        mapfile -t gateway_selection < <(python3 - "${polar_gateway_count}" "${slurm_node_records[@]}" <<'PYSEL'
import ipaddress
import sys
count = int(sys.argv[1])
records = []
for raw in sys.argv[2:]:
    rank, node, ip = raw.split(":", 2)
    records.append((ipaddress.ip_address(ip), int(rank), node, ip))
records.sort(key=lambda item: (item[0], item[1]))
train = records[0]
gateways = records[1:1 + count]
if len(gateways) != count:
    raise SystemExit(f"failed to select {count} non-train gateway host(s)")
print(f"TRAIN_NODE={train[2]}")
print(f"TRAIN_RANK={train[1]}")
print(f"TRAIN_IP={train[3]}")
print("GATEWAY_HOSTS=" + ",".join(item[2] for item in gateways))
print("GATEWAY_RANKS=" + ",".join(str(item[1]) for item in gateways))
PYSEL
)
        for item in "${gateway_selection[@]}"; do
            case "${item}" in
                TRAIN_NODE=*) slime_train_node="${item#TRAIN_NODE=}" ;;
                TRAIN_RANK=*) slime_train_rank="${item#TRAIN_RANK=}" ;;
                TRAIN_IP=*) slime_train_ip="${item#TRAIN_IP=}" ;;
                GATEWAY_HOSTS=*) polar_gateway_hosts="${item#GATEWAY_HOSTS=}" ;;
                GATEWAY_RANKS=*) polar_gateway_ranks="${item#GATEWAY_RANKS=}" ;;
            esac
        done
    else
        gateway_nodes=("${slurm_nodes[@]:0:${polar_gateway_count}}")
        [ "${#gateway_nodes[@]}" -eq "${polar_gateway_count}" ] || die "failed to select ${polar_gateway_count} gateway host(s)"
        polar_gateway_hosts="$(IFS=,; printf '%s' "${gateway_nodes[*]}")"
        if [ -z "${polar_gateway_ranks}" ]; then
            polar_gateway_ranks="$(seq -s, 0 $((polar_gateway_count - 1)))"
        fi
    fi
elif [ -z "${polar_gateway_ranks}" ]; then
    polar_gateway_ranks="$(seq -s, 0 $((polar_gateway_count - 1)))"
fi

# Lower-case variables are the inner launcher's public settings.
export project_root script_dir inner_launcher train_sqsh container_mounts slime_source_dir slime_repo slime_ref slime_dir megatron_dir
export POLAR_APPTAINER_DIRECT_EXEC
export gpus_per_node num_nodes total_gpus train_num_gpus actor_num_nodes actor_num_gpus_per_node
export rollout_num_gpus rollout_num_gpus_per_engine tensor_model_parallel_size qwen_gdn_backend
export attention_backend qkv_format use_dynamic_batch_size use_sequence_parallel micro_batch_size global_batch_size
export load_debug_rollout_data load_debug_rollout_data_subsample
export rollout_batch_size n_samples_per_prompt num_epoch target_num_rollout num_rollout start_rollout_id smoke_rows
export exit_duration_minutes train_idle_pulse_after_seconds train_idle_pulse_duration_seconds train_idle_pulse_matrix_size
export max_tokens_per_gpu log_probs_chunk_size rollout_max_response_len rollout_max_prompt_len
export sglang_context_length sglang_mem_fraction_static distributed_timeout_minutes save_interval
export train_lr clip_grad kl_loss_coef kl_loss_type use_tis eps_clip eps_clip_high eps_clip_c
export agent_harness agent_label harness_pool harness_seed harness_sampling_strategy
export hapo_epsilon hapo_learning_rate hapo_correct_threshold agent_cli_dir
export anthropic_max_tokens qwen_code_max_output_tokens pi_api_type pi_context_window pi_max_tokens pi_fail_on_context_limit
export polar_builder_strategy polar_max_async_level polar_min_complete_accept_fraction
export polar_multi_gateway polar_gateway_count polar_gateway_hosts polar_gateway_ranks slime_train_rank
export polar_gateway_max_init_workers polar_gateway_max_run_workers polar_gateway_max_postrun_workers
export polar_gateway_max_restarts
export polar_runtime_memory_mb polar_task_timeout_seconds polar_request_timeout
export use_fault_tolerance rollout_health_check_interval rollout_health_check_timeout rollout_health_check_first_wait
export rollout_port gateway_port
export run_id run_label run_generation run_dir run_log_dir save_dir rollout_save_dir
export use_wandb wandb_mode wandb_entity wandb_project wandb_group wandb_run_id wandb_random_suffix wandb_api_key wandb_single_owner
export dry_run ray_port ray_dashboard_port ray_num_cpus ray_expected_num_gpus ray_cluster_timeout_seconds
export ray_memory_usage_threshold ray_memory_monitor_refresh_ms
export ray_head_ip sglang_router_host stop_file

cat >"${worker_script}" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

rank="${SLURM_PROCID}"
node="$(hostname)"
node_ip="$(hostname -I | awk '{print $1}')"
rank_is_gateway() {
    [ "${polar_multi_gateway}" = "1" ] || return 1
    local ranks="${polar_gateway_ranks:-}"
    if [ -z "${ranks}" ]; then
        ranks="$(seq -s, 0 $((polar_gateway_count - 1)))"
    fi
    case ",${ranks}," in
        *,"${rank}",*) return 0 ;;
        *) return 1 ;;
    esac
}
ray_log_monitor_args=(--include-log-monitor=false)
if [ -n "${slime_train_rank:-}" ] && [ "${rank}" = "${slime_train_rank}" ]; then
    ray_log_monitor_args=()
fi
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
export RAY_MEMORY_USAGE_THRESHOLD="${ray_memory_usage_threshold:-0.99}"
export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}"
if [ -n "${ray_memory_monitor_refresh_ms:-}" ]; then
    export RAY_memory_monitor_refresh_ms="${ray_memory_monitor_refresh_ms}"
fi
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
patch_container_runtime_only=1 bash "${inner_launcher}" \
    >"${run_log_dir}/container-patch-rank-${rank}.log" 2>&1

ray stop --force >/dev/null 2>&1 || true
monitor_pid=""
mem_monitor_pid=""
gateway_pid=""
preserve_ray_logs() {
    local source_dir="${ray_tmpdir}/session_latest/logs"
    local target_dir="${run_log_dir}/ray-rank-${rank}"
    [ -d "${source_dir}" ] || return 0
    mkdir -p "${target_dir}"
    find -L "${source_dir}" -maxdepth 1 -type f \
        \( -name 'gcs_server.*' -o -name 'raylet.*' \
           -o -name 'ray_process_exit.log' -o -name 'dashboard*.log' \
          -o -name 'worker*.out' -o -name 'worker*.err' \
          -o -name 'python-core-worker*.log' -o -name 'runtime_env*.log' \
          -o -name 'log_monitor.*' \) \
        -exec cp -f {} "${target_dir}/" \; 2>/dev/null || true
}
cleanup() {
    [ -z "${monitor_pid}" ] || kill "${monitor_pid}" 2>/dev/null || true
    [ -z "${mem_monitor_pid}" ] || kill "${mem_monitor_pid}" 2>/dev/null || true
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

(
    while true; do
        printf 'timestamp=%s node=%s rank=%s\n' "$(date -u +%FT%TZ)" "${node}" "${rank}"
        free -g || true
        ps -eo pid,ppid,rss,vsz,comm,args --sort=-rss | head -16 || true
        sleep 30
    done
) >>"${run_log_dir}/mem-rank-${rank}.log" 2>&1 &
mem_monitor_pid="$!"

if [ "${rank}" = "0" ]; then
    ray start --head --node-ip-address="${ray_head_ip}" --port="${ray_port}" \
        --dashboard-host=0.0.0.0 --dashboard-port="${ray_dashboard_port}" \
        --num-cpus="${ray_num_cpus}" --num-gpus="${gpus_per_node}" \
        --temp-dir="${ray_tmpdir}" --disable-usage-stats "${ray_log_monitor_args[@]}" \
        >"${run_log_dir}/ray-head.log" 2>&1

    finish_workers() {
        touch "${stop_file}"
    }
    trap 'finish_workers; cleanup' EXIT

    ray_use_existing_cluster=1 ray_stop_on_exit=0 \
        bash "${inner_launcher}"
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
        --temp-dir="${ray_tmpdir}" --disable-usage-stats "${ray_log_monitor_args[@]}" --block \
        >"${run_log_dir}/ray-worker-${rank}.log" 2>&1 &
    ray_pid="$!"
    gateway_restarts=0
    start_gateway_sidecar_bg() {
        printf '[%s] starting gateway sidecar rank=%s restart=%s\n' \
            "$(date -u +%FT%TZ)" "${rank}" "${gateway_restarts}" \
            >>"${run_log_dir}/gateway-sidecar-rank-${rank}.supervisor.log"
        RAY_NODE_RANK="${rank}" pi_multigw_sidecar=1 ray_use_existing_cluster=1 ray_stop_on_exit=0 \
            bash "${inner_launcher}" >>"${run_log_dir}/gateway-sidecar-rank-${rank}.driver.log" 2>&1 &
        gateway_pid="$!"
    }
    if rank_is_gateway; then
        start_gateway_sidecar_bg
    fi
    while [ ! -f "${stop_file}" ]; do
        kill -0 "${ray_pid}" 2>/dev/null || { wait "${ray_pid}"; exit $?; }
        if [ -n "${gateway_pid}" ] && ! kill -0 "${gateway_pid}" 2>/dev/null; then
            set +e
            wait "${gateway_pid}"
            gateway_rc="$?"
            set -e
            printf '[%s] gateway sidecar rank=%s exited rc=%s restart=%s\n' \
                "$(date -u +%FT%TZ)" "${rank}" "${gateway_rc}" "${gateway_restarts}" \
                >>"${run_log_dir}/gateway-sidecar-rank-${rank}.supervisor.log"
            gateway_restarts=$((gateway_restarts + 1))
            if [ "${gateway_restarts}" -gt "${polar_gateway_max_restarts}" ]; then
                printf 'ERROR: gateway sidecar rank=%s exceeded restart limit %s\n' \
                    "${rank}" "${polar_gateway_max_restarts}" >&2
                exit "${gateway_rc}"
            fi
            sleep 5
            start_gateway_sidecar_bg
        fi
        sleep 5
    done
fi
WORKER
chmod +x "${worker_script}"

cat <<SUMMARY
============================================================
webarea multi-harness SWE-Gym GRPO
  run:       ${run_id}
  nodes:     ${slurm_nodes[*]}
  ray head:  ${ray_head_ip}
  train node:${slime_train_node:-slime-placement-default}${slime_train_rank:+ (rank ${slime_train_rank}, ip ${slime_train_ip})}
  account:   ${job_account}
  GPUs:      8 train + 8 rollout
  gateways:  ${polar_gateway_count} (${polar_gateway_hosts}), ranks=${polar_gateway_ranks}, per-gateway workers init/run/post=${polar_gateway_max_init_workers}/${polar_gateway_max_run_workers}/${polar_gateway_max_postrun_workers}, restarts=${polar_gateway_max_restarts}
  CPUs:      ${ray_num_cpus} per node
  Ray mem:   threshold=${ray_memory_usage_threshold}${ray_memory_monitor_refresh_ms:+, refresh_ms=${ray_memory_monitor_refresh_ms}}
  TP/DP:     ${tensor_model_parallel_size}/$((train_num_gpus / tensor_model_parallel_size))
  batch:     ${rollout_batch_size} prompts x ${n_samples_per_prompt} samples = ${global_batch_size} trajectories
  harnesses: ${harness_pool} (seed=${harness_seed}, strategy=${harness_sampling_strategy}, one harness per prompt group)
  HAPO:     epsilon=${hapo_epsilon}, lr=${hapo_learning_rate}, threshold=${hapo_correct_threshold}
  output cap:PI=${pi_max_tokens}, Codex=CLI default, Claude=${anthropic_max_tokens}, Qwen Code=${qwen_code_max_output_tokens}
  idle pulse:after=${train_idle_pulse_after_seconds}s, duration=${train_idle_pulse_duration_seconds}s, matrix=${train_idle_pulse_matrix_size}
  scheduling:${resume_mode}, graceful budget=${exit_duration_minutes} min, singleton job name=${SLURM_JOB_NAME}
  boundary:  ${num_rollout}/${target_num_rollout} (start=${start_rollout_id:-checkpoint})
  stability: lr=${train_lr}, KL=${kl_loss_coef}, clip=${clip_grad}, rollout_ft=${use_fault_tolerance}
  W&B:       ${wandb_entity}/${wandb_project}/${wandb_run_id}
============================================================
SUMMARY

# These workers are independent Bash/Ray processes. Disable PMIx exactly as in
# the Tmax reference so its unused fence cannot cancel a healthy long job step.
srun --mpi=none --overlap --kill-on-bad-exit=1 --nodes="${num_nodes}" --ntasks="${num_nodes}" --ntasks-per-node=1 --gres="gpu:${gpus_per_node}" \
    --cpus-per-task="${ray_num_cpus}" \
    --container-image="${train_sqsh}" \
    --container-mounts="${container_mounts}" \
    --container-workdir="${project_root}" \
    --container-writable --no-container-mount-home \
    bash "${worker_script}"
