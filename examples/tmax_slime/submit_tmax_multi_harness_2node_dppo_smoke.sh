#!/usr/bin/env bash
#SBATCH --job-name=tmax-multiharness-4b-dppo-2n-smoke
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
# Repeated `sbatch` calls with this same job name run serially.  A later job
# starts even if the previous one failed, then resumes the last valid checkpoint.
#SBATCH --dependency=singleton
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --exclude=pool1-[00001-00224,00230,00236,00254],pool0-[00126,04476,04843,04892,04982,05158,05432],pool0-[05800-05984]
#SBATCH --output=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/examples/tmax_slime/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/examples/tmax_slime/logs/slurm/%x-%j.err
#SBATCH --export=ALL

# Two-node multi-harness + Qwen3.5-4B DPPO five-step smoke test.
#
# It uses a smoke-sized 4x4 Polar configuration. Slime places the actor on
# the lowest-IP node and eight rollout engines plus one gateway on the other.
set -euo pipefail

project_root="${project_root:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server}"
script_dir="${project_root}/examples/tmax_slime"
inner_launcher="${script_dir}/run_tmax_apptainer_train_dppo.sh"
train_sqsh="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/docker/polr_swegym_qwen35_torch211_te2161_fa4b19_numpy126_scipy117_tebindcu130_20260707.sqsh"
container_mounts="${container_mounts:-/lustre/fs1:/lustre/fs1,/lustre/fsw:/lustre/fsw}"
# Stock Apptainer 1.5.2 unconditionally setns()es into the instance user
# namespace. Pyxis already placed us in that same namespace, so Linux returns
# EINVAL. Use the locally built binary that skips only this redundant setns;
# mount and PID namespaces are still entered normally.
POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-${project_root}/tmp/apptainer-v1.5.2-pyxis-fix/bin/apptainer}"
POLAR_APPTAINER_DIRECT_EXEC="${POLAR_APPTAINER_DIRECT_EXEC:-0}"
POLAR_APPTAINER_EXEC_MODE="${POLAR_APPTAINER_EXEC_MODE:-instance}"
POLAR_APPTAINER_ISOLATE_PID="${POLAR_APPTAINER_ISOLATE_PID:-1}"
# Friend Slime is allowed; Polar must stay in the current project.
slime_dir="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/example/slime"
megatron_dir="${project_root}/tmp/swegym_deps/Megatron-LM"
reference_recipe=1
polar_project_root="${polar_project_root:-${project_root}}"
hf_checkpoint="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/model/Qwen3.5-4B"
ref_load="${project_root}/tmp/checkpoints/Qwen3.5-4B_torch_dist"

# One training node and one rollout node. The actor uses TP=4/DP=2.
gpus_per_node=8
num_nodes=2
total_gpus=16
train_num_gpus=8
actor_num_nodes=1
actor_num_gpus_per_node=8
rollout_num_gpus=8
rollout_num_gpus_per_engine=1
tensor_model_parallel_size=4
qwen_gdn_backend=fla
attention_backend=flash
qkv_format="${qkv_format:-thd}"
use_dynamic_batch_size=1
use_sequence_parallel=1
micro_batch_size=1
global_batch_size=16
load_debug_rollout_data=""
load_debug_rollout_data_subsample=""
rollout_batch_size=4
n_samples_per_prompt=4
num_epoch=1
# Slime uses an exclusive boundary: rollout IDs 0..4 are five training steps.
target_num_rollout=5
num_rollout="${num_rollout:-}"
start_rollout_id="${start_rollout_id:-}"
smoke_rows="${smoke_rows:-0}"
# The launcher interprets this as a deadline relative to process startup, not
# as time reserved before the Slurm limit. A ten-minute value therefore expires
# during the first multi-harness rollout. Let the explicit five-rollout boundary
# terminate this bounded smoke job instead.
exit_duration_minutes=0

# Compact PI history before the 60k inference limit, then prefix-merge within
# each segment. TP=4/DP=2 is the two-node smoke-test validated topology.
max_tokens_per_gpu=67584
log_probs_chunk_size=256
rollout_max_response_len=16384
rollout_max_prompt_len=32000
sglang_context_length=262144
sglang_mem_fraction_static=0.7
distributed_timeout_minutes=180
save_interval=1

# DPPO uses rollout logprobs without KL loss or TIS.
train_lr=5e-7
clip_grad=0.5
kl_loss_coef=0
kl_loss_type="${kl_loss_type:-k2}"
use_tis=0
eps_clip=0.2
eps_clip_high=0.28
eps_clip_c=10.0
calculate_per_token_loss=1
max_train_rollout_logprob_abs_diff=0.5
polar_fully_async=1
polar_early_stop_grace_sessions=2
polar_max_trajectory_tokens=67584

# Multi-harness/Polar settings. agent_harness remains the backward-compatible
# single-harness template; harness_pool explicitly enables per-prompt sampling.
agent_harness=pi
agent_label=multi_harness
harness_pool=pi,codex,claude_code,qwen_code
harness_seed=0
pi_api_type=openai-completions
# Trigger PI compaction early enough to absorb a large tool result without
# overshooting SGLang's hard 60k request limit.
pi_context_window=24000
pi_max_tokens=512
# Compaction creates a clean prefix break; prefix_merging starts a new trace at
# that boundary instead of turning the session into a context-limit failure.
pi_fail_on_context_limit=0
polar_builder_strategy=prefix_merging
polar_max_async_level=2
polar_max_replacement_groups=32
polar_min_complete_accept_fraction=0.0
polar_multi_gateway=1
polar_gateway_count=1
polar_gateway_ranks=""
polar_gateway_hosts=""
polar_gateway_max_init_workers=8
polar_gateway_max_run_workers=64
polar_gateway_max_postrun_workers=16
polar_gateway_max_restarts=0
# Enforce a hard address-space ceiling inherited by forked/exec'd/daemonized
# agent processes. PI's Undici/OpenAI request path needs much more virtual
# address space than its RSS: 2/4/8 GiB fail in WebAssembly initialization,
# and a semantic streaming-completions probe at 16 GiB never reaches the
# gateway. Both 32 and 64 GiB complete the request. Keep 64 GiB as headroom
# for long tool trajectories while stopping the observed ~1.4 TiB runaway.
polar_runtime_memory_mb=65536
polar_task_timeout_seconds=1200
polar_max_steps=160
polar_request_timeout=3600
polar_task_timeout_from_metadata=0
# Do not mask rollout infrastructure failures during qualification.
use_fault_tolerance=0
rollout_health_check_interval=30
rollout_health_check_timeout=30
rollout_health_check_first_wait=0

# Fixed identity: this task cannot inherit an older experiment name or save path.
run_label="dppo-2n-multiharness-rb4-s4-5step-v4"
experiment_name="tmax_multiharness_4b_dppo_2n_smoke_rb4_s4_5step_v4"
run_id="${experiment_name}"
run_dir="${project_root}/tmp/${run_id}"
run_log_dir="${run_dir}/logs/job-${SLURM_JOB_ID}"
save_dir="${project_root}/tmp/ckpt/${run_id}"
run_generation="20260729-dppo-2n-multiharness-rb4-s4-5step-v4"
rollout_save_dir="${run_dir}/rollout_results"
full_prompt_data="${project_root}/examples/tmax_slime/data/tmax_ready_prefix256.jsonl"
prompt_data="${run_dir}/tmax_train.jsonl"
tmax_dataset_dir="${tmax_dataset_dir:-/lustre/fsw/portfolios/nvr/users/songyangh/bjin_works/agent_world_model/tmax15k/dataset}"
tmax_image_dir="${tmax_image_dir:-/lustre/fsw/portfolios/nvr/users/songyangh/bjin_works/agent_world_model/tmax15k/sif}"

# Production invariant: every training allocation must report to the user's
# approved W&B destination.  Keep these fixed instead of allowing inherited
# submit-shell variables to silently redirect or disable tracking.
use_wandb=0
wandb_mode=disabled
wandb_entity=hwinf_dcm
wandb_project=harnessgen
wandb_group="${run_id}"
wandb_run_id="${run_id}"
wandb_random_suffix=0
wandb_api_key="${wandb_api_key:-${WANDB_API_KEY:-}}"

dry_run=0
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

[ "${SLURM_JOB_NUM_NODES:-0}" = "2" ] || die "this smoke test requires exactly two allocated nodes"
[ "${SLURM_JOB_PARTITION:-}" = "interactive" ] || die "this smoke test must run on the interactive partition"
job_account="${SLURM_JOB_ACCOUNT:-}"
if [ -z "${job_account}" ]; then
    job_account="$(scontrol show job -o "${SLURM_JOB_ID}" | sed -n 's/.* Account=\([^ ]*\).*/\1/p')"
fi
case "${job_account}" in
    nvr_lpr_agentic) ;;
    *) die "account ${job_account:-unknown} is not allowed; use nvr_lpr_agentic" ;;
esac
[ -x "${inner_launcher}" ] || die "missing inner launcher: ${inner_launcher}"
[ -f "${train_sqsh}" ] || die "missing training image: ${train_sqsh}"
[ -s "${full_prompt_data}" ] || die "missing pre-generated TMax prompt pool: ${full_prompt_data}"
[ -f "${slime_dir}/train_async.py" ] || die "missing friend Slime: ${slime_dir}"
[ -f "${polar_project_root}/src/polar/__init__.py" ] || die "missing current-project Polar: ${polar_project_root}/src/polar"
project_root_real="$(readlink -f "${project_root}")"
polar_project_root_real="$(readlink -f "${polar_project_root}")"
friend_slime_real="$(readlink -f "/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/example/slime")"
[ "${polar_project_root_real}" = "${project_root_real}" ] || die "Polar must be the current project: ${project_root_real}; got ${polar_project_root_real}"
[ "$(readlink -f "${slime_dir}")" = "${friend_slime_real}" ] || die "Slime must be ${friend_slime_real}"
[ "${POLAR_APPTAINER_EXEC_MODE}" = "instance" ] || die "POLAR_APPTAINER_EXEC_MODE must be instance"
[ "${POLAR_APPTAINER_DIRECT_EXEC}" = "0" ] || die "POLAR_APPTAINER_DIRECT_EXEC must be 0"
[ "${POLAR_APPTAINER_ISOLATE_PID}" = "1" ] || die "POLAR_APPTAINER_ISOLATE_PID must be 1"
[ "$((rollout_batch_size * n_samples_per_prompt))" = "${global_batch_size}" ] || die "global batch must equal rollout_batch_size * n_samples_per_prompt"
# The inner launcher creates the converted checkpoint and PI CLI when they are
# absent. This keeps a fresh durable worktree self-contained after cleanup.
case "${target_num_rollout}" in
    ''|*[!0-9]*) die "target_num_rollout must be a positive integer" ;;
esac
case "${exit_duration_minutes}" in
    ''|*[!0-9]*) die "exit_duration_minutes must be a non-negative integer" ;;
esac
[ "${target_num_rollout}" -ge 1 ] || die "target_num_rollout must be a positive integer"
[ "${exit_duration_minutes}" -ge 0 ] || die "exit_duration_minutes must be a non-negative integer"

# This run may resume only checkpoints created by this exact training task.
# A directory from any older experiment is rejected before Ray starts.
run_manifest="${save_dir}/run_generation.txt"
expected_manifest="run_generation=${run_generation}
project_root=${project_root}
slime_dir=${slime_dir}
rollout_batch_size=4
n_samples_per_prompt=4
global_batch_size=16
target_num_rollout=5
harness_pool=pi,codex,claude_code,qwen_code
harness_seed=0
"
mkdir -p "${script_dir}/logs/slurm" "${run_log_dir}" "${save_dir}" "${rollout_save_dir}"
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

printf 'Polar per-session timeout: %ss; max steps: %s\n' \
    "${polar_task_timeout_seconds}" "${polar_max_steps}"

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
    # This metadata-only step must not inherit the batch allocation's
    # 128 CPUs, 8 GPUs, and mem=0/full-node request.  On a valid four-node
    # allocation that inheritance can make step creation fail with
    # "Memory required by task is not available" before training starts.
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
export project_root script_dir inner_launcher train_sqsh container_mounts slime_dir megatron_dir hf_checkpoint ref_load
export reference_recipe polar_project_root
export POLAR_APPTAINER_BIN POLAR_APPTAINER_DIRECT_EXEC POLAR_APPTAINER_EXEC_MODE POLAR_APPTAINER_ISOLATE_PID
export gpus_per_node num_nodes total_gpus train_num_gpus actor_num_nodes actor_num_gpus_per_node
export rollout_num_gpus rollout_num_gpus_per_engine tensor_model_parallel_size qwen_gdn_backend
export attention_backend qkv_format use_dynamic_batch_size use_sequence_parallel micro_batch_size global_batch_size
export load_debug_rollout_data load_debug_rollout_data_subsample
export rollout_batch_size n_samples_per_prompt num_epoch target_num_rollout num_rollout start_rollout_id smoke_rows
export exit_duration_minutes
export max_tokens_per_gpu log_probs_chunk_size rollout_max_response_len rollout_max_prompt_len
export sglang_context_length sglang_mem_fraction_static distributed_timeout_minutes save_interval
export train_lr clip_grad kl_loss_coef kl_loss_type use_tis eps_clip eps_clip_high eps_clip_c
export calculate_per_token_loss max_train_rollout_logprob_abs_diff
export agent_harness agent_label harness_pool harness_seed
export pi_api_type pi_context_window pi_max_tokens pi_fail_on_context_limit
export polar_builder_strategy polar_max_async_level polar_max_replacement_groups polar_min_complete_accept_fraction
export polar_fully_async polar_early_stop_grace_sessions polar_max_trajectory_tokens
export polar_multi_gateway polar_gateway_count polar_gateway_hosts polar_gateway_ranks slime_train_rank
export polar_gateway_max_init_workers polar_gateway_max_run_workers polar_gateway_max_postrun_workers
export polar_gateway_max_restarts
export polar_runtime_memory_mb polar_task_timeout_seconds polar_max_steps polar_request_timeout
export polar_task_timeout_from_metadata
export use_fault_tolerance rollout_health_check_interval rollout_health_check_timeout rollout_health_check_first_wait
export rollout_port gateway_port
export run_id run_label run_dir run_log_dir save_dir rollout_save_dir run_generation
export full_prompt_data prompt_data tmax_dataset_dir tmax_image_dir
export use_wandb wandb_mode wandb_entity wandb_project wandb_group wandb_run_id wandb_random_suffix wandb_api_key
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
export ray_tmpdir="/dev/shm/webarea-pi-${SLURM_JOB_ID}-${rank}/ray"
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
ray_log_stream_pid=""
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
stream_ray_control_logs() {
    local source_dir="${ray_tmpdir}/session_latest/logs"
    local stream_log="${run_log_dir}/ray-control-rank-${rank}.stream.log"
    tail -n 0 -F "${source_dir}/raylet.out" "${source_dir}/raylet.err" "${source_dir}/ray_process_exit.log" >>"${stream_log}" 2>&1
}
record_signal() {
    local signal="${1}"
    printf '[%s] rank=%s node=%s received=%s\n' "$(date -u +%FT%TZ)" "${rank}" "${node}" "${signal}" >>"${run_log_dir}/rank-signals.log"
}
cleanup() {
    [ -z "${monitor_pid}" ] || kill "${monitor_pid}" 2>/dev/null || true
    [ -z "${mem_monitor_pid}" ] || kill "${mem_monitor_pid}" 2>/dev/null || true
    [ -z "${ray_log_stream_pid}" ] || kill "${ray_log_stream_pid}" 2>/dev/null || true
    [ -z "${gateway_pid}" ] || kill "${gateway_pid}" 2>/dev/null || true
    [ -z "${gateway_pid}" ] || wait "${gateway_pid}" 2>/dev/null || true
    preserve_ray_logs
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'record_signal TERM; exit 143' TERM
trap 'record_signal INT; exit 130' INT
trap 'record_signal HUP; exit 129' HUP
trap 'record_signal QUIT; exit 131' QUIT
stream_ray_control_logs &
ray_log_stream_pid="$!"

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
        # NRT periodically removes stale top-level /tmp directories even while
        # the Ray daemons inside them are still alive. Existing workers keep
        # running, but later SGLang actor recovery then fails because Ray can no
        # longer bootstrap a worker from its deleted session directory.
        [ ! -d "${cache_root}" ] || touch "${cache_root}" 2>/dev/null || true
        [ ! -d "${ray_tmpdir}" ] || touch "${ray_tmpdir}" 2>/dev/null || true
        [ ! -e "${ray_tmpdir}/session_latest" ] || \
            touch "${ray_tmpdir}/session_latest" 2>/dev/null || true
        printf 'timestamp=%s node=%s rank=%s\n' "$(date -u +%FT%TZ)" "${node}" "${rank}"
        free -g || true
        printf '%s\n' 'cgroup memory.events:'
        cat /sys/fs/cgroup/memory.events 2>/dev/null || true
        ps -eo pid,ppid,rss,vsz,comm,args --sort=-rss | head -16 || true
        sleep 10
    done
) >>"${run_log_dir}/mem-rank-${rank}.log" 2>&1 &
mem_monitor_pid="$!"

if [ "${rank}" = "0" ]; then
    (
        cd "${slime_dir}"
        /opt/polr_venv/bin/python -m pytest -q \
            tests/test_train_async_checkpoint_order.py \
            tests/test_train_metric_commit.py \
            tests/test_update_weight_timing.py
    ) >"${run_log_dir}/slime-checkpoint-order-tests.log" 2>&1

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
        local gateway_tmpdir=""
        if [ "${POLAR_PRESERVE_FAILED_SESSIONS:-0}" = "1" ]; then
            # tempfile.mkdtemp() in GatewayNodeManager honors TMPDIR. Keep
            # diagnostic session artifacts on shared Lustre so Slurm's node
            # epilog cannot erase them before they are inspected.
            gateway_tmpdir="${run_log_dir}/preserved-sessions-rank-${rank}"
            mkdir -p "${gateway_tmpdir}"
        fi
        printf '[%s] starting gateway sidecar rank=%s restart=%s\n' \
            "$(date -u +%FT%TZ)" "${rank}" "${gateway_restarts}" \
            >>"${run_log_dir}/gateway-sidecar-rank-${rank}.supervisor.log"
        TMPDIR="${gateway_tmpdir:-/tmp}" RAY_NODE_RANK="${rank}" \
            pi_multigw_sidecar=1 ray_use_existing_cluster=1 ray_stop_on_exit=0 \
            bash "${inner_launcher}" >>"${run_log_dir}/gateway-sidecar-rank-${rank}.driver.log" 2>&1 &
        gateway_pid="$!"
    }
    if rank_is_gateway; then
        start_gateway_sidecar_bg
    fi
    while [ ! -f "${stop_file}" ]; do
        if ! kill -0 "${ray_pid}" 2>/dev/null; then
            set +e
            wait "${ray_pid}"
            ray_rc="$?"
            set -e
            printf '[%s] ray worker rank=%s node=%s exited rc=%s\n' \
                "$(date -u +%FT%TZ)" "${rank}" "${node}" "${ray_rc}" \
                >>"${run_log_dir}/ray-worker-${rank}.exit.log"
            exit "${ray_rc}"
        fi
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
TMax multi-harness 4B DPPO - two-node five-step smoke
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
  harnesses: ${harness_pool} (seed=${harness_seed}, one harness per prompt)
  scheduling:${resume_mode}, graceful budget=${exit_duration_minutes} min, singleton job name=${SLURM_JOB_NAME}
  boundary:  ${num_rollout}/${target_num_rollout} (start=${start_rollout_id:-checkpoint})
  stability: lr=${train_lr}, clip=${clip_grad}, per_token=${calculate_per_token_loss}, logprob_diff=${max_train_rollout_logprob_abs_diff}, rollout_ft=${use_fault_tolerance}
  limits:    async=${polar_max_async_level}, max_steps=${polar_max_steps}, session=${polar_task_timeout_seconds}s, request=${polar_request_timeout}s
  W&B:       ${wandb_entity}/${wandb_project}/${wandb_run_id}
============================================================
SUMMARY

batch_monitor_log="${run_log_dir}/batch-step-monitor.log"
set +e
# These are independent Bash/Ray workers, not MPI ranks. Leaving Slurm's
# default pmix_v4 plugin enabled causes its unused fence to time out after
# 180 minutes and slurmstepd to cancel an otherwise healthy job step.
srun --mpi=none --overlap --kill-on-bad-exit=1 --nodes="${num_nodes}" --ntasks="${num_nodes}" --ntasks-per-node=1 --gres="gpu:${gpus_per_node}" \
        --cpus-per-task="${ray_num_cpus}" \
        --container-image="${train_sqsh}" \
        --container-mounts="${container_mounts}" \
        --container-workdir="${project_root}" \
        --container-writable --no-container-mount-home \
        bash "${worker_script}" &
srun_pid="$!"
printf '[%s] srun started pid=%s\n' \
    "$(date -u +%FT%TZ)" "${srun_pid}" >>"${batch_monitor_log}"

(
    while true; do
        printf 'timestamp=%s host=%s monitor_pid=%s batch_pid=%s srun_pid=%s\n' \
            "$(date -u +%FT%TZ)" "$(hostname)" "${BASHPID}" "$$" "${srun_pid}"
        printf '%s\n' 'self cgroup:'
        cat /proc/self/cgroup 2>/dev/null || true
        cgroup_rel="$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup 2>/dev/null)"
        if [ -n "${cgroup_rel}" ]; then
            for metric in memory.events memory.current memory.peak pids.events pids.current; do
                printf 'cgroup %s: ' "${metric}"
                cat "/sys/fs/cgroup${cgroup_rel}/${metric}" 2>/dev/null || true
            done
        fi
        ps -o pid,ppid,state,rss,vsz,etimes,comm,args --forest \
            -p "$$,${srun_pid}" --ppid "$$,${srun_pid}" 2>/dev/null || true
        sleep 10
    done
) >>"${batch_monitor_log}" 2>&1 &
batch_monitor_pid="$!"

wait "${srun_pid}"
srun_rc="$?"
set -e
kill "${batch_monitor_pid}" 2>/dev/null || true
wait "${batch_monitor_pid}" 2>/dev/null || true
printf '[%s] main srun exited rc=%s\n' "$(date -u +%FT%TZ)" "${srun_rc}" >>"${batch_monitor_log}"
exit "${srun_rc}"
