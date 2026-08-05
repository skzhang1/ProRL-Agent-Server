#!/usr/bin/env bash
# Container-side launcher for SWE-Gym multi-harness + Slime GRPO.
# This script is intentionally current-repo self-contained: Slime, Megatron,
# PI CLI assets, runtime SIF links, and converted checkpoints live under this
# project's tmp/ directory rather than the old polar_3 checkout.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
project_root="${project_root:-$(cd -- "${script_dir}/../.." && pwd)}"
cd "${project_root}"

log() {
    printf '[%s] %s\n' "$(date +'%F %T')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "${project_root}" "$1" ;;
    esac
}

is_path_like() {
    case "$1" in
        /*|./*|../*|~*) return 0 ;;
        *) return 1 ;;
    esac
}

positive_int() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        0) return 1 ;;
        *) return 0 ;;
    esac
}

detect_host_ip() {
    "${python_bin}" - <<'PY'
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("8.8.8.8", 80))
    print(sock.getsockname()[0])
    sock.close()
except Exception:
    try:
        print(socket.gethostbyname(socket.gethostname()))
    except Exception:
        print("127.0.0.1")
PY
}

detect_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '
    else
        printf '0\n'
    fi
}

clone_checkout() {
    local name="$1"
    local repo="$2"
    local ref="$3"
    local dest="$4"
    if [ -d "${dest}/.git" ]; then
        log "${name} checkout exists: ${dest}"
        return
    fi
    if [ -e "${dest}" ]; then
        die "${name} path exists but is not a git checkout: ${dest}"
    fi
    mkdir -p "$(dirname "${dest}")"
    log "Cloning ${name} ${ref} -> ${dest}"
    case "${ref}" in
        ????????????????????????????????????????)
            git clone --no-checkout "${repo}" "${dest}"
            git -C "${dest}" checkout --detach "${ref}"
            ;;
        *)
            git clone --branch "${ref}" --depth 1 "${repo}" "${dest}"
            ;;
    esac
}

checkpoint_ready() {
    [ -f "$1/latest_checkpointed_iteration.txt" ]
}

runtime_ld_library_path() {
    local py_site
    py_site="$(${python_bin} - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
    printf '%s' "/usr/local/cuda-13.0/compat:${py_site}/torch/lib:${py_site}/nvidia/cuda_runtime/lib:${py_site}/nvidia/cuda_nvrtc/lib:${py_site}/nvidia/nvjitlink/lib:${py_site}/nvidia/cublas/lib:${py_site}/nvidia/cudnn/lib:${py_site}/nvidia/nccl/lib:${py_site}/nvidia/cusparse/lib:${py_site}/nvidia/cusolver/lib:${py_site}/nvidia/cufft/lib:${py_site}/nvidia/curand/lib:${LD_LIBRARY_PATH:-}"
}

if [ -x /opt/polr_venv/bin/python ]; then
    python_bin="${python_bin:-/opt/polr_venv/bin/python}"
elif [ -x "${project_root}/.venv/bin/python" ]; then
    python_bin="${python_bin:-${project_root}/.venv/bin/python}"
else
    python_bin="${python_bin:-$(command -v python3 || command -v python)}"
fi
[ -x "${python_bin}" ] || die "python not executable: ${python_bin}"
python_bin_dir="$(cd -- "$(dirname -- "${python_bin}")" &>/dev/null && pwd)"
export PATH="${python_bin_dir}:/opt/polr_venv/bin:/usr/local/cuda/bin:${PATH}"
if [ -d /opt/polr_venv ]; then
    export VIRTUAL_ENV="/opt/polr_venv"
fi
export PYTHONNOUSERSITE=1

# User-facing defaults. Use lower-case names for script settings; uppercase is
# exported only for tools that expect it.
run_id="${run_id:-webarea_swegym_pi_qwen35_4b_full293_$(date -u +%Y%m%dT%H%M%SZ)}"
run_label="${run_label:-full293}"
dry_run="${dry_run:-0}"
smoke_rows="${smoke_rows:-0}"

model_name="${model_name:-Qwen/Qwen3.5-4B}"
agent_harness="${agent_harness:-pi}"
harness_pool="${harness_pool:-}"
harness_seed="${harness_seed:-0}"
agent_label="${agent_label:-${agent_harness}}"
pi_model_name="${pi_model_name:-openai/${model_name}}"
codex_model_name="${codex_model_name:-${model_name}}"
codex_version="${codex_version:-}"
codex_reasoning_effort="${codex_reasoning_effort:-}"
codex_reasoning_summary="${codex_reasoning_summary:-}"
claude_model_name="${claude_model_name:-${model_name}}"
qwen_code_model_name="${qwen_code_model_name:-${model_name}}"
qwen_code_max_output_tokens="${qwen_code_max_output_tokens:-}"
claude_max_turns="${claude_max_turns:-}"
claude_max_thinking_tokens="${claude_max_thinking_tokens:-}"
anthropic_max_tokens="${anthropic_max_tokens:-}"
if [ -z "${anthropic_max_tokens}" ]; then
    case ",${harness_pool:-${agent_harness}}," in
        *,claude_code,*) anthropic_max_tokens=2048 ;;
    esac
fi
hf_checkpoint="${hf_checkpoint:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/model/Qwen3.5-4B}"
model_args_file="$(abs_path "${model_args_file:-${MODEL_ARGS_FILE:-${script_dir}/model_args.sh}}")"

train_sqsh="${train_sqsh:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/docker/polr_swegym_qwen35_torch211_te2161_fa4b19_numpy126_scipy117_tebindcu130_20260707.sqsh}"
dep_root="${dep_root:-${project_root}/tmp/swegym_deps}"
slime_dir="$(abs_path "${slime_dir:-${dep_root}/slime}")"
megatron_dir="$(abs_path "${megatron_dir:-${dep_root}/Megatron-LM}")"
slime_repo="${slime_repo:-https://github.com/THUDM/slime.git}"
slime_ref="${slime_ref:-v0.3.0}"
megatron_repo="${megatron_repo:-https://github.com/NVIDIA/Megatron-LM.git}"
megatron_ref="${megatron_ref:-26.04-alpha.rc1}"

ref_load="$(abs_path "${ref_load:-${project_root}/tmp/checkpoints/Qwen3.5-4B_torch_dist}")"
run_dir="$(abs_path "${run_dir:-${project_root}/tmp/${run_id}}")"
run_log_dir="$(abs_path "${run_log_dir:-${run_dir}/logs}")"
rollout_save_dir="$(abs_path "${rollout_save_dir:-${run_dir}/rollout_results}")"
load_debug_rollout_data="${load_debug_rollout_data:-}"
load_debug_rollout_data_subsample="${load_debug_rollout_data_subsample:-}"
save_dir="$(abs_path "${save_dir:-${project_root}/tmp/ckpt/${run_id}}")"
full_prompt_data="$(abs_path "${full_prompt_data:-${script_dir}/swegym_train_293.jsonl}")"
prompt_data="$(abs_path "${prompt_data:-${run_dir}/swegym_train_${smoke_rows}.jsonl}")"

shared_sif_dir="${shared_sif_dir:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/singularity_images_v3}"
apptainer_image_dir="$(abs_path "${apptainer_image_dir:-${project_root}/tmp/swegym_apptainer_images}")"
agent_cli_dir="$(abs_path "${agent_cli_dir:-${project_root}/tmp/swegym_agent_cli/opt_node}")"
force_agent_cli="${force_agent_cli:-0}"
prepare_missing_sifs="${prepare_missing_sifs:-0}"
convert_weights="${convert_weights:-auto}"

hf_cache_root="${hf_cache_root:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HG_Cache}"
hf_home="${hf_home:-${hf_cache_root}}"
huggingface_hub_cache="${huggingface_hub_cache:-${hf_cache_root}/hub}"
hf_hub_cache="${hf_hub_cache:-${huggingface_hub_cache}}"
transformers_cache="${transformers_cache:-${hf_cache_root}/transformers}"
hf_datasets_cache="${hf_datasets_cache:-${hf_cache_root}/datasets}"
hf_modules_cache="${hf_modules_cache:-${hf_cache_root}/modules}"
sentence_transformers_home="${sentence_transformers_home:-${hf_cache_root}/sentence_transformers}"

job_cache_root="${job_cache_root:-/tmp/webarea-swegym-${agent_label}-${SLURM_JOB_ID:-manual}-${run_id}}"
apptainer_cachedir="${apptainer_cachedir:-${job_cache_root}/apptainer-cache}"
apptainer_tmpdir="${apptainer_tmpdir:-${job_cache_root}/apptainer-tmp}"
apptainer_workdir="${apptainer_workdir:-${job_cache_root}/apptainer-work}"
triton_cache_dir="${triton_cache_dir:-${job_cache_root}/triton-cache}"
triton_home="${triton_home:-${job_cache_root}/triton-home}"
torchinductor_cache_dir="${torchinductor_cache_dir:-${job_cache_root}/torchinductor}"
torch_extensions_dir="${torch_extensions_dir:-${job_cache_root}/torch-extensions}"
xdg_cache_home="${xdg_cache_home:-${job_cache_root}/xdg-cache}"
xdg_config_home="${xdg_config_home:-${job_cache_root}/xdg-config}"
xdg_runtime_dir="${xdg_runtime_dir:-${job_cache_root}/xdg-runtime}"
cuda_cache_path="${cuda_cache_path:-${job_cache_root}/cuda-cache}"
numba_cache_dir="${numba_cache_dir:-${job_cache_root}/numba}"
flashinfer_workspace_dir="${flashinfer_workspace_dir:-${job_cache_root}/flashinfer-cache}"
ray_tmpdir="${ray_tmpdir:-/tmp/webarea-ray-${SLURM_JOB_ID:-manual}}"

# Two-node full-training defaults: one 8-GPU node for Megatron and one
# 8-GPU node for SGLang rollout. The Slurm launcher starts the second Ray node.
gpus_per_node="${gpus_per_node:-$(detect_gpu_count)}"
[ "${gpus_per_node}" -gt 0 ] || gpus_per_node=8
num_nodes="${num_nodes:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-2}}}"
total_gpus="${total_gpus:-$((gpus_per_node * num_nodes))}"
train_num_gpus="${train_num_gpus:-8}"
actor_num_nodes="${actor_num_nodes:-1}"
actor_num_gpus_per_node="${actor_num_gpus_per_node:-${train_num_gpus}}"
rollout_num_gpus="${rollout_num_gpus:-$((total_gpus - train_num_gpus))}"
rollout_num_gpus_per_engine="${rollout_num_gpus_per_engine:-1}"
tensor_model_parallel_size="${tensor_model_parallel_size:-2}"
qkv_format="${qkv_format:-thd}"
use_dynamic_batch_size="${use_dynamic_batch_size:-1}"
use_sequence_parallel="${use_sequence_parallel:-1}"
micro_batch_size="${micro_batch_size:-1}"
ray_num_cpus="${ray_num_cpus:-128}"
ray_head_num_gpus="${ray_head_num_gpus:-${gpus_per_node}}"
ray_expected_num_gpus="${ray_expected_num_gpus:-${total_gpus}}"
ray_cluster_timeout_seconds="${ray_cluster_timeout_seconds:-300}"

if [ "${smoke_rows}" -gt 0 ]; then
    rollout_batch_size="${rollout_batch_size:-1}"
    n_samples_per_prompt="${n_samples_per_prompt:-1}"
    num_rollout="${num_rollout:-1}"
    save_interval="${save_interval:-1}"
    rollout_max_response_len="${rollout_max_response_len:-4096}"
    rollout_max_prompt_len="${rollout_max_prompt_len:-24000}"
    sglang_context_length="${sglang_context_length:-32768}"
    pi_context_window="${pi_context_window:-24000}"
    polar_max_async_level="${polar_max_async_level:-1}"
    max_tokens_per_gpu="${max_tokens_per_gpu:-12000}"
else
    rollout_batch_size="${rollout_batch_size:-4}"
    n_samples_per_prompt="${n_samples_per_prompt:-16}"
    num_rollout="${num_rollout:-}"
    save_interval="${save_interval:-5}"
    rollout_max_response_len="${rollout_max_response_len:-16000}"
    rollout_max_prompt_len="${rollout_max_prompt_len:-32000}"
    sglang_context_length="${sglang_context_length:-50000}"
    pi_context_window="${pi_context_window:-32000}"
    polar_max_async_level="${polar_max_async_level:-2}"
    max_tokens_per_gpu="${max_tokens_per_gpu:-30000}"
fi
global_batch_size="${global_batch_size:-$((rollout_batch_size * n_samples_per_prompt))}"
qwen_code_max_output_tokens="${qwen_code_max_output_tokens:-${rollout_max_response_len}}"
num_epoch="${num_epoch:-1}"
start_rollout_id="${start_rollout_id:-}"
train_idle_pulse_after_seconds="${train_idle_pulse_after_seconds:-1680}"
train_idle_pulse_duration_seconds="${train_idle_pulse_duration_seconds:-120}"
train_idle_pulse_matrix_size="${train_idle_pulse_matrix_size:-2048}"
num_steps_per_rollout="${num_steps_per_rollout:-1}"
distributed_timeout_minutes="${distributed_timeout_minutes:-180}"
sglang_mem_fraction_static="${sglang_mem_fraction_static:-0.8}"
sglang_log_level="${sglang_log_level:-warning}"
qwen_gdn_backend="${qwen_gdn_backend:-flashqla}"
attention_backend="${attention_backend:-flash}"
train_lr="${train_lr:-5e-7}"
clip_grad="${clip_grad:-0.5}"
kl_loss_coef="${kl_loss_coef:-0.001}"
kl_loss_type="${kl_loss_type:-k2}"
use_tis="${use_tis:-0}"
eps_clip="${eps_clip:-0.2}"
eps_clip_high="${eps_clip_high:-0.28}"
eps_clip_c="${eps_clip_c:-10.0}"

polar_builder_strategy="${polar_builder_strategy:-prefix_merging}"
polar_min_complete_accept_fraction="${polar_min_complete_accept_fraction:-0.6}"
polar_multi_gateway="${polar_multi_gateway:-0}"
polar_gateway_count="${polar_gateway_count:-1}"
polar_gateway_hosts="${polar_gateway_hosts:-}"
polar_gateway_ranks="${polar_gateway_ranks:-}"
polar_gateway_max_init_workers="${polar_gateway_max_init_workers:-24}"
polar_gateway_max_run_workers="${polar_gateway_max_run_workers:-192}"
polar_gateway_max_postrun_workers="${polar_gateway_max_postrun_workers:-96}"
polar_runtime_memory_mb="${polar_runtime_memory_mb:-}"
polar_task_timeout_seconds="${polar_task_timeout_seconds:-1200}"
polar_request_timeout="${polar_request_timeout:-1200}"
use_fault_tolerance="${use_fault_tolerance:-0}"
rollout_health_check_interval="${rollout_health_check_interval:-30}"
rollout_health_check_timeout="${rollout_health_check_timeout:-30}"
rollout_health_check_first_wait="${rollout_health_check_first_wait:-0}"
log_probs_chunk_size="${log_probs_chunk_size:-256}"
pi_api_type="${pi_api_type:-openai-completions}"
pi_max_tokens="${pi_max_tokens:-512}"
pi_thinking="${pi_thinking:-}"
pi_fail_on_context_limit="${pi_fail_on_context_limit:-1}"
if [ "${pi_fail_on_context_limit}" = "1" ]; then
    pi_compaction_enabled="${pi_compaction_enabled:-false}"
    pi_retry_enabled="${pi_retry_enabled:-false}"
else
    pi_compaction_enabled="${pi_compaction_enabled:-true}"
    pi_retry_enabled="${pi_retry_enabled:-true}"
fi
pi_retry_max_retries="${pi_retry_max_retries:-0}"
pi_provider_max_retries="${pi_provider_max_retries:-0}"

rollout_port="${rollout_port:-18080}"
gateway_port="${gateway_port:-18100}"
sglang_router_port="${sglang_router_port:-26000}"
slime_sglang_base_port="${slime_sglang_base_port:-34000}"
ray_port="${ray_port:-6379}"
ray_dashboard_port="${ray_dashboard_port:-28265}"
ray_head_ip="${ray_head_ip:-$(detect_host_ip)}"
ray_use_existing_cluster="${ray_use_existing_cluster:-0}"
ray_stop_on_exit="${ray_stop_on_exit:-$([ "${ray_use_existing_cluster}" = "1" ] && printf 0 || printf 1)}"
sglang_router_host="${sglang_router_host:-$(detect_host_ip)}"
sglang_router_base_url="${sglang_router_base_url:-http://${sglang_router_host}:${sglang_router_port}}"

# Mirror the outer-launcher invariant inside the container.  This prevents a
# direct invocation or inherited environment from sending training elsewhere.
use_wandb=1
wandb_mode=online
wandb_entity=hwinf_dcm
wandb_project=harnessgen
wandb_group="${wandb_group:-webarea-swegym-${agent_label}-qwen35-4b-${run_label}}"
wandb_run_id="${wandb_run_id:-${run_id}}"
wandb_random_suffix="${wandb_random_suffix:-0}"
wandb_api_key="${wandb_api_key:-${WANDB_API_KEY:-}}"
wandb_dir="$(abs_path "${wandb_dir:-${project_root}/logs/wandb}")"

if [ "${patch_container_runtime_only:-0}" != "1" ]; then
    if [ "${train_num_gpus}" -le 0 ]; then
        die "train_num_gpus must be positive"
    fi
    if [ -z "${load_debug_rollout_data}" ] && [ "${rollout_num_gpus}" -le 0 ]; then
        die "rollout_num_gpus must be positive unless replaying debug rollout data"
    fi
    if [ "$((train_num_gpus + rollout_num_gpus))" -gt "${total_gpus}" ]; then
        die "train + rollout GPUs exceed total GPUs: train=${train_num_gpus}, rollout=${rollout_num_gpus}, total=${total_gpus}"
    fi
    if [ "$((train_num_gpus % tensor_model_parallel_size))" -ne 0 ]; then
        die "train_num_gpus must be divisible by tensor_model_parallel_size"
    fi
    if [ "$((rollout_batch_size * n_samples_per_prompt % num_steps_per_rollout))" -ne 0 ]; then
        die "rollout_batch_size * n_samples_per_prompt must divide num_steps_per_rollout"
    fi
fi

mkdir -p \
    "${run_dir}" "${run_log_dir}" "${rollout_save_dir}" "${save_dir}" "${wandb_dir}" \
    "${apptainer_image_dir}" "${agent_cli_dir}" "${hf_home}" "${huggingface_hub_cache}" \
    "${hf_hub_cache}" "${transformers_cache}" "${hf_datasets_cache}" \
    "${hf_modules_cache}" "${sentence_transformers_home}" "${job_cache_root}" \
    "${apptainer_cachedir}" "${apptainer_tmpdir}" "${apptainer_workdir}" \
    "${triton_cache_dir}" "${triton_home}" "${torchinductor_cache_dir}" \
    "${torch_extensions_dir}" "${xdg_cache_home}" "${xdg_config_home}" \
    "${xdg_runtime_dir}" "${cuda_cache_path}" "${numba_cache_dir}" \
    "${flashinfer_workspace_dir}" "${ray_tmpdir}"
chmod 700 "${xdg_runtime_dir}" || true
ulimit -n 1048576 >/dev/null 2>&1 || ulimit -n 65536 >/dev/null 2>&1 || true

export HF_HOME="${hf_home}"
export HUGGINGFACE_HUB_CACHE="${huggingface_hub_cache}"
export HF_HUB_CACHE="${hf_hub_cache}"
export TRANSFORMERS_CACHE="${transformers_cache}"
export HF_DATASETS_CACHE="${hf_datasets_cache}"
export HF_MODULES_CACHE="${hf_modules_cache}"
export SENTENCE_TRANSFORMERS_HOME="${sentence_transformers_home}"
export APPTAINER_CACHEDIR="${apptainer_cachedir}"
export APPTAINER_TMPDIR="${apptainer_tmpdir}"
export APPTAINER_WORKDIR="${apptainer_workdir}"
export SINGULARITY_CACHEDIR="${apptainer_cachedir}"
export SINGULARITY_TMPDIR="${apptainer_tmpdir}"
export TRITON_CACHE_DIR="${triton_cache_dir}"
export TRITON_HOME="${triton_home}"
export TORCHINDUCTOR_CACHE_DIR="${torchinductor_cache_dir}"
export TORCH_EXTENSIONS_DIR="${torch_extensions_dir}"
export XDG_CACHE_HOME="${xdg_cache_home}"
export XDG_CONFIG_HOME="${xdg_config_home}"
export XDG_RUNTIME_DIR="${xdg_runtime_dir}"
export CUDA_CACHE_PATH="${cuda_cache_path}"
export NUMBA_CACHE_DIR="${numba_cache_dir}"
export FLASHINFER_WORKSPACE_DIR="${flashinfer_workspace_dir}"
export POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-$(command -v apptainer || echo /usr/bin/apptainer)}"
export POLAR_APPTAINER_DIRECT_EXEC="${POLAR_APPTAINER_DIRECT_EXEC:-1}"
export POLAR_APPTAINER_EXEC_MODE="${POLAR_APPTAINER_EXEC_MODE:-direct}"
# Diagnostic compatibility hook remains available, but real runs must not hide
# even sparse non-finite gradients unless explicitly opted in.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
export SLIME_RESPONSE_ONLY_LOGPROBS="${SLIME_RESPONSE_ONLY_LOGPROBS:-0}"
export SLIME_DEBUG_ONE_PER_GROUP="${SLIME_DEBUG_ONE_PER_GROUP:-0}"
export SLIME_DEBUG_GRAD_HOOKS="${SLIME_DEBUG_GRAD_HOOKS:-0}"
export SLIME_DEBUG_PARAM_GRADS="${SLIME_DEBUG_PARAM_GRADS:-0}"
export SLIME_EXIT_DURATION_MINUTES="${exit_duration_minutes:-0}"
export SLIME_TRAIN_IDLE_PULSE_AFTER_SECONDS="${train_idle_pulse_after_seconds}"
export SLIME_TRAIN_IDLE_PULSE_DURATION_SECONDS="${train_idle_pulse_duration_seconds}"
export SLIME_TRAIN_IDLE_PULSE_MATRIX_SIZE="${train_idle_pulse_matrix_size}"
export RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD:-0.99}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-${RAY_MEMORY_USAGE_THRESHOLD}}"
export WANDB_API_KEY="${wandb_api_key}"
export WANDB_MODE="${wandb_mode}"
export WANDB_PROJECT="${wandb_project}"
export WANDB_ENTITY="${wandb_entity}"
export WANDB_DIR="${wandb_dir}"
fla_override_dir="${fla_override_dir:-}"
flashqla_override_dir="${flashqla_override_dir:-${project_root}/tmp/runtime_overrides/flashqla_code_0_1_1}"
if [ "${attention_backend}" = "flash" ]; then
    flash_attention_version="$(${python_bin} - <<'PY'
from importlib.metadata import version
print(version("flash-attn-4"))
PY
)"
    [ "${flash_attention_version}" = "4.0.0b19" ] || \
        die "expected flash-attn-4 4.0.0b19, found ${flash_attention_version}"
    # Force the independently verified FA4 packed-THD path and forbid silent
    # fallback to the cuDNN path that produced non-finite QKV gradients.
    export NVTE_FLASH_ATTN=1
    export NVTE_FUSED_ATTN=0
fi
if [ "${qwen_gdn_backend}" = "flashqla" ]; then
    [ -d "${flashqla_override_dir}/flash_qla" ] || die "FlashQLA 0.1.1 runtime override missing: ${flashqla_override_dir}"
    export PYTHONPATH="${flashqla_override_dir}:${megatron_dir}:${slime_dir}:${project_root}/src:${PYTHONPATH:-}"
elif [ -n "${fla_override_dir}" ]; then
    [ -d "${fla_override_dir}/fla" ] || die "FLA runtime override missing: ${fla_override_dir}"
    export PYTHONPATH="${fla_override_dir}:${megatron_dir}:${slime_dir}:${project_root}/src:${PYTHONPATH:-}"
else
    # Use the image's FLA 0.4.0 package. This is the exact GDN runtime that
    # completed the prior Qwen3.5 SWE-Gym training through optimizer step 73.
    export PYTHONPATH="${megatron_dir}:${slime_dir}:${project_root}/src:${PYTHONPATH:-}"
fi
export LD_LIBRARY_PATH="$(runtime_ld_library_path)"
[ -z "${fla_override_dir}" ] || export LD_LIBRARY_PATH="${fla_override_dir}/z3/lib:${LD_LIBRARY_PATH}"

fla_runtime_version="$(${python_bin} - <<'PY'
from importlib.metadata import version
import fla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401

print(f"{version('flash-linear-attention')} ({fla.__file__})")
PY
)"
log "Flash Linear Attention: ${fla_runtime_version}"
if [ "${qwen_gdn_backend}" = "flashqla" ]; then
    flashqla_runtime="$(${python_bin} - <<'PY'
import flash_qla
import tilelang
print(f"0.1.1 ({flash_qla.__file__}), TileLang {tilelang.__version__}")
PY
)"
    log "Qwen GDN backend: FlashQLA ${flashqla_runtime}"
fi

ensure_current_checkouts() {
    clone_checkout "Slime" "${slime_repo}" "${slime_ref}" "${slime_dir}"
    [ "$(git -C "${slime_dir}" rev-parse HEAD)" = "${slime_ref}" ] || die "Slime checkout does not match pinned ref ${slime_ref}: ${slime_dir}"
    clone_checkout "Megatron-LM" "${megatron_repo}" "${megatron_ref}" "${megatron_dir}"
    [ -f "${slime_dir}/train_async.py" ] || die "Slime train_async.py missing: ${slime_dir}"
    [ -d "${megatron_dir}/megatron" ] || die "Megatron package missing: ${megatron_dir}"
}

patch_container_runtime() {
    local sglang_version
    sglang_version="$(${python_bin} - <<PY
import sglang
print(sglang.__version__)
PY
)"
    case "${sglang_version}" in
        0.5.10|0.5.13)
            token_patch="${project_root}/scripts/patch/patch_sglang_${sglang_version//./}_token_metadata.sh"
            if [ -f "${token_patch}" ]; then
                bash "${token_patch}"
            else
                log "WARNING: optional SGLang token metadata patch is unavailable: ${token_patch}"
            fi
            ;;
        *) die "unsupported SGLang version for token metadata patch: ${sglang_version}" ;;
    esac
    "${python_bin}" - <<'PY'
from pathlib import Path
try:
    import ray
except Exception:
    raise SystemExit(0)
base = Path(ray.__file__).resolve().parent / "experimental" / "channel"
for name in ("communicator.py", "common.py"):
    path = base / name
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8")
    if "import ray.actor\n" not in text and "import ray\n" in text:
        path.write_text(text.replace("import ray\n", "import ray\nimport ray.actor\n", 1), encoding="utf-8")
PY
}

patch_wandb_compat() {
    "${python_bin}" - "${slime_dir}/slime/utils/wandb_utils.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
legacy_finish_timeout = "            finish_timeout=finish_timeout,\n"
legacy_count = text.count(legacy_finish_timeout)
if legacy_count == 2:
    text = text.replace(legacy_finish_timeout, "")
    path.write_text(text, encoding="utf-8")
elif legacy_count != 0:
    raise SystemExit(
        f"expected zero or two legacy W&B finish_timeout arguments in {path}; "
        f"found {legacy_count}"
    )
worker_finish_timeout = "        finish_timeout=_wandb_finish_timeout_seconds(),\n"
worker_count = text.count(worker_finish_timeout)
if worker_count == 2:
    text = text.replace(worker_finish_timeout, "")
    path.write_text(text, encoding="utf-8")
elif worker_count != 0:
    raise SystemExit(
        f"expected zero or two worker W&B finish_timeout arguments in {path}; "
        f"found {worker_count}"
    )
stable_id_lines = (
    "    if args.wandb_run_id is not None:",
    '        init_kwargs["id"] = args.wandb_run_id',
    '        init_kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")',
)
if not all(line in text for line in stable_id_lines):
    raise SystemExit(f"missing native W&B stable run-id support in {path}")
PY
}

patch_checkpoint_async_strategy_compat() {
    # The converted release checkpoint predates Megatron's async_strategy
    # field.  This Megatron revision otherwise interprets a missing checkpoint
    # field as nvrx, ignoring the current --async-strategy mcore argument.
    # Patch only the dedicated Slime clone and redirect that legacy fallback at
    # runtime; do not modify the shared Megatron checkout or checkpoint files.
    "${python_bin}" - "${slime_dir}/slime/backends/megatron_utils/checkpoint.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "# SLIME_NRT_ASYNC_STRATEGY_COMPAT"
if marker not in text:
    needle = "from megatron.training.global_vars import get_args\n"
    if text.count(needle) != 1:
        raise SystemExit(f"unexpected get_args import in {path}")
    insertion = needle + """

# SLIME_NRT_ASYNC_STRATEGY_COMPAT
from megatron.core.dist_checkpointing.strategies import torch as _torch_dist_strategy

_slime_original_get_async_strategy = _torch_dist_strategy.get_async_strategy


def _slime_get_async_strategy(async_strategy="nvrx", module=None):
    # Old converted checkpoints have no async_strategy in their saved args.
    # Megatron's serialization layer turns that absence into "nvrx". Honor
    # the current job's explicit mcore choice when nvrx is unavailable.
    if async_strategy == "nvrx":
        try:
            configured_strategy = getattr(get_args(), "async_strategy", None)
        except (AssertionError, RuntimeError):
            configured_strategy = None
        if configured_strategy == "mcore":
            async_strategy = "mcore"
    return _slime_original_get_async_strategy(async_strategy, module=module)


_torch_dist_strategy.get_async_strategy = _slime_get_async_strategy
"""
    text = text.replace(needle, insertion, 1)
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")
PY
}

patch_swegym_grpo_compat() {
    # Never patch the shared friend Slime checkout used by Tmax. The dedicated
    # clone under this example inherits its native idle-pulse implementation,
    # then receives only the numerical guards used by the working SWE GRPO run.
    case "${slime_dir}" in
        "${script_dir}"/tmp/*) ;;
        *) die "refusing to patch non-isolated Slime checkout: ${slime_dir}" ;;
    esac
    "${python_bin}" - \
        "${slime_dir}/slime/utils/ppo_utils.py" \
        "${slime_dir}/slime/backends/megatron_utils/loss.py" <<'PY'
from pathlib import Path
import sys

ppo_path = Path(sys.argv[1])
text = ppo_path.read_text(encoding="utf-8")
if "import math\n" not in text:
    text = text.replace("from argparse import Namespace\n", "from argparse import Namespace\n\nimport math\n", 1)
old = "    ratio = (-ppo_kl).exp()\n"
new = (
    "    max_log_ratio = math.log(eps_clip_c) if eps_clip_c is not None else 20.0\n"
    "    ratio = (-ppo_kl).clamp(min=-20.0, max=max_log_ratio).exp()\n"
)
if new in text:
    pass
elif old in text:
    text = text.replace(old, new, 1)
else:
    raise SystemExit(f"unexpected policy-ratio block in {ppo_path}")
ppo_path.write_text(text, encoding="utf-8")

loss_path = Path(sys.argv[2])
text = loss_path.read_text(encoding="utf-8")
old = "        pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)\n"
new = (
    "        pg_loss, pg_clipfrac = compute_policy_loss(\n"
    "            ppo_kl, advantages, args.eps_clip, args.eps_clip_high, args.eps_clip_c\n"
    "        )\n"
)
if new in text:
    pass
elif old in text:
    text = text.replace(old, new, 1)
else:
    raise SystemExit(f"unexpected compute_policy_loss call in {loss_path}")

old = "        with_entropy=True,\n"
new = "        with_entropy=args.entropy_coef != 0.0,\n"
if new in text:
    pass
elif old in text:
    text = text.replace(old, new, 1)
else:
    raise SystemExit(f"unexpected actor entropy request in {loss_path}")

old = """    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss
"""
new = """    # Do not attach the custom entropy autograd path when its coefficient
    # is zero: 0 * entropy is mathematically inert but its backward is not.
    if args.entropy_coef != 0.0:
        entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
        loss = pg_loss - args.entropy_coef * entropy_loss
    else:
        entropy_loss = torch.zeros_like(pg_loss)
        loss = pg_loss
"""
if new in text:
    pass
elif old in text:
    text = text.replace(old, new, 1)
else:
    raise SystemExit(f"unexpected entropy loss block in {loss_path}")
loss_path.write_text(text, encoding="utf-8")
PY
}

prepare_prompt_data_and_sifs() {
    "${python_bin}" - "${full_prompt_data}" "${prompt_data}" "${smoke_rows}" \
        "${apptainer_image_dir}" "${shared_sif_dir}" "${prepare_missing_sifs}" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

full_prompt = Path(sys.argv[1])
prompt_out = Path(sys.argv[2])
smoke_rows = int(sys.argv[3])
image_dir = Path(sys.argv[4])
shared_dir = Path(sys.argv[5])
prepare_missing = sys.argv[6] == "1"
script_dir = full_prompt.parent
sys.path.insert(0, str(script_dir))
from sample_tasks import registry_image_for_instance_id  # noqa: E402

rows = [json.loads(line) for line in full_prompt.read_text().splitlines() if line.strip()]
if smoke_rows > 0:
    rows = rows[:smoke_rows]
if not rows:
    raise SystemExit("no prompt rows selected")

prompt_out.parent.mkdir(parents=True, exist_ok=True)
prompt_out.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n")
image_dir.mkdir(parents=True, exist_ok=True)

missing = []
for row in rows:
    instance_id = str(row["metadata"]["instance_id"])
    target = image_dir / f"{instance_id}.sif"
    if target.exists() or target.is_symlink():
        continue
    image_ref = registry_image_for_instance_id(instance_id)
    shared_name = image_ref.split(":", 1)[0].replace("/", "_") + ".sif"
    source = shared_dir / shared_name
    if source.is_file():
        target.symlink_to(source)
    else:
        missing.append((instance_id, source))

if missing and prepare_missing:
    for instance_id, _ in missing:
        subprocess.run(
            [
                sys.executable,
                str(script_dir / "prepare_apptainer_images.py"),
                "--instance-id", instance_id,
                "--image-dir", str(image_dir),
                "--cache-dir", os.environ.get("APPTAINER_CACHEDIR", "tmp/apptainer_cache"),
                "--tmp-dir", os.environ.get("APPTAINER_TMPDIR", "tmp/apptainer_tmp"),
                "--skip-cli",
            ],
            check=True,
        )
elif missing:
    print("Missing runtime SIF(s):", file=sys.stderr)
    for instance_id, source in missing[:20]:
        print(f"  {instance_id}: expected shared source {source}", file=sys.stderr)
    raise SystemExit("Set prepare_missing_sifs=1 to pull missing runtime images.")

print(f"Prompt data: {prompt_out} ({len(rows)} row(s))")
print(f"Runtime SIF dir: {image_dir}")
PY
}

prepare_agent_cli() {
    if [ "${force_agent_cli}" != "1" ] && [ -x "${agent_cli_dir}/bin/pi" ] && [ -x "${agent_cli_dir}/bin/node" ]; then
        log "Agent CLI exists: ${agent_cli_dir}"
        return
    fi
    log "Preparing PI/agent CLI in current project: ${agent_cli_dir}"
    "${python_bin}" - "${script_dir}" "${agent_cli_dir}" "${force_agent_cli}" <<'PY'
from pathlib import Path
import sys
script_dir = Path(sys.argv[1])
agent_cli_dir = Path(sys.argv[2])
force = sys.argv[3] == "1"
sys.path.insert(0, str(script_dir))
from prepare_apptainer_images import ensure_agent_cli_dir  # noqa: E402
ensure_agent_cli_dir(agent_cli_dir, force=force)
PY
}

ensure_checkpoint() {
    if checkpoint_ready "${ref_load}"; then
        log "Megatron checkpoint exists: ${ref_load}"
        return
    fi
    if [ "${convert_weights}" = "0" ]; then
        die "Megatron checkpoint missing and convert_weights=0: ${ref_load}"
    fi
    if [ "${convert_weights}" != "1" ] && [ "${convert_weights}" != "auto" ]; then
        die "convert_weights must be 0, 1, or auto; got ${convert_weights}"
    fi
    log "Converting HF checkpoint to current project Megatron checkpoint: ${ref_load}"
    HF_CHECKPOINT="${hf_checkpoint}" \
    TORCH_DIST_DIR="${ref_load}" \
    SLIME_DIR="${slime_dir}" \
    MEGATRON_DIR="${megatron_dir}" \
    MODEL_ARGS_FILE="${model_args_file}" \
    PYTHON_BIN="${python_bin}" \
        bash "${script_dir}/convert_weights.sh"
    checkpoint_ready "${ref_load}" || die "conversion finished but checkpoint marker is missing: ${ref_load}"
}

render_runtime_configs() {
    topology_path="${run_dir}/topology.yaml"
    custom_config_path="${run_dir}/polar_config.yaml"
    export topology_path custom_config_path
    export ROLLOUT_PORT="${rollout_port}"
    export GATEWAY_PORT="${gateway_port}"
    export ROLLOUT_SAVE_DIR="${rollout_save_dir}"
    export MODEL_NAME="${model_name}"
    export AGENT_HARNESS="${agent_harness}"
    export HARNESS_POOL="${harness_pool}"
    export HARNESS_SEED="${harness_seed}"
    export CODEX_MODEL_NAME="${codex_model_name}"
    export CODEX_VERSION="${codex_version}"
    export CODEX_REASONING_EFFORT="${codex_reasoning_effort}"
    export CODEX_REASONING_SUMMARY="${codex_reasoning_summary}"
    export CLAUDE_MODEL_NAME="${claude_model_name}"
    export QWEN_CODE_MODEL_NAME="${qwen_code_model_name}"
    export CLAUDE_MAX_TURNS="${claude_max_turns}"
    export CLAUDE_MAX_THINKING_TOKENS="${claude_max_thinking_tokens}"
    export POLAR_ANTHROPIC_MAX_TOKENS="${anthropic_max_tokens}"
    export QWEN_CODE_MAX_OUTPUT_TOKENS="${qwen_code_max_output_tokens}"
    export SGLANG_ROUTER_BASE_URL="${sglang_router_base_url}"
    export AGENT_CLI_DIR="${agent_cli_dir}"
    export APPTAINER_IMAGE_DIR="${apptainer_image_dir}"
    export POLAR_REQUEST_TIMEOUT="${polar_request_timeout}"
    export POLAR_MAX_ASYNC_LEVEL="${polar_max_async_level}"
    export POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${polar_min_complete_accept_fraction}"
    export POLAR_TASK_TIMEOUT_SECONDS="${polar_task_timeout_seconds}"
    export POLAR_RUNTIME_MEMORY_MB="${polar_runtime_memory_mb}"
    export POLAR_MULTI_GATEWAY="${polar_multi_gateway}"
    export POLAR_GATEWAY_COUNT="${polar_gateway_count}"
    export POLAR_GATEWAY_HOSTS="${polar_gateway_hosts}"
    export POLAR_GATEWAY_RANKS="${polar_gateway_ranks}"
    export POLAR_GATEWAY_MAX_INIT_WORKERS="${polar_gateway_max_init_workers}"
    export POLAR_GATEWAY_MAX_RUN_WORKERS="${polar_gateway_max_run_workers}"
    export POLAR_GATEWAY_MAX_POSTRUN_WORKERS="${polar_gateway_max_postrun_workers}"
    export PI_COMPACTION_ENABLED="${pi_compaction_enabled}"
    export PI_RETRY_ENABLED="${pi_retry_enabled}"
    export PI_RETRY_MAX_RETRIES="${pi_retry_max_retries}"
    export PI_PROVIDER_MAX_RETRIES="${pi_provider_max_retries}"
    export PI_MODEL_NAME="${pi_model_name}"
    export PI_API_TYPE="${pi_api_type}"
    export PI_CONTEXT_WINDOW="${pi_context_window}"
    export PI_MAX_TOKENS="${pi_max_tokens}"
    export PI_THINKING="${pi_thinking}"
    export POLAR_BUILDER_STRATEGY="${polar_builder_strategy}"
    export POLAR_BIND_HOST="${polar_bind_host:-0.0.0.0}"
    export POLAR_PUBLIC_HOST="${polar_public_host:-${ray_head_ip}}"
    export POLAR_GATEWAY_PRIMARY_HOST="${polar_gateway_hosts%%,*}"

    "${python_bin}" - "${script_dir}/topology.yaml" "${topology_path}" \
        "${script_dir}/polar_config.yaml" "${custom_config_path}" <<'PY'
import base64
from copy import deepcopy
import json
import os
import shlex
import sys
from pathlib import Path
import yaml

topology_in, topology_out, config_in, config_out = sys.argv[1:]
with open(topology_in, encoding="utf-8") as fh:
    topology = yaml.safe_load(fh) or {}

topology["rollout"]["port"] = int(os.environ["ROLLOUT_PORT"])
topology["rollout"]["host"] = os.environ["POLAR_BIND_HOST"]
topology["rollout"]["public_url"] = f"http://{os.environ['POLAR_PUBLIC_HOST']}:{os.environ['ROLLOUT_PORT']}"
topology["rollout"]["save_dir"] = os.environ["ROLLOUT_SAVE_DIR"]
nodes = topology.get("gateway", {}).get("nodes", [])
if not nodes:
    raise SystemExit("topology has no gateway nodes")
prototype = deepcopy(nodes[0])
prototype["port"] = int(os.environ["GATEWAY_PORT"])
prototype["host"] = os.environ["POLAR_BIND_HOST"]
prototype["model_served"] = os.environ["MODEL_NAME"]
prototype["max_init_workers"] = int(os.environ["POLAR_GATEWAY_MAX_INIT_WORKERS"])
prototype["max_run_workers"] = int(os.environ["POLAR_GATEWAY_MAX_RUN_WORKERS"])
prototype["max_postrun_workers"] = int(os.environ["POLAR_GATEWAY_MAX_POSTRUN_WORKERS"])
prototype.pop("sglang", None)
inference = prototype.setdefault("inference", {})
inference["engine"] = "sglang"
inference["base_url"] = os.environ["SGLANG_ROUTER_BASE_URL"]
if os.environ["POLAR_MULTI_GATEWAY"] == "1":
    hosts = [host.strip() for host in os.environ["POLAR_GATEWAY_HOSTS"].split(",") if host.strip()]
    expected = int(os.environ["POLAR_GATEWAY_COUNT"])
    ranks = [rank.strip() for rank in os.environ.get("POLAR_GATEWAY_RANKS", "").split(",") if rank.strip()]
    if not ranks:
        ranks = [str(rank) for rank in range(expected)]
    if len(hosts) != expected:
        raise SystemExit(f"expected {expected} gateway hosts, got {len(hosts)}: {hosts}")
    if len(ranks) != expected:
        raise SystemExit(f"expected {expected} gateway ranks, got {len(ranks)}: {ranks}")
    if len(set(hosts)) != len(hosts):
        raise SystemExit(f"gateway hosts must be unique: {hosts}")
    if len(set(ranks)) != len(ranks):
        raise SystemExit(f"gateway ranks must be unique: {ranks}")
    nodes[:] = []
    for rank, host in zip(ranks, hosts):
        node = deepcopy(prototype)
        node["id"] = f"slurm-rank-{rank}"
        node["public_url"] = f"http://{host}:{os.environ['GATEWAY_PORT']}"
        nodes.append(node)
else:
    nodes[:] = [prototype]
    nodes[0]["id"] = "localhost-node-01"
    nodes[0]["public_url"] = f"http://{os.environ['POLAR_PUBLIC_HOST']}:{os.environ['GATEWAY_PORT']}"
Path(topology_out).parent.mkdir(parents=True, exist_ok=True)
with open(topology_out, "w", encoding="utf-8") as fh:
    yaml.safe_dump(topology, fh, sort_keys=False)

with open(config_in, encoding="utf-8") as fh:
    config = yaml.safe_load(fh) or {}
config["polar_rollout_url"] = f"http://{os.environ['POLAR_PUBLIC_HOST']}:{os.environ['ROLLOUT_PORT']}"
gateway_host = os.environ.get("POLAR_GATEWAY_PRIMARY_HOST") or os.environ["POLAR_PUBLIC_HOST"]
config["polar_gateway_url"] = f"http://{gateway_host}:{os.environ['GATEWAY_PORT']}"
config["polar_agent_cli_dir"] = os.environ["AGENT_CLI_DIR"]
config["polar_apptainer_image_dir"] = os.environ["APPTAINER_IMAGE_DIR"]
config["polar_request_timeout"] = int(os.environ["POLAR_REQUEST_TIMEOUT"])
config["polar_max_async_level"] = int(os.environ["POLAR_MAX_ASYNC_LEVEL"])
config["polar_min_complete_accept_fraction"] = float(os.environ["POLAR_MIN_COMPLETE_ACCEPT_FRACTION"])

task = config.setdefault("polar_task_template", {})
task["timeout_seconds"] = int(os.environ["POLAR_TASK_TIMEOUT_SECONDS"])
runtime = task.setdefault("runtime", {})
memory_mb = os.environ.get("POLAR_RUNTIME_MEMORY_MB", "").strip()
if memory_mb:
    value = int(memory_mb)
    if value <= 0:
        raise SystemExit(f"POLAR_RUNTIME_MEMORY_MB must be positive: {memory_mb}")
    runtime["memory_mb"] = value
else:
    runtime.pop("memory_mb", None)

def env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}

supported_harnesses = {"pi", "codex", "claude_code", "qwen_code"}
agent_harness = os.environ["AGENT_HARNESS"].strip()
pool_names = [
    name.strip()
    for name in os.environ.get("HARNESS_POOL", "").split(",")
    if name.strip()
]
enabled_harnesses = pool_names or [agent_harness]
unknown = sorted(set(enabled_harnesses) - supported_harnesses)
if unknown:
    raise SystemExit(f"Unsupported agent harness(es): {unknown}")
if len(enabled_harnesses) != len(set(enabled_harnesses)):
    raise SystemExit(f"Harness pool contains duplicates: {enabled_harnesses}")

if "pi" in enabled_harnesses:
    pi_settings = {
        "compaction": {"enabled": env_flag("PI_COMPACTION_ENABLED")},
        "retry": {
            "enabled": env_flag("PI_RETRY_ENABLED"),
            "maxRetries": int(os.environ["PI_RETRY_MAX_RETRIES"]),
            "provider": {"maxRetries": int(os.environ["PI_PROVIDER_MAX_RETRIES"])},
        },
    }
    settings_b64 = base64.b64encode(json.dumps(pi_settings, separators=(",", ":")).encode()).decode("ascii")
    prepare_steps = runtime.get("prepare")
    if not isinstance(prepare_steps, list):
        prepare_steps = []
        runtime["prepare"] = prepare_steps
    prepare_steps.insert(0, {
        "type": "exec",
        "command": (
            'mkdir -p "$HOME/.pi/agent" && '
            f"printf '%s' {shlex.quote(settings_b64)} | base64 -d > \"$HOME/.pi/agent/settings.json\""
        ),
    })

def build_agent(harness):
    agent = {"harness": harness, "settings": {}}
    settings = agent["settings"]
    if harness == "pi":
        agent["model_name"] = os.environ["PI_MODEL_NAME"]
        settings["api_type"] = os.environ["PI_API_TYPE"]
        settings["context_window"] = int(os.environ["PI_CONTEXT_WINDOW"])
        settings["max_tokens"] = int(os.environ["PI_MAX_TOKENS"])
        settings.setdefault("compat", {})["maxTokensField"] = "max_tokens"
        if os.environ.get("PI_THINKING"):
            settings["thinking"] = os.environ["PI_THINKING"]
    elif harness == "codex":
        agent["model_name"] = os.environ["CODEX_MODEL_NAME"]
        # The shared CLI is installed from @openai/codex@latest, so do not pin
        # the code default unless the caller requests a version check.
        settings["version"] = os.environ.get("CODEX_VERSION") or None
        if os.environ.get("CODEX_REASONING_EFFORT"):
            settings["reasoning_effort"] = os.environ["CODEX_REASONING_EFFORT"]
        if os.environ.get("CODEX_REASONING_SUMMARY"):
            settings["reasoning_summary"] = os.environ["CODEX_REASONING_SUMMARY"]
    elif harness == "claude_code":
        agent["model_name"] = os.environ["CLAUDE_MODEL_NAME"]
        if os.environ.get("POLAR_ANTHROPIC_MAX_TOKENS"):
            agent.setdefault("env", {})["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = os.environ[
                "POLAR_ANTHROPIC_MAX_TOKENS"
            ]
        if os.environ.get("CLAUDE_MAX_TURNS"):
            settings["max_turns"] = int(os.environ["CLAUDE_MAX_TURNS"])
        if os.environ.get("CLAUDE_MAX_THINKING_TOKENS"):
            settings["max_thinking_tokens"] = int(os.environ["CLAUDE_MAX_THINKING_TOKENS"])
    elif harness == "qwen_code":
        agent["model_name"] = os.environ["QWEN_CODE_MODEL_NAME"]
        agent.setdefault("env", {})["QWEN_CODE_MAX_OUTPUT_TOKENS"] = os.environ[
            "QWEN_CODE_MAX_OUTPUT_TOKENS"
        ]
    return agent

task["agent"] = build_agent(agent_harness)
if pool_names:
    config["polar_harness_pool"] = [build_agent(name) for name in pool_names]
    config["polar_harness_seed"] = int(os.environ["HARNESS_SEED"])
else:
    config.pop("polar_harness_pool", None)
    config.pop("polar_harness_seed", None)
task.setdefault("builder", {})["strategy"] = os.environ["POLAR_BUILDER_STRATEGY"]

Path(config_out).parent.mkdir(parents=True, exist_ok=True)
with open(config_out, "w", encoding="utf-8") as fh:
    yaml.safe_dump(config, fh, sort_keys=False)
print(f"Topology: {topology_out}")
print(f"Polar config: {config_out}")
PY
}

preflight() {
    command -v git >/dev/null 2>&1 || die "git not found"
    command -v ray >/dev/null 2>&1 || die "ray command not found"
    command -v "${POLAR_APPTAINER_BIN}" >/dev/null 2>&1 || [ -x "${POLAR_APPTAINER_BIN}" ] || die "apptainer not found: ${POLAR_APPTAINER_BIN}"
    if is_path_like "${hf_checkpoint}"; then
        [ -d "${hf_checkpoint}" ] || die "HF checkpoint not found: ${hf_checkpoint}"
    fi
    [ -f "${full_prompt_data}" ] || die "training data not found: ${full_prompt_data}"
    [ -f "${model_args_file}" ] || die "model args file not found: ${model_args_file}"
    [ -f "${slime_dir}/train_async.py" ] || die "Slime missing: ${slime_dir}"
    [ -d "${megatron_dir}/megatron" ] || die "Megatron missing: ${megatron_dir}"
    [ -x "${agent_cli_dir}/bin/node" ] || die "Node CLI missing: ${agent_cli_dir}/bin/node"
    local configured_harnesses="${harness_pool:-${agent_harness}}"
    local harness
    local harnesses=()
    IFS=, read -r -a harnesses <<<"${configured_harnesses}"
    for harness in "${harnesses[@]}"; do
    case "${harness}" in
        pi)
            [ -x "${agent_cli_dir}/bin/pi" ] || die "PI CLI missing: ${agent_cli_dir}/bin/pi"
            PATH="${agent_cli_dir}/bin:${PATH}" "${agent_cli_dir}/bin/pi" --version >/dev/null 2>&1 || \
                die "PI CLI dependency preflight failed: ${agent_cli_dir}/bin/pi --version"
            ;;
        codex) [ -x "${agent_cli_dir}/bin/codex" ] || die "Codex CLI missing: ${agent_cli_dir}/bin/codex" ;;
        claude_code) [ -x "${agent_cli_dir}/bin/claude" ] || die "Claude Code CLI missing: ${agent_cli_dir}/bin/claude" ;;
        qwen_code) [ -x "${agent_cli_dir}/bin/qwen" ] || die "Qwen Code CLI missing: ${agent_cli_dir}/bin/qwen" ;;
        *) die "Unsupported agent harness: ${harness}" ;;
    esac
    done
    "${python_bin}" - <<'PY'
import importlib
mods = ["ray", "torch", "sglang", "polar", "yaml", "httpx", "wandb"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("Missing Python dependencies:\n" + "\n".join(missing))
PY
    if [ "${use_wandb}" = "1" ] && [ "${wandb_mode}" = "online" ] && [ -z "${wandb_api_key}" ]; then
        die "wandb_api_key is required for online W&B"
    fi
}

export_training_env() {
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export SLIME_SGLANG_BASE_PORT="${slime_sglang_base_port}"
    export RAY_TMPDIR="${ray_tmpdir}"
    export RAY_ADDRESS="${ray_head_ip}:${ray_port}"
    export RAY_JOB_ADDRESS="http://127.0.0.1:${ray_dashboard_port}"
    export SGLANG_ROUTER_HOST="${sglang_router_host}"
    export SGLANG_ROUTER_BASE_URL="${sglang_router_base_url}"
    export LD_LIBRARY_PATH="$(runtime_ld_library_path)"
    export POLAR_GATEWAY_TOKENIZER="${hf_checkpoint}"
    export POLAR_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
    if [ -d /opt/polr_venv ]; then
        export VIRTUAL_ENV="/opt/polr_venv"
    fi
}

service_pids=()

gateway_rank_allowed() {
    if [ "${polar_multi_gateway}" = "1" ]; then
        local rank="${SLURM_NODEID:-${SLURM_PROCID:-${RAY_NODE_RANK:-0}}}"
        case "${rank}" in
            ''|*[!0-9]*) die "invalid gateway rank: ${rank}" ;;
        esac
        local ranks="${polar_gateway_ranks:-}"
        if [ -z "${ranks}" ]; then
            ranks="$(seq -s, 0 $((polar_gateway_count - 1)))"
        fi
        case ",${ranks}," in
            *,"${rank}",*) return 0 ;;
            *) return 1 ;;
        esac
    fi
    return 0
}

gateway_node_id() {
    if [ "${polar_multi_gateway}" = "1" ]; then
        local rank="${SLURM_NODEID:-${SLURM_PROCID:-${RAY_NODE_RANK:-0}}}"
        gateway_rank_allowed || die "rank ${rank} is not configured as a Polar gateway rank (${polar_gateway_ranks:-unset})"
        printf 'slurm-rank-%s\n' "${rank}"
    else
        printf 'localhost-node-01\n'
    fi
}

wait_http_health() {
    local name="$1"
    local url="$2"
    local attempts="${3:-120}"
    "${python_bin}" - "${name}" "${url}" "${attempts}" <<'PY'
import sys
import time
import urllib.request

name, url, attempts = sys.argv[1], sys.argv[2], int(sys.argv[3])
for _ in range(attempts):
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            if resp.status == 200:
                print(f"{name} healthy")
                raise SystemExit(0)
    except Exception:
        time.sleep(2)
raise SystemExit(f"{name} health check failed: {url}")
PY
}

wait_gateway_fleet_ready() {
    [ "${polar_multi_gateway}" = "1" ] || return 0
    "${python_bin}" - "${rollout_port}" "${polar_gateway_count}" "${polar_gateway_ranks}" <<'PY'
import json
import sys
import time
import urllib.request

port, expected = int(sys.argv[1]), int(sys.argv[2])
ranks = [rank.strip() for rank in sys.argv[3].split(",") if rank.strip()]
if not ranks:
    ranks = [str(rank) for rank in range(expected)]
if len(ranks) != expected:
    raise SystemExit(f"expected {expected} gateway ranks, got {ranks}")
url = f"http://127.0.0.1:{port}/nodes"
deadline = time.time() + 240
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        healthy = [node for node in payload if node.get("healthy")]
        ids = {str(node.get("node_id")) for node in healthy}
        want = {f"slurm-rank-{rank}" for rank in ranks}
        if ids == want:
            print(f"Polar gateway fleet healthy: {len(healthy)}/{expected}")
            raise SystemExit(0)
        print(f"Waiting for gateway fleet: healthy_ids={sorted(ids)} want={sorted(want)}", flush=True)
    except Exception as exc:
        print(f"Waiting for gateway fleet endpoint: {type(exc).__name__}: {exc}", flush=True)
    time.sleep(2)
raise SystemExit(f"Polar gateway fleet did not reach {expected} healthy nodes")
PY
}

start_gateway_sidecar() {
    export_training_env
    topology_path="${run_dir}/topology.yaml"
    custom_config_path="${run_dir}/polar_config.yaml"
    for _ in $(seq 1 180); do
        [ -s "${topology_path}" ] && [ -s "${custom_config_path}" ] && break
        sleep 1
    done
    [ -s "${topology_path}" ] || die "sidecar timed out waiting for topology: ${topology_path}"
    [ -s "${custom_config_path}" ] || die "sidecar timed out waiting for Polar config: ${custom_config_path}"
    wait_http_health "remote Polar rollout" "http://${ray_head_ip}:${rollout_port}/health" 180

    service_pids=()
    cleanup_sidecar() {
        set +e
        trap - EXIT
        local pid
        for pid in "${service_pids[@]}"; do
            kill "${pid}" 2>/dev/null || true
        done
        for pid in "${service_pids[@]}"; do
            wait "${pid}" 2>/dev/null || true
        done
    }
    trap cleanup_sidecar EXIT

    local node_id
    node_id="$(gateway_node_id)"
    log "Starting Polar gateway sidecar node_id=${node_id} on :${gateway_port}"
    polar serve_gateway -c "${topology_path}" --node-id "${node_id}" \
        >"${run_log_dir}/gateway-${node_id}.log" 2>&1 &
    service_pids+=("$!")
    wait_http_health "Polar gateway ${node_id}" "http://127.0.0.1:${gateway_port}/health" 60

    while [ ! -f "${stop_file}" ]; do
        kill -0 "${service_pids[0]}" 2>/dev/null || wait "${service_pids[0]}"
        sleep 5
    done
}

wait_for_ray_cluster() {
    "${python_bin}" - "${ray_head_ip}:${ray_port}" "${ray_expected_num_gpus}" "${ray_cluster_timeout_seconds}" <<'PY'
import sys
import time

import ray

address, expected, timeout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + timeout
ray.init(address=address, ignore_reinit_error=True)
while time.time() < deadline:
    actual = int(ray.cluster_resources().get("GPU", 0))
    if actual >= expected:
        print(f"Ray cluster ready: {actual} GPUs (expected {expected})")
        ray.shutdown()
        raise SystemExit(0)
    print(f"Waiting for Ray GPUs: {actual}/{expected}", flush=True)
    time.sleep(5)
ray.shutdown()
raise SystemExit(f"Ray cluster did not reach {expected} GPUs within {timeout}s")
PY
}

start_services_and_train() {
    export_training_env
    service_pids=()
    cleanup() {
        local rc="${1:-$?}"
        set +e
        trap - EXIT
        log "Cleaning up services"
        local pid
        for pid in "${service_pids[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                log "Terminating service pid ${pid}"
                kill -TERM "${pid}" 2>/dev/null || true
            fi
        done
        sleep 2
        for pid in "${service_pids[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                log "Killing service pid ${pid}"
                kill -KILL "${pid}" 2>/dev/null || true
            fi
            wait "${pid}" 2>/dev/null || true
        done
        if [ "${ray_stop_on_exit}" = "1" ]; then
            if command -v timeout >/dev/null 2>&1; then
                timeout 30s ray stop --force >/dev/null 2>&1 || true
            else
                ray stop --force >/dev/null 2>&1 || true
            fi
        fi
        return "${rc}"
    }
    trap 'rc=$?; cleanup "$rc"; exit "$rc"' EXIT

    if [ -z "${load_debug_rollout_data}" ]; then
        log "Starting Polar rollout on :${rollout_port}"
        polar serve_rollout -c "${topology_path}" >"${run_log_dir}/rollout.log" 2>&1 &
        service_pids+=("$!")
        sleep 2

        wait_http_health "rollout" "http://127.0.0.1:${rollout_port}/health" 60
        if gateway_rank_allowed; then
            local node_id
            node_id="$(gateway_node_id)"
            log "Starting Polar gateway ${node_id} on :${gateway_port}"
            polar serve_gateway -c "${topology_path}" --node-id "${node_id}" >"${run_log_dir}/gateway.log" 2>&1 &
            service_pids+=("$!")
            sleep 2
            wait_http_health "Polar gateway ${node_id}" "http://127.0.0.1:${gateway_port}/health" 60
        else
            log "Skipping local Polar gateway on non-gateway rank"
        fi
        wait_gateway_fleet_ready
    else
        log "Debug rollout replay: skipping Polar rollout and gateway services"
    fi

    if [ "${ray_use_existing_cluster}" = "1" ]; then
        log "Using existing Ray cluster at ${ray_head_ip}:${ray_port}"
    else
        log "Starting local Ray head with ${ray_head_num_gpus} GPUs"
        ray stop --force >"${run_log_dir}/ray-stop-before-start.log" 2>&1 || true
        ray start --head \
            --node-ip-address "${ray_head_ip}" \
            --port "${ray_port}" \
            --num-cpus "${ray_num_cpus}" \
            --num-gpus "${ray_head_num_gpus}" \
            --dashboard-host 0.0.0.0 \
            --dashboard-port "${ray_dashboard_port}" \
            --temp-dir "${ray_tmpdir}" \
            --disable-usage-stats >"${run_log_dir}/ray-start.log" 2>&1
    fi
    wait_for_ray_cluster

    local runtime_env_path="${run_dir}/ray_runtime_env.yaml"
    "${python_bin}" - "${runtime_env_path}" <<PY
import os
import sys
import yaml
keys = [
    "PYTHONPATH", "PATH", "VIRTUAL_ENV", "CUDA_DEVICE_MAX_CONNECTIONS",
    "NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_DEBUG", "NVTE_DEBUG_LEVEL",
    "SLIME_SGLANG_BASE_PORT", "SLIME_RESPONSE_ONLY_LOGPROBS", "SLIME_DEBUG_ONE_PER_GROUP",
    "SLIME_DEBUG_GRAD_HOOKS", "SLIME_DEBUG_PARAM_GRADS", "SLIME_EXIT_DURATION_MINUTES",
    "SLIME_TRAIN_IDLE_PULSE_AFTER_SECONDS", "SLIME_TRAIN_IDLE_PULSE_DURATION_SECONDS",
    "SLIME_TRAIN_IDLE_PULSE_MATRIX_SIZE",
    "WANDB_API_KEY", "WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_DIR",
    "HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
    "HF_DATASETS_CACHE", "HF_MODULES_CACHE", "SENTENCE_TRANSFORMERS_HOME",
    "APPTAINER_CACHEDIR", "APPTAINER_TMPDIR", "APPTAINER_WORKDIR",
    "SINGULARITY_CACHEDIR", "SINGULARITY_TMPDIR", "POLAR_APPTAINER_BIN",
    "POLAR_APPTAINER_DIRECT_EXEC", "POLAR_APPTAINER_EXEC_MODE", "TRITON_CACHE_DIR", "TRITON_HOME",
    "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR", "CUDA_CACHE_PATH", "NUMBA_CACHE_DIR",
    "FLASHINFER_WORKSPACE_DIR", "LD_LIBRARY_PATH", "POLAR_GATEWAY_TOKENIZER",
    "POLAR_CHAT_TEMPLATE_KWARGS",
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "RAY_MEMORY_USAGE_THRESHOLD",
    "RAY_memory_usage_threshold",
]
env = {key: os.environ[key] for key in keys if os.environ.get(key)}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    yaml.safe_dump({"env_vars": env}, fh, sort_keys=True)
os.chmod(sys.argv[1], 0o600)
PY
    log "Ray runtime env: ${runtime_env_path}"

    local load_dir="${ref_load}"
    if checkpoint_ready "${save_dir}"; then
        load_dir="${save_dir}"
    fi

    local wandb_args=()
    if [ "${use_wandb}" = "1" ]; then
        wandb_args=(
            --use-wandb
            --wandb-mode "${wandb_mode}"
            --wandb-dir "${wandb_dir}"
            --wandb-project "${wandb_project}"
            --wandb-group "${wandb_group}"
            --wandb-team "${wandb_entity}"
            --wandb-run-id "${wandb_run_id}"
        )
        [ "${wandb_random_suffix}" = "1" ] || wandb_args+=(--disable-wandb-random-suffix)
    fi

    local length_args=()
    if [ -n "${num_rollout}" ]; then
        length_args=(--num-rollout "${num_rollout}")
    else
        length_args=(--num-epoch "${num_epoch}")
    fi

    local start_rollout_args=()
    if [ -n "${start_rollout_id}" ]; then
        start_rollout_args=(--start-rollout-id "${start_rollout_id}")
    fi

    # shellcheck source=/dev/null
    source "${model_args_file}"
    local model_args=()
    local model_arg
    for model_arg in "${MODEL_ARGS[@]}"; do
        if [ "${model_arg}" = "--use-gated-attention" ] && \
            ! grep -R -- "--use-gated-attention" "${megatron_dir}/megatron/training" >/dev/null 2>&1; then
            log "Skipping --use-gated-attention; this Megatron checkout does not expose that CLI flag."
            continue
        fi
        model_args+=("${model_arg}")
    done

    local clip_grad_args=()
    if [ -n "${clip_grad}" ]; then
        clip_grad_args=(--clip-grad "${clip_grad}")
    fi

    # Slime's built-in TIS computes exp(log-ratio) before clipping. Extremely
    # long PI trajectories can therefore overflow in backward even when the
    # clipped forward value is finite. Keep it opt-in until a bounded TIS
    # implementation is used.
    local tis_args=()
    if [ "${use_tis}" = "1" ]; then
        tis_args=(--use-tis)
    fi

    local fault_tolerance_args=()
    if [ "${use_fault_tolerance}" = "1" ]; then
        fault_tolerance_args=(
            --use-fault-tolerance
            --rollout-health-check-interval "${rollout_health_check_interval}"
            --rollout-health-check-timeout "${rollout_health_check_timeout}"
            --rollout-health-check-first-wait "${rollout_health_check_first_wait}"
        )
    fi

    local batching_args=()
    if [ "${use_dynamic_batch_size}" = "1" ]; then
        batching_args=(
            --qkv-format "${qkv_format}"
            --use-dynamic-batch-size
            --max-tokens-per-gpu "${max_tokens_per_gpu}"
        )
    else
        batching_args=(
            --qkv-format "${qkv_format}"
            --micro-batch-size "${micro_batch_size}"
            --global-batch-size "${global_batch_size}"
        )
    fi

    local sequence_parallel_args=()
    if [ "${use_sequence_parallel}" = "1" ]; then
        sequence_parallel_args=(--sequence-parallel)
    fi

    local graceful_exit_args=()
    if [ "${exit_duration_minutes:-0}" -gt 0 ]; then
        graceful_exit_args=(--graceful-exit-at-unix-time "$(( $(date +%s) + exit_duration_minutes * 60 ))")
    fi

    local train_args=(
        "${slime_dir}/train_async.py"
        --actor-num-nodes "${actor_num_nodes}"
        --actor-num-gpus-per-node "${actor_num_gpus_per_node}"
        --rollout-num-gpus "${rollout_num_gpus}"
        --rollout-num-gpus-per-engine "${rollout_num_gpus_per_engine}"
        "${model_args[@]}"
        --hf-checkpoint "${hf_checkpoint}"
        --tokenizer-model "${hf_checkpoint}"
        --tokenizer-type HuggingFaceTokenizer
        --no-use-tokenizer-model-from-checkpoint-args
        --ref-load "${ref_load}"
        --load "${load_dir}"
        --save "${save_dir}"
        --dist-ckpt-strictness "${dist_ckpt_strictness:-log_all}"
        # The NRT training image does not ship nvidia-resiliency-ext.  Recent
        # Megatron defaults to the nvrx checkpoint reader even when async save
        # is disabled, so select the built-in reader explicitly for both load
        # and save/resume.
        --async-strategy "${async_strategy:-mcore}"
        --save-interval "${save_interval}"
        "${graceful_exit_args[@]}"
        --update-weights-interval 1
        --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async
        --custom-rm-path slime_bridge.reward.reward_func
        --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards
        --custom-config-path "${custom_config_path}"
        --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer
        --prompt-data "${prompt_data}"
        --input-key prompt
        --label-key label
        --metadata-key metadata
        --rollout-shuffle
        --reward-key score
        "${start_rollout_args[@]}"
        "${length_args[@]}"
        --rollout-batch-size "${rollout_batch_size}"
        --n-samples-per-prompt "${n_samples_per_prompt}"
        --rollout-max-response-len "${rollout_max_response_len}"
        --rollout-max-prompt-len "${rollout_max_prompt_len}"
        --save-debug-rollout-data "${run_dir}/debug_rollout_{rollout_id}.pt"
        --dynamic-history
        --num-steps-per-rollout "${num_steps_per_rollout}"
        --distributed-timeout-minutes "${distributed_timeout_minutes}"
        "${fault_tolerance_args[@]}"
        --qwen-gdn-backend "${qwen_gdn_backend}"
        --tensor-model-parallel-size "${tensor_model_parallel_size}"
        "${sequence_parallel_args[@]}"
        --pipeline-model-parallel-size 1
        --context-parallel-size 1
        --expert-model-parallel-size 1
        --expert-tensor-parallel-size 1
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
        "${batching_args[@]}"
        --log-probs-chunk-size "${log_probs_chunk_size}"
        --advantage-estimator grpo
        --normalize-advantages
        "${tis_args[@]}"
        --use-kl-loss
        --kl-loss-coef "${kl_loss_coef}"
        --kl-loss-type "${kl_loss_type}"
        --entropy-coef 0.0
        --eps-clip "${eps_clip}"
        --eps-clip-high "${eps_clip_high}"
        --eps-clip-c "${eps_clip_c}"
        --optimizer adam
        --lr "${train_lr}"
        --lr-decay-style constant
        # Keep the scheduler horizon independent of the per-allocation
        # --num-rollout boundary.  Checkpoints otherwise encode 1*64, 4*64,
        # ... as incompatible horizons when a run is resumed in chunks.
        --lr-decay-iters "${target_num_rollout:-74}"
        # The initial step-0 checkpoint predates the fixed horizon.  Override
        # its scheduler metadata while preserving its loaded num_steps,
        # optimizer tensors, model weights, and constant LR/weight decay.
        --override-opt-param-scheduler
        --weight-decay 0.1
        "${clip_grad_args[@]}"
        --adam-beta1 0.9
        --adam-beta2 0.98
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend "${attention_backend}"
        --no-gradient-accumulation-fusion
        --sglang-mem-fraction-static "${sglang_mem_fraction_static}"
        --sglang-context-length "${sglang_context_length}"
        --sglang-served-model-name "${model_name}"
        --sglang-tool-call-parser qwen3_coder
        --sglang-log-level "${sglang_log_level}"
        --router-policy "${sglang_router_policy:-round_robin}"
        --sglang-router-port "${sglang_router_port}"
        "${wandb_args[@]}"
    )

    if [ -n "${load_debug_rollout_data}" ]; then
        train_args+=(--load-debug-rollout-data "${load_debug_rollout_data}")
        if [ -n "${load_debug_rollout_data_subsample}" ]; then
            train_args+=(--load-debug-rollout-data-subsample "${load_debug_rollout_data_subsample}")
        fi
    fi
    if [ -n "${fetch_trajectory_retry_times:-}" ]; then
        train_args+=(--fetch-trajectory-retry-times "${fetch_trajectory_retry_times}")
    fi

    log "Launching Slime train_async.py"
    set +e
    "${python_bin}" - "${runtime_env_path}" "${train_args[@]}" <<'PY' 2>&1 | tee "${run_log_dir}/train-direct.log"
import os
import runpy
import sys

import ray
import yaml

runtime_env_path = sys.argv[1]
script = sys.argv[2]
script_args = sys.argv[3:]
with open(runtime_env_path, "r", encoding="utf-8") as fh:
    runtime_env = yaml.safe_load(fh) or {}
ray.init(
    address=os.environ.get("RAY_ADDRESS") or "auto",
    ignore_reinit_error=True,
    runtime_env=runtime_env,
)
sys.argv = [script, *script_args]
runpy.run_path(script, run_name="__main__")
PY
    local rc="${PIPESTATUS[0]}"
    set -e
    return "${rc}"
}

log "Project: ${project_root}"
log "Run ID: ${run_id}"
log "Training sqsh: ${train_sqsh}"
log "Slime dir: ${slime_dir}"
log "Megatron dir: ${megatron_dir}"
log "Attention backend: ${attention_backend} (flash-attn-4 ${flash_attention_version:-n/a})"
log "HF checkpoint: ${hf_checkpoint}"
log "Model args: ${model_args_file}"
log "Megatron load: ${ref_load}"
log "Prompt source: ${full_prompt_data}"
log "Smoke rows: ${smoke_rows}"
log "Runtime SIF dir: ${apptainer_image_dir}"
log "Agent CLI dir: ${agent_cli_dir}"
log "Agent harness: ${agent_harness}"
if [ -n "${harness_pool}" ]; then
    log "Harness pool: ${harness_pool} (seed=${harness_seed})"
fi
if [ -n "${anthropic_max_tokens}" ]; then
    log "Anthropic max tokens cap: ${anthropic_max_tokens}"
fi
log "Qwen Code max output tokens: ${qwen_code_max_output_tokens}"
log "GPU split: total=${total_gpus}, train=${train_num_gpus}, rollout=${rollout_num_gpus}, tp=${tensor_model_parallel_size}"
log "Batch: rollout=${rollout_batch_size}, samples/prompt=${n_samples_per_prompt}, num_rollout=${num_rollout:-epoch}"
log "Rollout fault tolerance: ${use_fault_tolerance} (interval=${rollout_health_check_interval}s, timeout=${rollout_health_check_timeout}s)"
log "Rollout start: ${start_rollout_id:-checkpoint}"
log "W&B: entity=${wandb_entity}, project=${wandb_project}, group=${wandb_group}, run_id=${wandb_run_id}"

if [ "${patch_container_runtime_only:-0}" = "1" ]; then
    patch_container_runtime
    exit 0
fi

if [ "${pi_multigw_sidecar:-0}" = "1" ]; then
    start_gateway_sidecar
    exit 0
fi

ensure_current_checkouts
SLIME_DIR="${slime_dir}" bash "${project_root}/scripts/patch/patch_slime_megatron_compat.sh"
patch_swegym_grpo_compat
patch_wandb_compat
patch_checkpoint_async_strategy_compat
patch_container_runtime
prepare_prompt_data_and_sifs
prepare_agent_cli
ensure_checkpoint
render_runtime_configs
preflight

if [ "${dry_run}" = "1" ]; then
    log "dry_run=1; dependency preparation and config rendering complete."
    exit 0
fi

start_services_and_train
