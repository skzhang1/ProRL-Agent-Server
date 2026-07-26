#!/usr/bin/env bash
#SBATCH --job-name=swebench-harness-eval
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=03:55:00
#SBATCH --output=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/examples/swebench_verified/results/slurm/%x-%A_%a.out
#SBATCH --error=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/examples/swebench_verified/results/slurm/%x-%A_%a.err
#SBATCH --export=ALL

# Strict pass@1 SWE-bench Verified eval for a Polar harness + Qwen3.5-4B.
#
# One script covers both requested variants:
#   - base:  Qwen3.5-4B base Megatron checkpoint
#   - final: the PI-trained checkpoint from swegym_slime_grpo, iter_0000073
#
# Login node usage, no Slurm allocation needed:
#   bash examples/swebench_verified/submit_swebench_pi_apptainer.sh plan
#   bash examples/swebench_verified/submit_swebench_pi_apptainer.sh smoke
#   bash examples/swebench_verified/submit_swebench_pi_apptainer.sh submit
#   bash examples/swebench_verified/submit_swebench_pi_apptainer.sh aggregate
#
# The submit mode uses Slurm arrays over 1-based SWE-bench Verified ranges.
# Defaults are intentionally conservative: 50 tasks/shard, at most 2 active
# shard jobs, and final-checkpoint evaluation starts after the base array.
set -euo pipefail


# ---------------------------------------------------------------------------
# Login-node driver. Running this script outside Slurm only prints a plan or
# submits Slurm array jobs; the actual evaluation happens inside each array task.
# ---------------------------------------------------------------------------
SCRIPT_MODE="${1:-${SCRIPT_MODE:-plan}}"
if [ "$#" -gt 0 ]; then
    shift
fi
while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint)
            [ "$#" -ge 2 ] || { printf 'ERROR: --checkpoint requires a path\n' >&2; exit 2; }
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --run-group)
            [ "$#" -ge 2 ] || { printf 'ERROR: --run-group requires a value\n' >&2; exit 2; }
            RUN_GROUP="$2"
            shift 2
            ;;
        --shard-size)
            [ "$#" -ge 2 ] || { printf 'ERROR: --shard-size requires an integer\n' >&2; exit 2; }
            SHARD_SIZE="$2"
            shift 2
            ;;
        --array-concurrency)
            [ "$#" -ge 2 ] || { printf 'ERROR: --array-concurrency requires an integer\n' >&2; exit 2; }
            ARRAY_CONCURRENCY="$2"
            shift 2
            ;;
        --variants)
            [ "$#" -ge 2 ] || { printf 'ERROR: --variants requires a value\n' >&2; exit 2; }
            EVAL_VARIANTS="$2"
            shift 2
            ;;
        --harness)
            [ "$#" -ge 2 ] || { printf 'ERROR: --harness requires a value\n' >&2; exit 2; }
            AGENT_HARNESS="$2"
            shift 2
            ;;
        *)
            printf 'ERROR: unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

PROJECT_ROOT_DEFAULT="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server"
TOTAL_DATASET_INSTANCES="${TOTAL_DATASET_INSTANCES:-500}"
SHARD_SIZE="${SHARD_SIZE:-50}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
RUN_GROUP="${RUN_GROUP:-swebench_verified_pi_qwen35_4b}"
FINAL_AFTER_BASE="${FINAL_AFTER_BASE:-1}"
EVAL_VARIANTS="${EVAL_VARIANTS:-base,checkpoint}"
AGENT_HARNESS="${AGENT_HARNESS:-pi}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/checkpoints/swe/iter_0000073}"

case "${ARRAY_CONCURRENCY}" in
    1|2) ;;
    *) printf 'ERROR: array concurrency must be 1 or 2 (interactive-node limit)\n' >&2; exit 2 ;;
esac
case "${AGENT_HARNESS}" in
    pi|codex|claude_code|qwen_code) ;;
    *) printf 'ERROR: unsupported harness: %s\n' "${AGENT_HARNESS}" >&2; exit 2 ;;
esac
CHECKPOINT_TAG="$(basename -- "${CHECKPOINT_PATH}")"
CHECKPOINT_TAG="$(printf '%s' "${CHECKPOINT_TAG}" | tr -c 'A-Za-z0-9_.-' '_')"

ceil_div() {
    local n="$1"
    local d="$2"
    printf '%s\n' "$(( (n + d - 1) / d ))"
}

print_login_plan() {
    local shards
    shards="$(ceil_div "${TOTAL_DATASET_INSTANCES}" "${SHARD_SIZE}")"
    cat <<EOF
=============================================
SWE-bench Verified harness + Qwen3.5 evaluation plan
  Script:        ${BASH_SOURCE[0]}
  Project root:  ${PROJECT_ROOT_DEFAULT}
  Run group:     ${RUN_GROUP}
  Harness:       ${AGENT_HARNESS}
  Variants:      ${EVAL_VARIANTS}  (base + final checkpoint by default)
  Dataset:       SWE-bench Verified test (${TOTAL_DATASET_INSTANCES} tasks)
  Shard size:    ${SHARD_SIZE} tasks -> ${shards} shard(s) per variant
  Concurrency:   ${ARRAY_CONCURRENCY} active one-node shard job(s), hard cap 2
  Ordering:      final-after-base=${FINAL_AFTER_BASE}
  Checkpoint:    ${CHECKPOINT_PATH}
  Slurm:         account=nvr_lpr_agentic, partition=interactive, nodes=1, gpus=8, time=03:55:00

Commands:
  Plan only:     bash ${BASH_SOURCE[0]} plan
  Smoke test:    bash ${BASH_SOURCE[0]} smoke
  Full submit:   bash ${BASH_SOURCE[0]} submit
  Aggregate:     bash ${BASH_SOURCE[0]} aggregate
=============================================
EOF
}

submit_one_array() {
    local variant="$1"
    local dependency_arg="${2:-}"
    local shards last_idx export_vars job_id
    shards="$(ceil_div "${TOTAL_DATASET_INSTANCES}" "${SHARD_SIZE}")"
    last_idx="$((shards - 1))"
    export_vars="ALL,SCRIPT_MODE=run,MODEL_SOURCE=${variant},AGENT_HARNESS=${AGENT_HARNESS},RUN_GROUP=${RUN_GROUP},SHARD_SIZE=${SHARD_SIZE},TOTAL_DATASET_INSTANCES=${TOTAL_DATASET_INSTANCES},CHECKPOINT_PATH=${CHECKPOINT_PATH}"
    if [ -n "${dependency_arg}" ]; then
        job_id="$(sbatch --parsable --dependency="${dependency_arg}" --array="0-${last_idx}%${ARRAY_CONCURRENCY}" --export="${export_vars}" "${BASH_SOURCE[0]}")"
    else
        job_id="$(sbatch --parsable --array="0-${last_idx}%${ARRAY_CONCURRENCY}" --export="${export_vars}" "${BASH_SOURCE[0]}")"
    fi
    printf '%s\n' "${job_id}"
}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    case "${SCRIPT_MODE}" in
        plan|--plan|-n)
            print_login_plan
            exit 0
            ;;
        smoke)
            mkdir -p "${PROJECT_ROOT_DEFAULT}/examples/swebench_verified/results/slurm"
            print_login_plan
            base_job=""
            case ",${EVAL_VARIANTS}," in
                *,base,*)
                    base_job="$(sbatch --parsable --export="ALL,SCRIPT_MODE=run,MODEL_SOURCE=base,AGENT_HARNESS=${AGENT_HARNESS},RUN_GROUP=${RUN_GROUP}_smoke,INSTANCE_RANGE=1-1,MAX_TASKS=-1,CHECKPOINT_PATH=${CHECKPOINT_PATH}" "${BASH_SOURCE[0]}")"
                    printf 'Submitted base smoke: %s\n' "${base_job}"
                    ;;
            esac
            case ",${EVAL_VARIANTS}," in
                *,checkpoint,*|*,final,*)
                    dep=""
                    if [ "${FINAL_AFTER_BASE}" = "1" ] && [ -n "${base_job}" ]; then
                        dep="--dependency=afterany:${base_job}"
                    fi
                    final_job="$(sbatch --parsable ${dep} --export="ALL,SCRIPT_MODE=run,MODEL_SOURCE=checkpoint,AGENT_HARNESS=${AGENT_HARNESS},RUN_GROUP=${RUN_GROUP}_smoke,INSTANCE_RANGE=1-1,MAX_TASKS=-1,CHECKPOINT_PATH=${CHECKPOINT_PATH}" "${BASH_SOURCE[0]}")"
                    printf 'Submitted checkpoint smoke: %s\n' "${final_job}"
                    ;;
            esac
            exit 0
            ;;
        submit)
            mkdir -p "${PROJECT_ROOT_DEFAULT}/examples/swebench_verified/results/slurm"
            print_login_plan
            base_job=""
            case ",${EVAL_VARIANTS}," in
                *,base,*) base_job="$(submit_one_array base)"; printf 'Submitted base array: %s\n' "${base_job}" ;;
            esac
            case ",${EVAL_VARIANTS}," in
                *,checkpoint,*|*,final,*)
                    dep=""
                    if [ "${FINAL_AFTER_BASE}" = "1" ] && [ -n "${base_job}" ]; then
                        dep="afterany:${base_job}"
                    fi
                    final_job="$(submit_one_array checkpoint "${dep}")"
                    printf 'Submitted final-checkpoint array: %s\n' "${final_job}"
                    ;;
            esac
            exit 0
            ;;
        aggregate)
            print_login_plan
            aggregate_script="${AGGREGATE_SCRIPT:-${PROJECT_ROOT_DEFAULT}/examples/swebench_verified/aggregate_results.py}"
            case ",${EVAL_VARIANTS}," in
                *,base,*)
                    python3 "${aggregate_script}" \
                        --project-root "${PROJECT_ROOT_DEFAULT}" \
                        --run-group "${RUN_GROUP}_base" \
                        --expected-total "${TOTAL_DATASET_INSTANCES}"
                    ;;
            esac
            case ",${EVAL_VARIANTS}," in
                *,checkpoint,*|*,final,*)
                    python3 "${aggregate_script}" \
                        --project-root "${PROJECT_ROOT_DEFAULT}" \
                        --run-group "${RUN_GROUP}_checkpoint_${CHECKPOINT_TAG}" \
                        --expected-total "${TOTAL_DATASET_INSTANCES}"
                    ;;
            esac
            exit 0
            ;;
        run)
            printf 'ERROR: SCRIPT_MODE=run must execute inside Slurm. Use: bash %s submit\n' "${BASH_SOURCE[0]}" >&2
            exit 1
            ;;
        *)
            printf 'ERROR: unknown mode %s. Use plan, smoke, submit, aggregate.\n' "${SCRIPT_MODE}" >&2
            exit 1
            ;;
    esac
fi

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
    SCRIPT_DIR="${PROJECT_ROOT}/examples/swebench_verified"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/examples/swebench_verified" ]; then
    PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
    SCRIPT_DIR="${PROJECT_ROOT}/examples/swebench_verified"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
    PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi
TRAIN_ROOT="${TRAIN_ROOT:-${PROJECT_ROOT}}"
SCRIPT_PATH="${SCRIPT_PATH:-${SCRIPT_DIR}/submit_swebench_pi_apptainer.sh}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "${PROJECT_ROOT}" "$1" ;;
    esac
}

count_jsonl_rows() {
    awk 'NF { count += 1 } END { print count + 0 }' "$1"
}

detect_host_ip() {
    python3 - <<'PY'
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

# ---------------------------------------------------------------------------
# Paths and run identity
# ---------------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
MODEL_SOURCE="${MODEL_SOURCE:-${EVAL_VARIANT:-base}}"
case "${MODEL_SOURCE}" in
    final|ckpt) MODEL_SOURCE="checkpoint" ;;
    base|checkpoint) ;;
    *) die "MODEL_SOURCE must be base or checkpoint/final, got: ${MODEL_SOURCE}" ;;
esac
HF_CHECKPOINT="${HF_CHECKPOINT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/model/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist}"
BASE_LOAD="${BASE_LOAD:-${REF_LOAD}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
if [ "${MODEL_SOURCE}" = "checkpoint" ]; then
    checkpoint_name="$(basename -- "${CHECKPOINT_PATH}")"
    case "${checkpoint_name}" in
        iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9])
            CHECKPOINT_DIR="$(dirname -- "${CHECKPOINT_PATH}")"
            CHECKPOINT_STEP="$((10#${checkpoint_name#iter_}))"
            ;;
        *)
            CHECKPOINT_DIR="${CHECKPOINT_PATH}"
            CHECKPOINT_STEP=""
            ;;
    esac
fi

SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/tmp/swegym_deps/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/tmp/swegym_deps/Megatron-LM}"
TRAIN_SQSH="${TRAIN_SQSH:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/docker/polr_swegym_qwen35_torch211_te2161_fa4b19_numpy126_scipy117_tebindcu130_20260707.sqsh}"

INSTANCE_RANGE="${INSTANCE_RANGE:-}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -z "${INSTANCE_RANGE}" ]; then
    shard_start=$((SLURM_ARRAY_TASK_ID * SHARD_SIZE + 1))
    shard_end=$((shard_start + SHARD_SIZE - 1))
    if [ "${shard_end}" -gt "${TOTAL_DATASET_INSTANCES}" ]; then
        shard_end="${TOTAL_DATASET_INSTANCES}"
    fi
    INSTANCE_RANGE="${shard_start}-${shard_end}"
fi
MAX_TASKS="${MAX_TASKS:--1}"
RESUME_COMPLETED="${RESUME_COMPLETED:-0}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
STRICT_ONE_SHOT="${STRICT_ONE_SHOT:-1}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
if [ "${STRICT_ONE_SHOT}" = "1" ]; then
    RESUME_COMPLETED=0
    [ "${N_SAMPLES_PER_EVAL_PROMPT}" = "1" ] || die "strict pass@1 requires N_SAMPLES_PER_EVAL_PROMPT=1"
fi
if [ "${MODEL_SOURCE}" = "base" ]; then
    MODEL_TAG="base"
else
    MODEL_TAG="checkpoint_${CHECKPOINT_TAG}"
fi
ARRAY_TAG="${SLURM_ARRAY_TASK_ID:-single}"
RANGE_TAG="$(printf '%s' "${INSTANCE_RANGE:-all}" | tr -c 'A-Za-z0-9_-' '_')"
DEFAULT_RUN_ID="${RUN_GROUP}_${MODEL_TAG}_range${RANGE_TAG}_job${SLURM_JOB_ID:-manual}_${ARRAY_TAG}"
RUN_ID="${RUN_ID:-${DEFAULT_RUN_ID}}"
RUN_TMP_ID="${RUN_TMP_ID:-${SLURM_JOB_ID:-$$}-$(date +%H%M%S)}"
RUN_DIR="$(abs_path "${RUN_DIR:-${SCRIPT_DIR}/results/${RUN_ID}}")"
RUN_LOG_DIR="${RUN_DIR}/logs"
ROLLOUT_SAVE_DIR="${RUN_DIR}/rollout_results"
EVAL_DATA="${RUN_DIR}/swebench_verified_eval.jsonl"
MANIFEST_DATA="${RUN_DIR}/swebench_verified_selected.jsonl"
SIF_DIR="${RUN_DIR}/apptainer_images"
CHECKPOINT_VIEW_DIR="${RUN_DIR}/checkpoint_view"
CUSTOM_CONFIG_PATH="${RUN_DIR}/polar_config.yaml"
TOPOLOGY_PATH="${RUN_DIR}/topology.yaml"
RUNTIME_ENV_PATH="${RUN_DIR}/ray_runtime_env.yaml"
SUMMARY_JSON="${RUN_DIR}/summary.json"

DATASET_CACHE="${DATASET_CACHE:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/polar_3/ProRL-Agent-Server/examples/swebench_verified/data/swebench_verified.json}"
SHARED_SIF_DIR="${SHARED_SIF_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/singularity_images_v3}"
AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/tmp/swegym_agent_cli/opt_node}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-${SCRIPT_DIR}/prepare_apptainer_eval.py}"
POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config_apptainer_pi.yaml}"
TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.sgl.yaml}"

# ---------------------------------------------------------------------------
# Runtime sizing
# ---------------------------------------------------------------------------
VISIBLE_GPU_COUNT="${SLURM_GPUS_ON_NODE:-}"
if [ -z "${VISIBLE_GPU_COUNT}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
fi
TOTAL_GPUS="${TOTAL_GPUS:-${VISIBLE_GPU_COUNT:-8}}"
EVAL_COLOCATE="${EVAL_COLOCATE:-0}"
DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
DEFAULT_ACTOR_GPUS=4
DEFAULT_TENSOR_MODEL_PARALLEL_SIZE=4
DEFAULT_ROLLOUT_GPUS="$((TOTAL_GPUS - DEFAULT_ACTOR_GPUS))"
if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ]; then
    DEFAULT_ACTOR_GPUS=0
    DEFAULT_ROLLOUT_GPUS="${TOTAL_GPUS}"
    DEFAULT_TENSOR_MODEL_PARALLEL_SIZE="${TOTAL_GPUS}"
    EVAL_COLOCATE=0
elif [ "${EVAL_COLOCATE}" = "1" ]; then
    DEFAULT_ACTOR_GPUS="${TOTAL_GPUS}"
    DEFAULT_ROLLOUT_GPUS="${TOTAL_GPUS}"
    DEFAULT_TENSOR_MODEL_PARALLEL_SIZE="${TOTAL_GPUS}"
elif [ "${TOTAL_GPUS}" -le 4 ]; then
    DEFAULT_ACTOR_GPUS=2
    DEFAULT_ROLLOUT_GPUS="$((TOTAL_GPUS - DEFAULT_ACTOR_GPUS))"
    DEFAULT_TENSOR_MODEL_PARALLEL_SIZE=2
fi
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${DEFAULT_ACTOR_GPUS}}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${DEFAULT_ROLLOUT_GPUS}}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-${DEFAULT_TENSOR_MODEL_PARALLEL_SIZE}}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-60000}"
TRAIN_LR="${TRAIN_LR:-1e-6}"
MIN_LR="${MIN_LR:-0.0}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-0}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-1}"
USE_KL_LOSS="${USE_KL_LOSS:-0}"

ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16000}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-32000}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN}}"
EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-${ROLLOUT_MAX_PROMPT_LEN}}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-60000}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-1}"
SGLANG_ROUTER_POLICY="${SGLANG_ROUTER_POLICY:-round_robin}"

PORT_OFFSET="${PORT_OFFSET:-$(( (${SLURM_JOB_ID:-0} + ${SLURM_PROCID:-0}) % 1000 ))}"
RAY_PORT="${RAY_PORT:-$((24000 + PORT_OFFSET))}"
ROLLOUT_PORT="${ROLLOUT_PORT:-$((25000 + PORT_OFFSET))}"
GATEWAY_PORT="${GATEWAY_PORT:-$((26000 + PORT_OFFSET))}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-$((27000 + PORT_OFFSET))}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-$((28000 + PORT_OFFSET))}"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-$(detect_host_ip)}"
SGLANG_ROUTER_BASE_URL="${SGLANG_ROUTER_BASE_URL:-http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}}"
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT}}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/polar-ray-eval-${RUN_TMP_ID}}"
JOB_CACHE_ROOT="${JOB_CACHE_ROOT:-/tmp/polar-eval-${RUN_TMP_ID}}"

GATEWAY_MAX_WORKERS="${GATEWAY_MAX_WORKERS:-32}"
POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-4}"
POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-3900}"
TASK_TIMEOUT_SECONDS="${TASK_TIMEOUT_SECONDS:-3600}"
AGENT_MODEL_NAME="${AGENT_MODEL_NAME:-${MODEL_NAME}}"
PI_MODEL_NAME="${PI_MODEL_NAME:-openai/${MODEL_NAME}}"
PI_API_TYPE="${PI_API_TYPE:-openai-completions}"
PI_CONTEXT_WINDOW="${PI_CONTEXT_WINDOW:-24000}"
PI_MAX_TOKENS="${PI_MAX_TOKENS:-512}"
PI_THINKING="${PI_THINKING:-}"
CODEX_VERSION="${CODEX_VERSION:-0.145.0}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-xhigh}"
CODEX_REASONING_SUMMARY="${CODEX_REASONING_SUMMARY:-}"
CLAUDE_MAX_TURNS="${CLAUDE_MAX_TURNS:-}"
CLAUDE_MAX_THINKING_TOKENS="${CLAUDE_MAX_THINKING_TOKENS:-}"
QWEN_CODE_MAX_OUTPUT_TOKENS="${QWEN_CODE_MAX_OUTPUT_TOKENS:-16000}"
POLAR_RUNTIME_MEMORY_MB="${POLAR_RUNTIME_MEMORY_MB:-}"
POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-${PROJECT_ROOT}/tmp/apptainer-v1.5.2-pyxis-fix/bin/apptainer}"
POLAR_APPTAINER_DIRECT_EXEC="${POLAR_APPTAINER_DIRECT_EXEC:-0}"
POLAR_APPTAINER_ISOLATE_PID="${POLAR_APPTAINER_ISOLATE_PID:-1}"
USE_TRAIN_SQSH="${USE_TRAIN_SQSH:-1}"
USE_SRUN_CONTAINER="${USE_SRUN_CONTAINER:-1}"

if [ -z "${MEGATRON_TO_HF_MODE:-}" ]; then
    if [ "${MODEL_SOURCE}" = "base" ] && [ ! -f "${BASE_LOAD}/latest_checkpointed_iteration.txt" ]; then
        MEGATRON_TO_HF_MODE="bridge"
    else
        MEGATRON_TO_HF_MODE="raw"
    fi
fi

PYTHON_BIN="${PYTHON_BIN:-/opt/polr_venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

preflight() {
    [ -d "${PROJECT_ROOT}" ] || die "PROJECT_ROOT not found: ${PROJECT_ROOT}"
    [ -d "${TRAIN_ROOT}" ] || die "TRAIN_ROOT not found: ${TRAIN_ROOT}"
    [ -f "${PREPARE_SCRIPT}" ] || die "prepare_apptainer_eval.py not found: ${PREPARE_SCRIPT}"
    [ -f "${TRAIN_SQSH}" ] || die "TRAIN_SQSH not found: ${TRAIN_SQSH}"
    [ -d "${SLIME_DIR}" ] || die "SLIME_DIR not found: ${SLIME_DIR}"
    [ -d "${MEGATRON_DIR}" ] || die "MEGATRON_DIR not found: ${MEGATRON_DIR}"
    [ -d "${HF_CHECKPOINT}" ] || die "HF_CHECKPOINT not found: ${HF_CHECKPOINT}"
    case "${MODEL_SOURCE}" in
        checkpoint)
            [ -d "${CHECKPOINT_DIR}" ] || die "CHECKPOINT_DIR not found: ${CHECKPOINT_DIR}"
            ;;
        base)
            [ -d "${BASE_LOAD}" ] || die "BASE_LOAD not found: ${BASE_LOAD}"
            if [ "${MEGATRON_TO_HF_MODE}" = "raw" ] && [ ! -f "${BASE_LOAD}/latest_checkpointed_iteration.txt" ]; then
                die "BASE_LOAD must be a Megatron checkpoint for raw mode: ${BASE_LOAD}"
            fi
            ;;
        *)
            die "MODEL_SOURCE must be checkpoint or base, got: ${MODEL_SOURCE}"
            ;;
    esac
    if [ "${USE_KL_LOSS}" = "1" ]; then
        [ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ] || die "REF_LOAD marker missing: ${REF_LOAD}"
    fi
    [ -f "${DATASET_CACHE}" ] || die "DATASET_CACHE not found: ${DATASET_CACHE}"
    [ -d "${SHARED_SIF_DIR}" ] || die "SHARED_SIF_DIR not found: ${SHARED_SIF_DIR}"
    [ -x "${AGENT_CLI_DIR}/bin/node" ] || die "Node.js not found under AGENT_CLI_DIR: ${AGENT_CLI_DIR}"
    case "${AGENT_HARNESS}" in
        pi) agent_bin="pi" ;;
        codex) agent_bin="codex" ;;
        claude_code) agent_bin="claude" ;;
        qwen_code) agent_bin="qwen" ;;
        *) die "unsupported harness: ${AGENT_HARNESS}" ;;
    esac
    [ -x "${AGENT_CLI_DIR}/bin/${agent_bin}" ] || \
        die "${AGENT_HARNESS} CLI not found under AGENT_CLI_DIR: ${AGENT_CLI_DIR}/bin/${agent_bin}"
    [ -x "${POLAR_APPTAINER_BIN}" ] || die "patched Apptainer not found: ${POLAR_APPTAINER_BIN}"
    [ "${SLURM_JOB_PARTITION:-interactive}" = "interactive" ] || die "eval is restricted to the interactive partition"
    [ "${SLURM_JOB_NUM_NODES:-1}" -le 1 ] || die "each eval shard must use exactly one node"
    [ "${TOTAL_GPUS}" -ge 2 ] || die "TOTAL_GPUS must be at least 2"
    if [ "${EVAL_COLOCATE}" = "1" ]; then
        [ "$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))" -le "${TOTAL_GPUS}" ] || \
            die "actor GPUs exceed TOTAL_GPUS in colocate mode"
        [ "${ROLLOUT_NUM_GPUS}" -le "${TOTAL_GPUS}" ] || \
            die "rollout GPUs exceed TOTAL_GPUS in colocate mode"
    else
        [ "$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))" -le "${TOTAL_GPUS}" ] || \
            die "actor + rollout GPUs exceed TOTAL_GPUS"
    fi
    [ "$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE % TENSOR_MODEL_PARALLEL_SIZE))" -eq 0 ] || \
        die "actor GPU count must be divisible by TENSOR_MODEL_PARALLEL_SIZE"
}

maybe_reexec_in_sqsh() {
    if [ "${USE_TRAIN_SQSH}" != "1" ] || [ "${INSIDE_EVAL_SQSH:-0}" = "1" ]; then
        return
    fi
    mkdir -p "${PROJECT_ROOT}/logs/slurm" "${RUN_LOG_DIR}"
    local mounts="/lustre/fs1:/lustre/fs1"
    if [ -d /lustre/fsw ]; then
        mounts="${mounts},/lustre/fsw:/lustre/fsw"
    fi
    if [ "${USE_SRUN_CONTAINER}" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
        exec srun \
            --overlap \
            --nodes="${SLURM_JOB_NUM_NODES:-1}" \
            --ntasks=1 \
            --ntasks-per-node=1 \
            --gpus-per-node="${TOTAL_GPUS}" \
            --container-image="${TRAIN_SQSH}" \
            --container-mounts="${mounts}" \
            --container-workdir="${PROJECT_ROOT}" \
            --container-writable \
            --no-container-mount-home \
            env INSIDE_EVAL_SQSH=1 USE_TRAIN_SQSH=0 bash "${SCRIPT_PATH}"
    fi
    command -v apptainer >/dev/null 2>&1 || die "apptainer not found for direct interactive execution"
    exec apptainer exec --nv --no-home --pwd "${PROJECT_ROOT}" \
        --bind /lustre/fs1:/lustre/fs1 \
        ${APPTAINER_EXTRA_BINDS:-} \
        --env "INSIDE_EVAL_SQSH=1" \
        --env "USE_TRAIN_SQSH=0" \
        --env "PATH=/opt/polr_venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        --env "PYTHONNOUSERSITE=1" \
        "${TRAIN_SQSH}" \
        bash "${SCRIPT_PATH}"
}

make_checkpoint_view() {
    if [ "${MODEL_SOURCE}" = "base" ]; then
        LOAD_DIR="${BASE_LOAD}"
        LOAD_LABEL="base:${BASE_LOAD}"
        return
    fi
    if [ -z "${CHECKPOINT_STEP}" ]; then
        LOAD_DIR="${CHECKPOINT_DIR}"
        LOAD_LABEL="checkpoint:${CHECKPOINT_DIR}"
        return
    fi
    local iter_name
    iter_name="$(printf 'iter_%07d' "${CHECKPOINT_STEP}")"
    [ -d "${CHECKPOINT_DIR}/${iter_name}" ] || die "Checkpoint step not found: ${CHECKPOINT_DIR}/${iter_name}"
    mkdir -p "${CHECKPOINT_VIEW_DIR}"
    ln -sfn "${CHECKPOINT_DIR}/${iter_name}" "${CHECKPOINT_VIEW_DIR}/${iter_name}"
    printf '%s\n' "${CHECKPOINT_STEP}" >"${CHECKPOINT_VIEW_DIR}/latest_checkpointed_iteration.txt"
    LOAD_DIR="${CHECKPOINT_VIEW_DIR}"
    LOAD_LABEL="checkpoint:${CHECKPOINT_DIR}/${iter_name}"
}

prepare_eval_assets() {
    local prepare_args=(
        --output-jsonl "${EVAL_DATA}" \
        --manifest-jsonl "${MANIFEST_DATA}" \
        --sif-dir "${SIF_DIR}" \
        --shared-sif-dir "${SHARED_SIF_DIR}" \
        --cache-path "${DATASET_CACHE}" \
        --max-tasks "${MAX_TASKS}"
    )
    if [ -n "${INSTANCE_RANGE}" ]; then
        prepare_args+=(--instance-range "${INSTANCE_RANGE}")
    fi
    if [ "${RESUME_COMPLETED}" = "1" ]; then
        prepare_args+=(
            --resume-completed
            --rollout-dir "${ROLLOUT_SAVE_DIR}"
            --completed-sessions-needed "${N_SAMPLES_PER_EVAL_PROMPT}"
        )
    fi
    "${PYTHON_BIN}" "${PREPARE_SCRIPT}" "${prepare_args[@]}"
    SELECTED_TASKS="$(count_jsonl_rows "${MANIFEST_DATA}")"
    REMAINING_TASKS="$(count_jsonl_rows "${EVAL_DATA}")"
}

render_configs() {
    "${PYTHON_BIN}" - "${TOPOLOGY_TEMPLATE}" "${TOPOLOGY_PATH}" \
        "${POLAR_CONFIG_TEMPLATE}" "${CUSTOM_CONFIG_PATH}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

topology_in, topology_out, config_in, config_out = sys.argv[1:]

with open(topology_in, encoding="utf-8") as handle:
    topology = yaml.safe_load(handle) or {}

topology["rollout"]["port"] = int(os.environ["ROLLOUT_PORT"])
topology["rollout"]["public_url"] = f"http://127.0.0.1:{os.environ['ROLLOUT_PORT']}"
topology["rollout"]["save_dir"] = os.environ["ROLLOUT_SAVE_DIR"]
nodes = topology.get("gateway", {}).get("nodes", [])
if not nodes:
    raise SystemExit("topology has no gateway nodes")
del nodes[1:]
node = nodes[0]
node["id"] = "localhost-node-01"
node["port"] = int(os.environ["GATEWAY_PORT"])
node["public_url"] = f"http://127.0.0.1:{os.environ['GATEWAY_PORT']}"
node["model_served"] = os.environ["MODEL_NAME"]
node["max_init_workers"] = int(os.environ["GATEWAY_MAX_WORKERS"])
node["max_run_workers"] = int(os.environ["GATEWAY_MAX_WORKERS"])
node["max_postrun_workers"] = int(os.environ["GATEWAY_MAX_WORKERS"])
node.pop("sglang", None)
inference = node.setdefault("inference", {})
inference["engine"] = "sglang"
inference["base_url"] = os.environ["SGLANG_ROUTER_BASE_URL"]

Path(topology_out).parent.mkdir(parents=True, exist_ok=True)
with open(topology_out, "w", encoding="utf-8") as handle:
    yaml.safe_dump(topology, handle, sort_keys=False)

with open(config_in, encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

config["polar_rollout_url"] = f"http://127.0.0.1:{os.environ['ROLLOUT_PORT']}"
config["polar_gateway_url"] = f"http://127.0.0.1:{os.environ['GATEWAY_PORT']}"
config["polar_agent_cli_dir"] = os.environ["AGENT_CLI_DIR"]
config["polar_apptainer_image_dir"] = os.environ["SIF_DIR"]
config["polar_max_async_level"] = int(os.environ["POLAR_MAX_ASYNC_LEVEL"])
config["polar_request_timeout"] = float(os.environ["POLAR_REQUEST_TIMEOUT"])
task = config.setdefault("polar_task_template", {})
task["timeout_seconds"] = int(os.environ["TASK_TIMEOUT_SECONDS"])
runtime = task.setdefault("runtime", {})
memory_mb = os.environ.get("POLAR_RUNTIME_MEMORY_MB", "").strip()
if memory_mb:
    memory_value = int(memory_mb)
    if memory_value <= 0:
        raise SystemExit(f"POLAR_RUNTIME_MEMORY_MB must be positive when set: {memory_mb}")
    runtime["memory_mb"] = memory_value
else:
    runtime.pop("memory_mb", None)
agent = task.setdefault("agent", {})
harness = os.environ["AGENT_HARNESS"]
agent["harness"] = harness
agent["env"] = {}
if harness == "pi":
    agent["model_name"] = os.environ["PI_MODEL_NAME"]
    settings = agent.setdefault("settings", {})
    settings["api_type"] = os.environ["PI_API_TYPE"]
    settings["context_window"] = int(os.environ["PI_CONTEXT_WINDOW"])
    settings["max_tokens"] = int(os.environ["PI_MAX_TOKENS"])
    if os.environ.get("PI_THINKING", ""):
        settings["thinking"] = os.environ["PI_THINKING"]
else:
    agent["model_name"] = os.environ["AGENT_MODEL_NAME"]
    settings = {}
    agent["settings"] = settings
    runtime["prepare"] = [
        step for step in runtime.get("prepare", [])
        if ".pi/agent/settings.json" not in str(step.get("command", ""))
    ]
    if harness == "codex":
        settings["version"] = os.environ["CODEX_VERSION"]
        settings["reasoning_effort"] = os.environ["CODEX_REASONING_EFFORT"]
        if os.environ.get("CODEX_REASONING_SUMMARY", ""):
            settings["reasoning_summary"] = os.environ["CODEX_REASONING_SUMMARY"]
    elif harness == "claude_code":
        if os.environ.get("CLAUDE_MAX_TURNS", ""):
            settings["max_turns"] = int(os.environ["CLAUDE_MAX_TURNS"])
        if os.environ.get("CLAUDE_MAX_THINKING_TOKENS", ""):
            settings["max_thinking_tokens"] = int(os.environ["CLAUDE_MAX_THINKING_TOKENS"])
    elif harness == "qwen_code":
        agent["env"]["QWEN_CODE_MAX_OUTPUT_TOKENS"] = os.environ["QWEN_CODE_MAX_OUTPUT_TOKENS"]
    else:
        raise SystemExit(f"unsupported harness: {harness}")

Path(config_out).parent.mkdir(parents=True, exist_ok=True)
with open(config_out, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
}

write_runtime_env() {
    "${PYTHON_BIN}" - "${RUNTIME_ENV_PATH}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
py_site = "/opt/polr_venv/lib/python3.12/site-packages"
ld_parts = [
    "/usr/local/cuda-13.0/compat",
    f"{py_site}/torch/lib",
    f"{py_site}/nvidia/cuda_runtime/lib",
    f"{py_site}/nvidia/cuda_nvrtc/lib",
    f"{py_site}/nvidia/nvjitlink/lib",
    f"{py_site}/nvidia/cublas/lib",
    f"{py_site}/nvidia/cudnn/lib",
    f"{py_site}/nvidia/nccl/lib",
    f"{py_site}/nvidia/cusparse/lib",
    f"{py_site}/nvidia/cusolver/lib",
    f"{py_site}/nvidia/cufft/lib",
    f"{py_site}/nvidia/curand/lib",
    os.environ.get("LD_LIBRARY_PATH", ""),
]
env = {
    "PYTHONPATH": os.environ["PYTHONPATH"],
    "PATH": os.environ["PATH"],
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "LD_LIBRARY_PATH": ":".join(part for part in ld_parts if part),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "APPTAINER_CACHEDIR": os.environ["APPTAINER_CACHEDIR"],
    "APPTAINER_TMPDIR": os.environ["APPTAINER_TMPDIR"],
    "SINGULARITY_CACHEDIR": os.environ["APPTAINER_CACHEDIR"],
    "SINGULARITY_TMPDIR": os.environ["APPTAINER_TMPDIR"],
    "POLAR_APPTAINER_BIN": os.environ["POLAR_APPTAINER_BIN"],
    "POLAR_APPTAINER_DIRECT_EXEC": os.environ["POLAR_APPTAINER_DIRECT_EXEC"],
    "POLAR_APPTAINER_ISOLATE_PID": os.environ["POLAR_APPTAINER_ISOLATE_PID"],
    "HF_HOME": os.environ["HF_HOME"],
    "HUGGINGFACE_HUB_CACHE": os.environ["HUGGINGFACE_HUB_CACHE"],
    "TRANSFORMERS_CACHE": os.environ["TRANSFORMERS_CACHE"],
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(yaml.safe_dump({"env_vars": env}, sort_keys=True))
PY
}

start_services() {
    PIDS=()
    "${PYTHON_BIN}" -m polar.cli serve_rollout -c "${TOPOLOGY_PATH}" >"${RUN_LOG_DIR}/polar-rollout.log" 2>&1 &
    PIDS+=("$!")
    sleep 2
    "${PYTHON_BIN}" -m polar.cli serve_gateway -c "${TOPOLOGY_PATH}" --node-id localhost-node-01 >"${RUN_LOG_DIR}/polar-gateway.log" 2>&1 &
    PIDS+=("$!")

    for _ in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:${ROLLOUT_PORT}/health" >/dev/null; then
            return
        fi
        sleep 2
    done
    die "Polar rollout server did not become healthy"
}

start_ray() {
    if [ "${EVAL_RAY_STOP_BEFORE_START:-0}" = "1" ]; then
        ray stop --force >/dev/null 2>&1 || true
        sleep 1
    fi
    ray start --head \
        --node-ip-address=127.0.0.1 \
        --port="${RAY_PORT}" \
        --num-gpus="${TOTAL_GPUS}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT}" \
        --dashboard-agent-listen-port=0 \
        --dashboard-agent-grpc-port=0 \
        --runtime-env-agent-port=0 \
        --temp-dir="${RAY_TMPDIR}" \
        --disable-usage-stats >"${RUN_LOG_DIR}/ray-head.log" 2>&1

    for attempt in $(seq 1 60); do
        if ray job submit --address="${RAY_JOB_ADDRESS}" --no-wait \
            -- python -c 'print("ray job server ready")' \
            >"${RUN_LOG_DIR}/ray-job-readiness.log" 2>&1; then
            return
        fi
        printf '[%s] waiting for Ray job server readiness (%s/60)\n' \
            "$(date +'%F %T')" "${attempt}" >>"${RUN_LOG_DIR}/ray-job-readiness.log"
        sleep 2
    done
    die "Ray job server did not become ready at ${RAY_JOB_ADDRESS}"
}

run_eval_job() {
    local model_args=(
        --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
        --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
        --disable-bias-linear
        --qk-layernorm
        --group-query-attention
        --num-attention-heads 16
        --num-query-groups 4
        --kv-channels 256
        --num-layers 32
        --hidden-size 2560
        --ffn-hidden-size 9216
        --use-gated-attention
        --normalization RMSNorm
        --apply-layernorm-1p
        --position-embedding-type rope
        --norm-epsilon 1e-6
        --rotary-percent 0.25
        --swiglu
        --vocab-size 248320
        --rotary-base 10000000
        --attention-output-gate
    )
    local load_args=(
        --load "${LOAD_DIR}"
        --no-load-optim
        --no-load-rng
        --dist-ckpt-strictness "${DIST_CKPT_STRICTNESS:-log_all}"
    )
    local ref_args=()
    local kl_args=()
    local mode_args=(
        --num-gpus-per-node "${TOTAL_GPUS}"
    )
    local sglang_extra_args=()
    if [ "${USE_KL_LOSS}" = "1" ]; then
        ref_args=(--ref-load "${REF_LOAD}")
        kl_args=(
            --use-kl-loss
            --kl-loss-coef "${KL_LOSS_COEF:-0.001}"
            --kl-loss-type low_var_kl
        )
    fi
    if [ "${EVAL_COLOCATE}" = "1" ]; then
        mode_args+=(--colocate)
    fi
    if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ]; then
        mode_args+=(--debug-rollout-only)
    fi
    if [ "${SGLANG_DISABLE_CUDA_GRAPH}" = "1" ]; then
        sglang_extra_args+=(--sglang-disable-cuda-graph)
    fi

    local submission_id
    submission_id="${RAY_JOB_SUBMISSION_ID:-${RUN_ID}}"
    submission_id="$(printf '%s' "${submission_id}" | tr -c 'A-Za-z0-9_-' '_' | cut -c1-200)"

    set +e
    ray job submit --address="${RAY_JOB_ADDRESS}" \
        --runtime-env="${RUNTIME_ENV_PATH}" \
        --submission-id "${submission_id}" \
        --no-wait \
        -- "${PYTHON_BIN}" "${SLIME_DIR}/train.py" \
        --actor-num-nodes "${ACTOR_NUM_NODES}" \
        --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
        --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
        --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
        "${mode_args[@]}" \
        "${model_args[@]}" \
        --hf-checkpoint "${HF_CHECKPOINT}" \
        --tokenizer-model "${HF_CHECKPOINT}" \
        --tokenizer-type HuggingFaceTokenizer \
        --no-use-tokenizer-model-from-checkpoint-args \
        "${ref_args[@]}" \
        "${load_args[@]}" \
        --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async \
        --eval-function-path slime_bridge.rollout.generate_rollout_polar_async \
        --custom-eval-rollout-log-function-path swebench_eval_hooks.log_eval_rollout \
        --custom-rm-path slime_bridge.reward.reward_func \
        --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards \
        --custom-config-path "${CUSTOM_CONFIG_PATH}" \
        --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer \
        --prompt-data "${EVAL_DATA}" \
        --input-key prompt \
        --label-key label \
        --metadata-key metadata \
        --reward-key score \
        --num-rollout 0 \
        --eval-interval 1 \
        --eval-prompt-data swebench_verified "${EVAL_DATA}" \
        --eval-input-key prompt \
        --eval-label-key label \
        --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}" \
        --eval-temperature 1.0 \
        --eval-top-p 1.0 \
        --eval-top-k -1 \
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}" \
        --eval-max-prompt-len "${EVAL_MAX_PROMPT_LEN}" \
        --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
        --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
        --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
        --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}" \
        --dynamic-history \
        --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}" \
        --qwen-gdn-backend fla \
        --qkv-format thd \
        --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
        --sequence-parallel \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --expert-tensor-parallel-size 1 \
        --recompute-granularity full \
        --recompute-method uniform \
        --recompute-num-layers 1 \
        --use-dynamic-batch-size \
        --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
        --log-probs-chunk-size 256 \
        --advantage-estimator grpo \
        --normalize-advantages \
        "${kl_args[@]}" \
        --entropy-coef 0.0 \
        --eps-clip "${EPS_CLIP:-0.2}" \
        --eps-clip-high "${EPS_CLIP_HIGH:-0.28}" \
        --optimizer adam \
        --lr "${TRAIN_LR}" \
        --min-lr "${MIN_LR}" \
        --lr-warmup-iters "${LR_WARMUP_ITERS}" \
        --lr-decay-iters "${LR_DECAY_ITERS}" \
        --lr-decay-style constant \
        --weight-decay 0.1 \
        --clip-grad "${CLIP_GRAD:-1.0}" \
        --adam-beta1 0.9 \
        --adam-beta2 0.98 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --accumulate-allreduce-grads-in-fp32 \
        --attention-softmax-in-fp32 \
        --attention-backend flash \
        --no-gradient-accumulation-fusion \
        --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
        --sglang-context-length "${SGLANG_CONTEXT_LENGTH}" \
        "${sglang_extra_args[@]}" \
        --sglang-served-model-name "${MODEL_NAME}" \
        --sglang-tool-call-parser qwen3_coder \
        --router-policy "${SGLANG_ROUTER_POLICY}" \
        --sglang-router-port "${SGLANG_ROUTER_PORT}" \
        2>&1 | tee "${RUN_LOG_DIR}/ray-job-submit.log"
    local rc="${PIPESTATUS[0]}"
    set -e
    if [ "${rc}" -ne 0 ]; then
        return "${rc}"
    fi

    local status_out status_rc
    local poll_interval="${RAY_JOB_POLL_INTERVAL:-30}"
    local max_polls="${RAY_JOB_MAX_POLLS:-480}"
    local max_status_failures="${RAY_JOB_STATUS_MAX_FAILURES:-6}"
    local status_failures=0
    for attempt in $(seq 1 "${max_polls}"); do
        set +e
        status_out="$(ray job status --address="${RAY_JOB_ADDRESS}" "${submission_id}" 2>&1)"
        status_rc="$?"
        set -e
        printf '[%s] poll=%s/%s rc=%s %s\n' \
            "$(date +'%F %T')" "${attempt}" "${max_polls}" "${status_rc}" "${status_out}" \
            | tee -a "${RUN_LOG_DIR}/ray-job-status.log"
        if [ "${status_rc}" -ne 0 ]; then
            status_failures=$((status_failures + 1))
            if [ "${status_failures}" -ge "${max_status_failures}" ]; then
                printf 'Ray job status failed %s consecutive times for %s at %s\n' \
                    "${status_failures}" "${submission_id}" "${RAY_JOB_ADDRESS}" \
                    | tee -a "${RUN_LOG_DIR}/ray-job-status.log"
                return 1
            fi
            sleep "${poll_interval}"
            continue
        fi
        status_failures=0
        if printf '%s\n' "${status_out}" | grep -Eiq 'SUCCEEDED|succeeded'; then
            return 0
        fi
        if printf '%s\n' "${status_out}" | grep -Eiq 'FAILED|STOPPED|failed|stopped'; then
            ray job logs --address="${RAY_JOB_ADDRESS}" "${submission_id}" \
                >"${RUN_LOG_DIR}/ray-job-final.log" 2>&1 || true
            return 1
        fi
        sleep "${poll_interval}"
    done
    printf 'Timed out waiting for Ray job %s at %s\n' "${submission_id}" "${RAY_JOB_ADDRESS}" \
        | tee -a "${RUN_LOG_DIR}/ray-job-status.log"
    return 1
}

summarize_results() {
    "${PYTHON_BIN}" - "${MANIFEST_DATA}" "${ROLLOUT_SAVE_DIR}" "${SUMMARY_JSON}" "${RUN_DIR}" <<'PY' || true
import csv
import json
import sys
import time
from pathlib import Path

manifest_path = Path(sys.argv[1])
rollout_dir = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
run_dir = Path(sys.argv[4])


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise SystemExit(f"failed to parse {path}:{line_no}: {exc}") from exc
    return rows


def instance_id_from_row(row: dict) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("instance_id"):
        return str(metadata["instance_id"])
    instance = metadata.get("instance") or {}
    if instance.get("instance_id"):
        return str(instance["instance_id"])
    raise SystemExit(f"manifest row missing metadata.instance_id: {row.keys()}")


def dataset_index_from_row(row: dict, fallback: int) -> int:
    metadata = row.get("metadata") or {}
    try:
        return int(metadata.get("dataset_index"))
    except Exception:
        return fallback


def safe_load_session(path: Path) -> tuple[dict | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, str(exc)


def collect_sessions() -> dict[str, list[Path]]:
    by_instance: dict[str, list[Path]] = {iid: [] for iid in instance_ids}
    if not rollout_dir.exists():
        return by_instance
    for path in rollout_dir.rglob("ses_*.json"):
        path_text = path.as_posix()
        for iid in instance_ids:
            if iid in path_text:
                by_instance[iid].append(path)
                break
    for paths in by_instance.values():
        paths.sort(key=lambda p: (p.stat().st_mtime_ns, p.as_posix()))
    return by_instance


def reward_from_session(session: dict) -> float | None:
    trajectory = session.get("trajectory") or {}
    metadata = trajectory.get("metadata") or {}
    evaluation = metadata.get("evaluation") or {}
    if evaluation.get("outcome_reward") is not None:
        try:
            return float(evaluation["outcome_reward"])
        except Exception:
            return None
    rewards: list[float] = []
    for trace in trajectory.get("traces") or []:
        if isinstance(trace, dict) and trace.get("reward") is not None:
            try:
                rewards.append(float(trace["reward"]))
            except Exception:
                pass
    return max(rewards) if rewards else None


def classify_first_session(path: Path) -> dict:
    session, parse_error = safe_load_session(path)
    base = {
        "first_session_path": path.as_posix(),
        "parse_error": parse_error,
        "session_id": None,
        "task_id": None,
        "session_status": "PARSE_ERROR" if parse_error else None,
        "session_error": parse_error,
        "trajectory_status": None,
        "trajectory_error": None,
        "outcome_reward": None,
        "resolved": False,
        "clean_completed": False,
        "failure_kind": "PARSE_ERROR" if parse_error else "UNKNOWN",
    }
    if session is None:
        return base
    trajectory = session.get("trajectory") or {}
    metadata = trajectory.get("metadata") or {}
    evaluation = metadata.get("evaluation") or {}
    report = evaluation.get("report") or {}
    session_status = str(session.get("status") or "UNKNOWN").upper()
    trajectory_status = str(trajectory.get("status") or "").upper() or None
    session_error = session.get("error")
    trajectory_error = trajectory.get("error")
    error_text = " ".join(
        str(part)
        for part in [session_status, trajectory_status, session_error, trajectory_error]
        if part
    ).lower()
    reward = reward_from_session(session)
    clean_completed = (
        session_status == "COMPLETED"
        and not session_error
        and trajectory_status not in {"ERROR", "FAILED", "TIMEOUT"}
        and not trajectory_error
    )
    resolved = bool(clean_completed and reward == 1.0)
    if resolved:
        failure_kind = "RESOLVED"
    elif "timeout" in error_text or report.get("test_timeout"):
        failure_kind = "TIMEOUT"
    elif not clean_completed:
        failure_kind = session_status if session_status not in {"", "UNKNOWN"} else "ERROR"
    else:
        failure_kind = "UNRESOLVED"
    base.update(
        {
            "session_id": session.get("session_id"),
            "task_id": session.get("task_id"),
            "session_status": session_status,
            "session_error": session_error,
            "trajectory_status": trajectory_status,
            "trajectory_error": trajectory_error,
            "outcome_reward": reward,
            "resolved": resolved,
            "clean_completed": clean_completed,
            "failure_kind": failure_kind,
        }
    )
    return base


manifest = load_jsonl(manifest_path)
instance_ids = [instance_id_from_row(row) for row in manifest]
session_paths = collect_sessions()
results = []
for ordinal, row in enumerate(manifest, 1):
    iid = instance_id_from_row(row)
    paths = session_paths.get(iid, [])
    dataset_index = dataset_index_from_row(row, ordinal - 1)
    if paths:
        item = classify_first_session(paths[0])
    else:
        item = {
            "first_session_path": None,
            "parse_error": None,
            "session_id": None,
            "task_id": None,
            "session_status": "MISSING",
            "session_error": None,
            "trajectory_status": None,
            "trajectory_error": None,
            "outcome_reward": None,
            "resolved": False,
            "clean_completed": False,
            "failure_kind": "MISSING",
        }
    item.update(
        {
            "ordinal": ordinal,
            "dataset_index": dataset_index,
            "instance_id": iid,
            "session_count": len(paths),
            "extra_session_paths": [p.as_posix() for p in paths[1:]],
            "score": 1.0 if item["resolved"] else 0.0,
        }
    )
    results.append(item)

total = len(results)
resolved = sum(1 for item in results if item["resolved"])
attempted = sum(1 for item in results if item["session_count"] > 0)
clean_completed = sum(1 for item in results if item["clean_completed"])
timeouts = sum(1 for item in results if item["failure_kind"] == "TIMEOUT")
errors = sum(1 for item in results if item["failure_kind"] in {"ERROR", "FAILED", "PARSE_ERROR"} or (item["session_status"] not in {"COMPLETED", "MISSING"} and item["failure_kind"] != "TIMEOUT"))
missing = sum(1 for item in results if item["failure_kind"] == "MISSING")
duplicates = sum(1 for item in results if item["session_count"] > 1)
summary = {
    "metric": "strict_pass_at_1",
    "total_tasks": total,
    "attempted_tasks": attempted,
    "clean_completed_tasks": clean_completed,
    "resolved_tasks": resolved,
    "pass_at_1": (resolved / total) if total else 0.0,
    "timeout_tasks": timeouts,
    "error_tasks": errors,
    "missing_tasks": missing,
    "duplicate_session_tasks": duplicates,
    "scoring_rule": "first ses_*.json per instance by mtime/path; resolved only when first session COMPLETED without session/trajectory error and outcome reward is 1.0; timeout/error/missing count as 0; extra sessions ignored and reported",
    "rollout_dir": rollout_dir.as_posix(),
    "generated_at_unix": time.time(),
}
payload = {"summary": summary, "instances": results}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

jsonl_path = run_dir / "strict_results.jsonl"
with jsonl_path.open("w", encoding="utf-8") as handle:
    for item in results:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")

csv_path = run_dir / "summary.csv"
fieldnames = [
    "ordinal",
    "dataset_index",
    "instance_id",
    "score",
    "resolved",
    "failure_kind",
    "session_count",
    "session_status",
    "outcome_reward",
    "first_session_path",
    "session_id",
    "task_id",
    "session_error",
    "trajectory_status",
    "trajectory_error",
]
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for item in results:
        writer.writerow({key: item.get(key) for key in fieldnames})

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"summary_json={summary_path}")
print(f"strict_results_jsonl={jsonl_path}")
print(f"summary_csv={csv_path}")
PY
}

PIDS=()

cleanup() {
    set +e
    if [ "${#PIDS[@]}" -gt 0 ]; then
        for pid in "${PIDS[@]}"; do
            kill "${pid}" 2>/dev/null || true
        done
    fi
    if [ "${EVAL_RAY_STOP_ON_CLEANUP:-0}" = "1" ]; then
        ray stop --force >/dev/null 2>&1 || true
    fi
}

preflight
maybe_reexec_in_sqsh

mkdir -p "${RUN_DIR}" "${RUN_LOG_DIR}" "${ROLLOUT_SAVE_DIR}" "${SIF_DIR}" \
    "${JOB_CACHE_ROOT}/home" "${JOB_CACHE_ROOT}/apptainer-cache" "${JOB_CACHE_ROOT}/apptainer-tmp" \
    "${JOB_CACHE_ROOT}/xdg-cache" "${JOB_CACHE_ROOT}/xdg-config" "${JOB_CACHE_ROOT}/xdg-runtime" \
    "${RAY_TMPDIR}" "${PROJECT_ROOT}/logs/slurm"
chmod 700 "${JOB_CACHE_ROOT}/home" "${JOB_CACHE_ROOT}/xdg-runtime" || true

export HOME="${JOB_CACHE_ROOT}/home"
export APPTAINER_CACHEDIR="${JOB_CACHE_ROOT}/apptainer-cache"
export APPTAINER_TMPDIR="${JOB_CACHE_ROOT}/apptainer-tmp"
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
export XDG_CACHE_HOME="${JOB_CACHE_ROOT}/xdg-cache"
export XDG_CONFIG_HOME="${JOB_CACHE_ROOT}/xdg-config"
export XDG_RUNTIME_DIR="${JOB_CACHE_ROOT}/xdg-runtime"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="/opt/polr_venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_ROOT}/src:${SLIME_DIR}:${MEGATRON_DIR}:${PYTHONPATH:-}"
export ROLLOUT_PORT GATEWAY_PORT ROLLOUT_SAVE_DIR MODEL_NAME SGLANG_ROUTER_BASE_URL
export GATEWAY_MAX_WORKERS POLAR_MAX_ASYNC_LEVEL POLAR_REQUEST_TIMEOUT TASK_TIMEOUT_SECONDS POLAR_RUNTIME_MEMORY_MB
export AGENT_CLI_DIR SIF_DIR AGENT_HARNESS AGENT_MODEL_NAME
export PI_MODEL_NAME PI_API_TYPE PI_CONTEXT_WINDOW PI_MAX_TOKENS PI_THINKING
export CODEX_VERSION CODEX_REASONING_EFFORT CODEX_REASONING_SUMMARY
export CLAUDE_MAX_TURNS CLAUDE_MAX_THINKING_TOKENS
export QWEN_CODE_MAX_OUTPUT_TOKENS
export APPTAINER_CACHEDIR APPTAINER_TMPDIR
export POLAR_APPTAINER_BIN POLAR_APPTAINER_DIRECT_EXEC POLAR_APPTAINER_ISOLATE_PID
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HG_Cache}"
export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"

if [ "${STRICT_ONE_SHOT}" = "1" ] && find "${ROLLOUT_SAVE_DIR}" -path '*/ses_*.json' -print -quit 2>/dev/null | grep -q .; then
    die "STRICT_ONE_SHOT refuses to reuse existing rollout sessions: ${ROLLOUT_SAVE_DIR}"
fi

make_checkpoint_view
prepare_eval_assets
if [ "${REMAINING_TASKS}" -eq 0 ]; then
    cat <<EOF
=============================================
SWE-bench Verified harness strict eval
  Run ID:        ${RUN_ID}
  Model source:  ${MODEL_SOURCE}
  Harness:       ${AGENT_HARNESS}
  Instance range:${INSTANCE_RANGE:-all}
  Selected:      ${SELECTED_TASKS}
  Remaining:     0
  Resume:        ${RESUME_COMPLETED}
  Strict:        ${STRICT_ONE_SHOT}
  Run dir:       ${RUN_DIR}
=============================================
All selected instances already have completed results. Writing summary only.
EOF
    summarize_results
    exit 0
fi
render_configs
write_runtime_env
if [ "${PREPARE_ONLY}" = "1" ]; then
    cat <<EOF
=============================================
SWE-bench Verified harness strict eval prepare-only
  Run ID:        ${RUN_ID}
  Model source:  ${MODEL_SOURCE}
  Harness:       ${AGENT_HARNESS}
  Instance range:${INSTANCE_RANGE:-all}
  Selected:      ${SELECTED_TASKS}
  Remaining:     ${REMAINING_TASKS}
  Strict:        ${STRICT_ONE_SHOT}
  Config:        ${CUSTOM_CONFIG_PATH}
  Topology:      ${TOPOLOGY_PATH}
  Runtime env:   ${RUNTIME_ENV_PATH}
  Run dir:       ${RUN_DIR}
=============================================
EOF
    exit 0
fi

cat <<EOF
=============================================
SWE-bench Verified harness strict eval
  Run ID:        ${RUN_ID}
  Model source:  ${MODEL_SOURCE}
  Harness:       ${AGENT_HARNESS}
  Instance range:${INSTANCE_RANGE:-all}
  Selected:      ${SELECTED_TASKS}
  Remaining:     ${REMAINING_TASKS}
  Resume:        ${RESUME_COMPLETED}
  Strict:        ${STRICT_ONE_SHOT}
  Checkpoint:    ${CHECKPOINT_DIR}
  Step:          ${CHECKPOINT_STEP:-<latest marker>}
  Load:          ${LOAD_LABEL}
  Load dir:      ${LOAD_DIR}
  HF checkpoint: ${HF_CHECKPOINT}
  Base load:     ${BASE_LOAD}
  Prepare script:${PREPARE_SCRIPT}
  Dataset cache: ${DATASET_CACHE}
  MCore/HF mode: ${MEGATRON_TO_HF_MODE}
  Use KL/ref:    ${USE_KL_LOSS}
  SIF dir:       ${SIF_DIR}
  Shared SIF dir:${SHARED_SIF_DIR}
  Agent CLI dir: ${AGENT_CLI_DIR}
  Agent model:    ${AGENT_MODEL_NAME}
  PI settings:   model=${PI_MODEL_NAME}, context=${PI_CONTEXT_WINDOW}, max_tokens=${PI_MAX_TOKENS}, api=${PI_API_TYPE}
  Codex settings:version=${CODEX_VERSION}, reasoning=${CODEX_REASONING_EFFORT}
  Qwen settings: max_output_tokens=${QWEN_CODE_MAX_OUTPUT_TOKENS}
  Sandbox mem:   ${POLAR_RUNTIME_MEMORY_MB:-<none>} MB per command
  Run dir:       ${RUN_DIR}
  Rollout dir:   ${ROLLOUT_SAVE_DIR}
  Ray address:   ${RAY_JOB_ADDRESS}
  Ray port:      ${RAY_PORT}
  SGLang router: ${SGLANG_ROUTER_BASE_URL}
  SGLang ctx/cg: context=${SGLANG_CONTEXT_LENGTH}, disable_cuda_graph=${SGLANG_DISABLE_CUDA_GRAPH}
  Ports:         rollout=${ROLLOUT_PORT}, gateway=${GATEWAY_PORT}, router=${SGLANG_ROUTER_PORT}, dashboard=${RAY_DASHBOARD_PORT}
  GPUs:          total=${TOTAL_GPUS}, actor=${ACTOR_NUM_GPUS_PER_NODE}, rollout=${ROLLOUT_NUM_GPUS}, TP=${TENSOR_MODEL_PARALLEL_SIZE}, colocate=${EVAL_COLOCATE}, debug_rollout_only=${DEBUG_ROLLOUT_ONLY}
=============================================
EOF

trap cleanup EXIT
start_services
start_ray
set +e
run_eval_job
JOB_RC="$?"
set -e
summarize_results
exit "${JOB_RC}"





