#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server"
SCRIPT_DIR="${PROJECT_ROOT}/examples/terminalbench"
SWE_LAUNCHER="${PROJECT_ROOT}/examples/swebench_verified/submit_swebench_pi_apptainer.sh"
export SLIME_DIR="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/example/slime"

: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
: "${INSTANCE_RANGE:?INSTANCE_RANGE is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"

export PROJECT_ROOT
export SCRIPT_MODE=run
export MODEL_SOURCE=checkpoint
export RUN_GROUP=terminalbench21_pi
export TOTAL_DATASET_INSTANCES=89
export MAX_TASKS=-1
export RESUME_COMPLETED=1
export STRICT_ONE_SHOT=0
export N_SAMPLES_PER_EVAL_PROMPT=1

export PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_terminalbench.py"
export POLAR_CONFIG_TEMPLATE="${SCRIPT_DIR}/polar_config_pi.yaml"
export DATASET_CACHE="${SCRIPT_DIR}/data/terminal-bench-2-1/tasks/dataset.toml"
export SHARED_SIF_DIR=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/terminal_bench_sif
# Training Slime restores the rollout cursor alongside the model checkpoint.
# The shared eval launcher creates RUN_DIR/checkpoint_view; expose the original
# read-only rollout state there so the exact training Slime can load iter checkpoints.
checkpoint_name="$(basename -- "${CHECKPOINT_PATH}")"
if [[ "${checkpoint_name}" == iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9] ]]; then
    checkpoint_parent="$(dirname -- "${CHECKPOINT_PATH}")"
    if [ -d "${checkpoint_parent}/rollout" ]; then
        mkdir -p "${RUN_DIR}/checkpoint_view"
        ln -sfn "${checkpoint_parent}/rollout" "${RUN_DIR}/checkpoint_view/rollout"
    fi
fi
export AGENT_CLI_DIR="${PROJECT_ROOT}/tmp/swegym_agent_cli/opt_node"

# Match the PI training setup. The explicit 1500-second session timeout is the
# user-requested evaluation override; the verifier still uses each task's own
# timeout but cannot exceed the remaining session budget.
export TASK_TIMEOUT_SECONDS=1500
export POLAR_REQUEST_TIMEOUT=1800
export POLAR_MAX_ASYNC_LEVEL=1
export PI_MODEL_NAME="openai/Qwen/Qwen3.5-4B"
export PI_API_TYPE="openai-completions"
export PI_CONTEXT_WINDOW=65536
export PI_MAX_TOKENS=16384
export EVAL_MAX_PROMPT_LEN=32000
export EVAL_MAX_RESPONSE_LEN=16384
export ROLLOUT_MAX_PROMPT_LEN=32000
export ROLLOUT_MAX_RESPONSE_LEN=16384
export SGLANG_CONTEXT_LENGTH=262144
export MAX_TOKENS_PER_GPU=67584
export SGLANG_MEM_FRACTION_STATIC=0.7
export SGLANG_DISABLE_CUDA_GRAPH=0

# Pull the selected official task image on the compute host before entering the
# outer training container. This keeps SIF creation independent of nested runtime support.
export POLAR_APPTAINER_BIN="${PROJECT_ROOT}/tmp/apptainer-v1.5.2-pyxis-fix/bin/apptainer"
export JOB_CACHE_ROOT="/tmp/tb21-eval-${SLURM_JOB_ID}"
export RAY_TMPDIR="/tmp/tb21-ray-${SLURM_JOB_ID}"
export APPTAINER_CACHEDIR="${JOB_CACHE_ROOT}/apptainer-cache"
export APPTAINER_TMPDIR="${JOB_CACHE_ROOT}/apptainer-tmp"
mkdir -p "${RUN_DIR}/prebuild_sif_links" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# NRT periodically removes stale top-level /tmp directories even when the
# processes using their contents are still alive. Match the proven TMax worker
# behavior by keeping only this job's local runtime roots fresh.
keepalive_pid=""
cleanup_keepalive() {
    if [ -n "${keepalive_pid}" ]; then
        kill "${keepalive_pid}" 2>/dev/null || true
        wait "${keepalive_pid}" 2>/dev/null || true
    fi
}
trap cleanup_keepalive EXIT
(
    while true; do
        for path in "${JOB_CACHE_ROOT}" "${RAY_TMPDIR}" \
                    "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"; do
            [ ! -d "${path}" ] || touch "${path}" 2>/dev/null || true
        done
        sleep 10
    done
) &
keepalive_pid="$!"

python3 "${PREPARE_SCRIPT}" \
    --output-jsonl "${RUN_DIR}/prebuild_eval.jsonl" \
    --manifest-jsonl "${RUN_DIR}/prebuild_selected.jsonl" \
    --sif-dir "${RUN_DIR}/prebuild_sif_links" \
    --shared-sif-dir "${SHARED_SIF_DIR}" \
    --instance-range "${INSTANCE_RANGE}"

set +e
bash "${SWE_LAUNCHER}" run --checkpoint "${CHECKPOINT_PATH}"
eval_rc="$?"
set -e

# A scored failure or timeout is a final pass@1=0, not a retry candidate. Only
# an interrupted task with no readable session remains unfinished on resume.
if python3 "${SCRIPT_DIR}/aggregate_results.py" shard-status "${RUN_DIR}"; then
    touch "${RUN_DIR}/.complete"
fi
python3 "${SCRIPT_DIR}/aggregate_results.py" aggregate "${RESULT_ROOT}" || true
exit "${eval_rc}"
