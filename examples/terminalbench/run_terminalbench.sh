#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server"
SCRIPT_DIR="${PROJECT_ROOT}/examples/terminalbench"
WORKER="${SCRIPT_DIR}/eval_worker.sh"
PREPARE="${SCRIPT_DIR}/prepare_terminalbench.py"
RESULTS_DIR="${SCRIPT_DIR}/results"

usage() {
    cat <<EOF
Usage:
  bash ${SCRIPT_DIR}/run_terminalbench.sh CHECKPOINT [submit|smoke|status]

Examples:
  bash ${SCRIPT_DIR}/run_terminalbench.sh /path/to/iter_0000080 smoke
  bash ${SCRIPT_DIR}/run_terminalbench.sh /path/to/iter_0000080
  bash ${SCRIPT_DIR}/run_terminalbench.sh /path/to/iter_0000080 status
EOF
}

[ "$#" -ge 1 ] || { usage >&2; exit 2; }
CHECKPOINT="$1"
MODE="${2:-submit}"
[ -d "${CHECKPOINT}" ] || { printf 'ERROR: checkpoint not found: %s\n' "${CHECKPOINT}" >&2; exit 1; }
case "${CHECKPOINT}" in
    *,*|*$'\n'*) printf 'ERROR: checkpoint path may not contain commas or newlines\n' >&2; exit 2 ;;
esac

CHECKPOINT_TAG="$(basename -- "${CHECKPOINT}")"
CHECKPOINT_TAG="$(printf '%s' "${CHECKPOINT_TAG}" | tr -c 'A-Za-z0-9_.-' '_')"
RESULT_ROOT="${RESULTS_DIR}/tb21_pi_${CHECKPOINT_TAG}"
SHARD_SIZE="${SHARD_SIZE:-6}"
TRIALS="${TRIALS:-3}"
TOTAL_TASKS=89
MAX_INTERACTIVE_NODES="${MAX_INTERACTIVE_NODES:-2}"
[ "${MAX_INTERACTIVE_NODES}" = "1" ] || [ "${MAX_INTERACTIVE_NODES}" = "2" ] || \
    { printf 'ERROR: MAX_INTERACTIVE_NODES must be 1 or 2\n' >&2; exit 2; }

python3 "${PREPARE}" --download-only
mkdir -p "${RESULT_ROOT}/slurm"

if [ "${MODE}" = "status" ]; then
    python3 "${SCRIPT_DIR}/aggregate_results.py" aggregate "${RESULT_ROOT}"
    exit 0
fi

submit_one() {
    local trial="$1" shard="$2" start="$3" end="$4" run_dir="$5" dependency="$6"
    local job_name="tb21-${CHECKPOINT_TAG}-t${trial}-s${shard}-r${start}-${end}"
    job_name="$(printf '%s' "${job_name}" | cut -c1-100)"

    if [ -f "${run_dir}/.complete" ]; then
        printf 'skip complete: trial=%s shard=%s range=%s-%s\n' "${trial}" "${shard}" "${start}" "${end}" >&2
        printf '%s\n' "${dependency}"
        return
    fi

    local queued
    queued="$(squeue -h -u "${USER}" -n "${job_name}" -o '%i' | head -n 1)"
    if [ -n "${queued}" ]; then
        printf 'skip queued: %s job=%s\n' "${job_name}" "${queued}" >&2
        printf '%s\n' "${queued}"
        return
    fi

    local dep_args=()
    if [ -n "${dependency}" ]; then
        dep_args=(--dependency="afterany:${dependency}")
    fi
    local job_id
    job_id="$(sbatch --parsable \
        --account=nvr_lpr_agentic \
        --partition=interactive \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gpus-per-node=8 \
        --time=03:55:00 \
        --job-name="${job_name}" \
        --output="${RESULT_ROOT}/slurm/%x-%j.out" \
        --error="${RESULT_ROOT}/slurm/%x-%j.err" \
        "${dep_args[@]}" \
        --export="ALL,CHECKPOINT_PATH=${CHECKPOINT},INSTANCE_RANGE=${start}-${end},RUN_DIR=${run_dir},RESULT_ROOT=${RESULT_ROOT},EVAL_TRIAL=${trial},EVAL_SHARD=${shard}" \
        "${WORKER}")"
    printf 'submitted: trial=%s shard=%s range=%s-%s job=%s\n' "${trial}" "${shard}" "${start}" "${end}" "${job_id}" >&2
    printf '%s\n' "${job_id}"
}

# Protect the requested node limit even when another same-user interactive
# allocation already exists. Shards are distributed over independent chains.
initial_dependency="$(squeue -h -u "${USER}" -p interactive -o '%i' | paste -sd: -)"
lane_dependencies=()
for ((lane = 0; lane < MAX_INTERACTIVE_NODES; lane++)); do
    lane_dependencies+=("${initial_dependency}")
done

case "${MODE}" in
    smoke)
        smoke_dir="${RESULT_ROOT}/smoke/task_030"
        lane_dependencies[0]="$(submit_one 0 1 30 30 "${smoke_dir}" "${lane_dependencies[0]}")"
        printf 'Smoke test queued as job %s. Result directory: %s\n' "${lane_dependencies[0]}" "${smoke_dir}"
        ;;
    submit)
        next_lane=0
        for trial in $(seq 1 "${TRIALS}"); do
            shard=0
            start=1
            while [ "${start}" -le "${TOTAL_TASKS}" ]; do
                shard=$((shard + 1))
                end=$((start + SHARD_SIZE - 1))
                [ "${end}" -le "${TOTAL_TASKS}" ] || end="${TOTAL_TASKS}"
                run_dir="${RESULT_ROOT}/trial_$(printf '%02d' "${trial}")/shard_$(printf '%03d' "${shard}")_${start}_${end}"
                lane_dependencies[next_lane]="$(submit_one "${trial}" "${shard}" "${start}" "${end}" "${run_dir}" "${lane_dependencies[next_lane]}")"
                next_lane=$(((next_lane + 1) % MAX_INTERACTIVE_NODES))
                start=$((end + 1))
            done
        done
        printf 'Evaluation chains submitted. Last jobs: %s\n' "${lane_dependencies[*]}"
        printf 'Final result will be written to: %s/final_summary.json\n' "${RESULT_ROOT}"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
