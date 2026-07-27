#!/usr/bin/env bash
#SBATCH --job-name=swebench-harness-matrix
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=03:55:00
#SBATCH --output=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/examples/swebench_verified/results/slurm/%x-%A_%a.out
#SBATCH --error=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/examples/swebench_verified/results/slurm/%x-%A_%a.err
#SBATCH --export=ALL
set -euo pipefail

MODE="${1:-${MATRIX_MODE:-plan}}"
if [ "$#" -gt 0 ]; then shift; fi
while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT_PATH="$2"; shift 2 ;;
        --run-group) MATRIX_RUN_GROUP="$2"; shift 2 ;;
        --shard-size) SHARD_SIZE="$2"; shift 2 ;;
        --array-concurrency) ARRAY_CONCURRENCY="$2"; shift 2 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

PROJECT_ROOT="${PROJECT_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server}"
SCRIPT_DIR="${PROJECT_ROOT}/examples/swebench_verified"
GENERIC_LAUNCHER="${SCRIPT_DIR}/submit_swebench_pi_apptainer.sh"
AGGREGATOR="${SCRIPT_DIR}/aggregate_results.py"
COMPARATOR="${SCRIPT_DIR}/compare_harness_results.py"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/checkpoints/swe/iter_0000073}"
MATRIX_RUN_GROUP="${MATRIX_RUN_GROUP:-swebench_verified_other_harness_qwen35_4b}"
TOTAL_DATASET_INSTANCES="${TOTAL_DATASET_INSTANCES:-500}"
SHARD_SIZE="${SHARD_SIZE:-10}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
HARNESS_CSV="codex,claude_code,qwen_code"
IFS=, read -r -a HARNESSES <<<"${HARNESS_CSV}"

case "${ARRAY_CONCURRENCY}" in 1|2) ;; *) printf 'ERROR: concurrency must be 1 or 2\n' >&2; exit 2 ;; esac
[ "${SHARD_SIZE}" -gt 0 ] || { printf 'ERROR: shard size must be positive\n' >&2; exit 2; }
SHARDS=$(( (TOTAL_DATASET_INSTANCES + SHARD_SIZE - 1) / SHARD_SIZE ))
CHECKPOINT_TAG="$(printf '%s' "$(basename -- "${CHECKPOINT_PATH}")" | tr -c 'A-Za-z0-9_.-' '_')"

print_plan() {
    cat <<EOF
SWE-bench Verified non-PI harness matrix
  harnesses:       ${HARNESS_CSV}
  comparison:      checkpoint versus base within each harness
  tasks/cell:      ${TOTAL_DATASET_INSTANCES}
  cells:           6 (3 harnesses x 2 model variants)
  shards/cell:     ${SHARDS} (size ${SHARD_SIZE})
  array tasks:     $((SHARDS * 6))
  node hard cap:   ${ARRAY_CONCURRENCY} interactive nodes via one Slurm array
  run group:       ${MATRIX_RUN_GROUP}
  checkpoint:      ${CHECKPOINT_PATH}
  Codex CLI:       ${CODEX_VERSION:-0.145.0}, reasoning=${CODEX_REASONING_EFFORT:-xhigh}
  Claude Code CLI: installed shared version, native defaults
  Qwen Code CLI:  installed shared version, native defaults
EOF
}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    case "${MODE}" in
        plan)
            print_plan
            ;;
        smoke|submit)
            mkdir -p "${SCRIPT_DIR}/results/slurm"
            print_plan
            if [ "${MODE}" = smoke ]; then
                last_index=5
                smoke=1
            else
                last_index=$((SHARDS * 6 - 1))
                smoke=0
            fi
            job_id="$(sbatch --parsable \
                --array="0-${last_index}%${ARRAY_CONCURRENCY}" \
                --export="ALL,MATRIX_MODE=run,MATRIX_SMOKE=${smoke},MATRIX_RUN_GROUP=${MATRIX_RUN_GROUP},CHECKPOINT_PATH=${CHECKPOINT_PATH},SHARD_SIZE=${SHARD_SIZE},TOTAL_DATASET_INSTANCES=${TOTAL_DATASET_INSTANCES},ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY}" \
                "${BASH_SOURCE[0]}")"
            printf 'Submitted matrix array: %s\n' "${job_id}"
            ;;
        aggregate)
            output_dir="${SCRIPT_DIR}/results/${MATRIX_RUN_GROUP}_comparisons"
            for harness in "${HARNESSES[@]}"; do
                harness_group="${MATRIX_RUN_GROUP}_${harness}"
                bash "${GENERIC_LAUNCHER}" aggregate \
                    --harness "${harness}" --run-group "${harness_group}" \
                    --checkpoint "${CHECKPOINT_PATH}" --variants base,checkpoint
                python3 "${COMPARATOR}" \
                    --harness "${harness}" \
                    --base-summary "${SCRIPT_DIR}/results/${harness_group}_base/aggregate_summary.json" \
                    --checkpoint-summary "${SCRIPT_DIR}/results/${harness_group}_checkpoint_${CHECKPOINT_TAG}/aggregate_summary.json" \
                    --output-dir "${output_dir}"
            done
            ;;
        *) printf 'ERROR: use plan, smoke, submit, or aggregate\n' >&2; exit 2 ;;
    esac
    exit 0
fi

[ "${MODE}" = run ] || { printf 'ERROR: Slurm matrix task requires MATRIX_MODE=run\n' >&2; exit 2; }
[ "${SLURM_JOB_PARTITION:-}" = interactive ] || { printf 'ERROR: interactive partition required\n' >&2; exit 2; }
[ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ] || { printf 'ERROR: exactly one node per array task required\n' >&2; exit 2; }
array_id="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
if [ "${MATRIX_SMOKE:-0}" = 1 ]; then
    harness_index=$((array_id / 2))
    variant_index=$((array_id % 2))
    range="1-1"
    run_group="${MATRIX_RUN_GROUP}_smoke_${HARNESSES[harness_index]}"
else
    cells_per_harness=$((SHARDS * 2))
    harness_index=$((array_id / cells_per_harness))
    within_harness=$((array_id % cells_per_harness))
    shard_index=$((within_harness / 2))
    variant_index=$((within_harness % 2))
    start=$((shard_index * SHARD_SIZE + 1))
    end=$((start + SHARD_SIZE - 1))
    [ "${end}" -le "${TOTAL_DATASET_INSTANCES}" ] || end="${TOTAL_DATASET_INSTANCES}"
    range="${start}-${end}"
    run_group="${MATRIX_RUN_GROUP}_${HARNESSES[harness_index]}"
fi
if [ "${variant_index}" -eq 0 ]; then variant=base; else variant=checkpoint; fi
harness="${HARNESSES[harness_index]}"
printf 'matrix cell: harness=%s variant=%s range=%s run_group=%s\n' "${harness}" "${variant}" "${range}" "${run_group}"
export MODEL_SOURCE="${variant}" AGENT_HARNESS="${harness}" RUN_GROUP="${run_group}"
export CHECKPOINT_PATH INSTANCE_RANGE="${range}" MAX_TASKS=-1
exec bash "${GENERIC_LAUNCHER}" run \
    --harness "${harness}" --run-group "${run_group}" --checkpoint "${CHECKPOINT_PATH}"
