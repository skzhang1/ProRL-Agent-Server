#!/usr/bin/env bash
# One-click 4-node Qwen3.5-4B + PI training using the jrdev Slime recipe.
# The proven three-gateway topology and PI harness settings remain owned by
# submit_tmax_pi_4node_minimal.sh.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
project_root="${project_root:-$(cd -- "${script_dir}/../.." && pwd)}"
reference_root="${reference_root:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/example}"

export reference_recipe=1
export slime_dir="${slime_dir:-${reference_root}/slime}"
export polar_project_root="${polar_project_root:-${reference_root}/ProRL-Agent-Server-jrdev 3}"

# Match the reference GRPO/DPPO recipe while retaining the validated 4B model,
# PI agent, 8 training GPUs, 24 rollout GPUs, and three gateway sidecars.
export rollout_batch_size="${rollout_batch_size:-8}"
export n_samples_per_prompt="${n_samples_per_prompt:-32}"
export global_batch_size="${global_batch_size:-256}"
export target_num_rollout="${target_num_rollout:-32}"
export max_tokens_per_gpu="${max_tokens_per_gpu:-67584}"
export rollout_max_response_len="${rollout_max_response_len:-16384}"
export sglang_context_length="${sglang_context_length:-262144}"
export log_probs_chunk_size="${log_probs_chunk_size:-256}"
export train_lr="${train_lr:-1e-6}"
export clip_grad="${clip_grad:-1.0}"
export save_interval="${save_interval:-10}"
export polar_fully_async="${polar_fully_async:-1}"
export polar_min_complete_accept_fraction="${polar_min_complete_accept_fraction:-0.0}"
export polar_max_trajectory_tokens="${polar_max_trajectory_tokens:-67584}"

# Explicitly lock the validated multi-gateway settings.
export polar_multi_gateway=1
export polar_gateway_count=3
export polar_gateway_max_init_workers="${polar_gateway_max_init_workers:-24}"
export polar_gateway_max_run_workers="${polar_gateway_max_run_workers:-96}"
export polar_gateway_max_postrun_workers="${polar_gateway_max_postrun_workers:-64}"
export polar_runtime_memory_mb="${polar_runtime_memory_mb:-65536}"

export run_label="${run_label:-tmax-refrecipe-4n-rb8-s32}"
export experiment_name="${experiment_name:-tmax_pi_4b_refrecipe_4n_rb8_s32}"
export run_id="${run_id:-${experiment_name}}"

exec sbatch \
    --job-name=tmax-pi-4b-refrecipe-4n \
    --output="${script_dir}/logs/slurm/%x-%j.out" \
    --error="${script_dir}/logs/slurm/%x-%j.err" \
    "${script_dir}/submit_tmax_pi_4node_dppo.sh"
