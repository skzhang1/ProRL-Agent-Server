#!/usr/bin/env bash
#SBATCH --job-name=webarea-debug-tmax-pi-1n-validgrad
#SBATCH --account=nemotron_omni_vision
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.err
#SBATCH --export=ALL

# One-node TMax PI valid-gradient smoke.
# Reuses the standard 1-node launcher but runs two rollout boundaries, which
# covers one optimizer update in Slime.
set -euo pipefail

project_root="${project_root:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server}"
export project_root

export run_id="${run_id:-webarea-debug_tmax_pi_1n_validgrad_${SLURM_JOB_ID}}"
export run_label="${run_label:-1n-tmax-validgrad}"

export smoke_rows="${smoke_rows:-4}"
export rollout_batch_size="${rollout_batch_size:-4}"
export n_samples_per_prompt="${n_samples_per_prompt:-4}"
export num_rollout="${num_rollout:-2}"
export target_num_rollout="${target_num_rollout:-2}"
export save_interval="${save_interval:-1}"

export polar_max_async_level="${polar_max_async_level:-4}"
export pi_context_window="${pi_context_window:-32000}"
export rollout_max_prompt_len="${rollout_max_prompt_len:-32000}"

export ray_tmpdir="${ray_tmpdir:-/tmp/wdtr-${SLURM_JOB_ID}}"
export job_cache_root="${job_cache_root:-/tmp/wdtp-${SLURM_JOB_ID}}"
export wandb_project="${wandb_project:-harness-tma}"

exec bash "${project_root}/examples/tmax_slime/submit_tmax_pi_1node_interactive.sh"
