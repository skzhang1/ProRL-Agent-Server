#!/usr/bin/env bash
#SBATCH --job-name=hapo7-q35-3ep-1616-v1
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=batch_block1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --dependency=singleton
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --exclude=pool1-[00001-00224],pool0-[00126,04476,04843,04892,04982,05158,05432],pool0-[05800-05984]
#SBATCH --output=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server/logs/slurm/%x-%j.err
#SBATCH --export=ALL

# Seven-harness HAPO training. The implementation remains feature-gated in the
# shared launcher; this entry point owns an isolated checkpoint and CLI bundle.
set -euo pipefail
project_root="${project_root:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server}"
export project_root
export agent_harness=pi
export agent_label=hapo_multi_harness
export harness_pool=claude_code,pi,qwen_code,codex,mini_swe_agent,openclaw,nanobot
export harness_seed=9
export harness_sampling_strategy=hapo
export hapo_epsilon=0.35
export hapo_learning_rate=0.07
export hapo_correct_threshold=0.50
export train_num_gpus=24
export actor_num_nodes=3
export rollout_num_gpus=8
export allow_checkpoint_dp_reshard=1
export allow_uneven_dp_batch=1
export polar_gateway_count=2
export agent_cli_dir="${project_root}/tmp/swegym_hapo_agent_cli_portable/opt_node"
export num_epoch=3
export target_num_rollout=111
export train_lr=3e-7
export exit_duration_minutes=0
export wandb_single_owner=1
export run_label=hapo7-q35-3ep-1616-v1
export experiment_name=hapo7-q35-3ep-1616-v1
export run_id="${experiment_name}"
export wandb_run_id=hapo7-q35-3ep-1616-v1
export run_generation=20260819-swegym-grpo-hapo7-3ep-1616-v1
exec bash "${project_root}/examples/swegym_slime_grpo/submit_pi_multi_harness_slurm_apptainer.sh"
