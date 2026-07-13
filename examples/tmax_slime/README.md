# TMax Slime PI Debug

This directory contains the NRT TMax + PI + Slime GRPO launcher for
Qwen3.5-4B. It is intentionally scoped to this example directory and reuses the
validated SWE-Gym PI training container and Qwen3.5-4B model settings.

Default data sources:

- TMax tasks: `/lustre/fsw/portfolios/nvr/users/songyangh/bjin_works/agent_world_model/tmax15k/dataset`
- TMax SIFs: `/lustre/fsw/portfolios/nvr/users/songyangh/bjin_works/agent_world_model/tmax15k/sif`

One-node debug:

```bash
sbatch examples/tmax_slime/submit_tmax_pi_1node_interactive.sh
```

The job name is `webarea-debug-tmax-pi-1n` by default and the script rejects
runtime job names that do not start with `webarea-debug`. The smoke run scans a
prefix of TMax tasks for ready SIFs, selects two prompt groups, samples two
trajectories per group, and stops after one rollout.

One-node valid-gradient debug:

```bash
sbatch examples/tmax_slime/submit_tmax_pi_1node_validgrad.sh
```

This wrapper keeps the same one-node layout but uses four prompt groups, four
trajectories per group, and two rollout boundaries so Slime covers one optimizer
update. W&B defaults to `hwinf_dcm/harness-tma`.

Useful overrides:

```bash
smoke_rows=4 rollout_batch_size=4 n_samples_per_prompt=2 \
tmax_scan_tasks=1024 sbatch examples/tmax_slime/submit_tmax_pi_1node_interactive.sh
```
