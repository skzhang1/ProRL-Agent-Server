# SWE-bench Verified evaluation on NRT

This directory contains the NRT/Apptainer evaluation path for PI, Codex, Claude Code, and Qwen Code with Qwen3.5-4B.
It evaluates the pre-training checkpoint and an arbitrary Megatron distributed
checkpoint with the same model-serving, sampling, SGLang, timeout, and SWE-bench
configuration within each harness.

The generic upstream Docker example files remain for reference. On NRT use
`submit_swebench_pi_apptainer.sh`; it is restricted to the `interactive`
partition, one node per shard, and at most two concurrent shard jobs.

## One-command workflow

```bash
cd /lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/ProRL-Agent-Server

# Print the exact plan without submitting.
bash examples/swebench_verified/submit_swebench_pi_apptainer.sh plan \
  --checkpoint /lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/checkpoints/swe/iter_0000073

# One Verified instance for base, followed by the checkpoint.
bash examples/swebench_verified/submit_swebench_pi_apptainer.sh smoke \
  --checkpoint /lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/checkpoints/swe/iter_0000073

# Full strict pass@1: 500 tasks for base, then 500 for the checkpoint.
bash examples/swebench_verified/submit_swebench_pi_apptainer.sh submit \
  --checkpoint /lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/checkpoints/swe/iter_0000073

# Aggregate both variants after all array jobs finish.
bash examples/swebench_verified/submit_swebench_pi_apptainer.sh aggregate \
  --checkpoint /lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/checkpoints/swe/iter_0000073
```

`--checkpoint` accepts either an `iter_XXXXXXX` directory or a checkpoint root
containing `latest_checkpointed_iteration.txt`. `--variants base` and
`--variants checkpoint` run only one model. `--shard-size` defaults to 50 and
`--array-concurrency` accepts only 1 or 2.

## Non-PI harness matrix

Use one global Slurm array for the three requested harnesses. Array concurrency is validated as 1 or 2, so the six model/harness cells can never collectively exceed two interactive nodes. Base and checkpoint shards are interleaved within each harness.

```bash
# Review only.
bash examples/swebench_verified/submit_swebench_harness_matrix.sh plan \
  --checkpoint /lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/checkpoints/swe/iter_0000073

# Six one-instance smoke tests (3 harnesses x 2 model variants).
bash examples/swebench_verified/submit_swebench_harness_matrix.sh smoke \
  --checkpoint /lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/checkpoints/swe/iter_0000073

# Full 6 x 500 strict pass-at-1 matrix.
bash examples/swebench_verified/submit_swebench_harness_matrix.sh submit \
  --checkpoint /lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/checkpoints/swe/iter_0000073

# Aggregate and compare checkpoint against base separately within each harness.
bash examples/swebench_verified/submit_swebench_harness_matrix.sh aggregate \
  --checkpoint /lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/checkpoints/swe/iter_0000073
```

The full matrix defaults to 10 tasks per shard so four-way asynchronous execution remains inside the 3:55 Slurm walltime even when individual tasks approach the one-hour timeout; sharding does not change per-instance evaluation parameters.

The matrix freezes the already installed CLIs rather than modifying shared software: Codex 0.145.0 (explicit version check, `xhigh` reasoning), Claude Code 2.1.217, and Qwen Code 0.20.1 (fixed 16k output limit via its documented `QWEN_CODE_MAX_OUTPUT_TOKENS` setting). The two model variants within a harness use the same CLI, Polar configuration, SIFs, sampling, SGLang limits, and timeouts. Harness-native turn and token policies differ across harnesses, so only the base-versus-checkpoint comparison within the same harness is treated as controlled. PI is not rerun by the matrix.

## Evaluation semantics

- Dataset: all 500 SWE-bench Verified test instances.
- Metric: strict pass@1; missing, timeout, evaluator error, and malformed output
  count as unresolved.
- PI: `openai-completions`, 24k context, 512 tokens per call, compaction enabled.
- Sampling: temperature 1.0, top-p 1.0, top-k -1.
- Trajectory limits: 32k prompt, 16k response, 60k SGLang context.
- Builder/evaluator: `prefix_merging` and `swebench_harness` in a refreshed SIF.
- Timeout: 3600 seconds per task and 3900 seconds per request. This is longer
  than training because it includes potentially slow official verification.

Each model is loaded through Megatron and synchronized into the same SGLang
configuration. A supplied iteration directory is exposed through a per-run
checkpoint view; the source checkpoint is never modified.

## Outputs

All generated state is under `examples/swebench_verified/results/`:

- `slurm/`: scheduler stdout/stderr;
- `<run-id>/logs/`: Polar, Ray, Slime, and aggregation logs;
- `<run-id>/rollout_results/task_*/ses_*.json`: trajectories, patches, and scores;
- `<run-group>/aggregate_summary.json`: strict pass@1 summary and per-instance data;
- `<run-group>/aggregate_results.csv`: flat per-instance results.

## External read-only resources

The launcher uses the current project checkout and its existing Slime/Megatron
checkouts, the current training sqsh, the project-local patched Apptainer and shared agent
CLIs, the cached Verified JSON from the old eval directory, and the shared
`singularity_images_v3` SIF store. It does not modify Polar, Slime, Megatron,
or any TMax file. `swebench_eval_hooks.py` uses Slime's documented custom
eval-log hook so PI sessions without trainable token arrays do not trigger the
default logger's empty-list division; authoritative scoring still comes from
Polar's persisted SWE-bench session result.
