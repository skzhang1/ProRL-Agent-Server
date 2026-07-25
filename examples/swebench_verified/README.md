# SWE-bench Verified evaluation on NRT

This directory contains the NRT/Apptainer evaluation path for PI + Qwen3.5-4B.
It evaluates the pre-training checkpoint and an arbitrary Megatron distributed
checkpoint with the same Polar, PI, sampling, SGLang, and SWE-bench harness
configuration.

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
checkouts, the current training sqsh, the project-local patched Apptainer and PI
CLI, the cached Verified JSON from the old eval directory, and the shared
`singularity_images_v3` SIF store. It does not modify Polar, Slime, Megatron,
or any TMax file. `swebench_eval_hooks.py` uses Slime's documented custom
eval-log hook so PI sessions without trainable token arrays do not trigger the
default logger's empty-list division; authoritative scoring still comes from
Polar's persisted SWE-bench session result.
