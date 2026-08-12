# Terminal-Bench 2.1 evaluation

This directory evaluates a PI-trained Qwen3.5-4B checkpoint on the pinned
89-task Terminal-Bench 2.1 test set. It reuses the project's existing
Slime/SGLang serving, Polar PI harness, Apptainer runtime, Harbor verifier, and
trajectory recording.

Run one smoke task:

```bash
bash examples/terminalbench/run_terminalbench.sh /path/to/iter_0000080 smoke
```

Submit the full evaluation (three strict pass@1 trials, six tasks per shard,
and at most one interactive node active at a time):

```bash
bash examples/terminalbench/run_terminalbench.sh /path/to/iter_0000080
```

Show progress or the final mean and standard error:

```bash
bash examples/terminalbench/run_terminalbench.sh /path/to/iter_0000080 status
```

Re-running the submit command resumes unfinished shards. A readable first
session is final even when it failed or timed out; only tasks interrupted before
a session was recorded are run again. SIF images are stored in
`/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HarnessGen/terminal_bench_sif`.
