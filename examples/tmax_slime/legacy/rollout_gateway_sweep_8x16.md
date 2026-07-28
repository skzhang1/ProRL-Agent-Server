# 2-node rollout gateway sweep（8×16）

日期：2026-07-23 至 2026-07-24（America/Los_Angeles）

## 结论

在本轮每个 setting 一个 rollout step 的受控比较中，推荐每个 gateway 使用：

- `max_init_workers=8`
- `max_run_workers=64`
- `max_postrun_workers=16`
- 每个 node 一个 gateway
- `session_timeout=900s`
- `max_steps=160`
- `num_samples=16`
- `early-stop target=12/16`（75%）
- `min_complete_accept_fraction=0.5`（至少 8/16 个 completed trainable sessions）
- `max_async_level=2`
- `max_replacement_groups=32`
- Apptainer instance mode，PID isolation 开启

64-worker setting 用时 2072.4 秒（34.54 分钟），是三组中最快的；成功率、GPU busy fraction 和有效 token 吞吐也最好。由于每组只运行了一个 step，结果包含题目难度和生成随机性的方差，因此该结论是当前最优的工程基线，而不是严格统计显著性结论。

## 实验配置

三个 setting 只改变每个 gateway 的 `max_run_workers`：32、48、64。其余参数保持一致。每个 step 需要收集 8 个可接受 group，每个 group 16 个 session。若 group 不满足接受条件，则从数据源继续取题并 replacement，直到取得 8 个 group。

使用的独立测试脚本：

`examples/tmax_slime/submit_tmax_pi_2node_gateway_sweep.sh`

已验证的原始 rollout 脚本未修改：

`examples/tmax_slime/submit_tmax_pi_2node_rollout_8step.sh`

## 性能结果

| Setting | Job | 状态 | Rollout 时间 | 尝试/接受 group | Replacement | Session queue mean | Session run mean | Tokens/GPU/s | Effective tokens/GPU/s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| W32 | 5377103 step 0 | 成功；step 1 按要求取消 | 2635.6s | 14/8 | 6 | 64.93s | 155.49s | 47.42 | 14.13 |
| W48 | 5378459 | COMPLETED | 3449.0s | 19/8 | 11 | 2.04s | 96.93s | 32.09 | 9.43 |
| W64 | 5378463 | COMPLETED | 2072.4s | 14/8 | 6 | 2.43s | 100.15s | 57.72 | 19.71 |

Rollout 窗口内的 GPU 遥测：

| Setting | 平均 GPU util | GPU 非零利用率采样占比 | 平均显存/GPU |
|---|---:|---:|---:|
| W32 | 40.83% | 50.07% | 58,986 MiB |
| W48 | 45.77% | 54.59% | 58,987 MiB |
| W64 | 44.38% | 55.06% | 58,765 MiB |

W48 的 wall time 更慢不是因为 worker 使单个 session 变慢。相反，W48 相比 W32：

- queue 从 64.93s 降到 2.04s；
- run mean 从 155.49s 降到 96.93s；
- GPU busy fraction 从 50.07% 提高到 54.59%。

W48 变慢的直接原因是它随机遇到 11 个不可接受 group，而 W32 和 W64 各为 6 个。按尝试 group 粗略归一化，wall time/attempt 为 W32 188.3s、W48 181.5s、W64 148.0s；虽然并发执行使这个指标不是严格的单组 latency，但它同样不支持“增加 worker 导致基础设施变慢”的解释。

## 题目难度与 replacement

相同 group 在多个 setting 中反复失败：

- group 1：W32 零 trainable、W48 1/16、W64 零 trainable；
- group 3：W32 零 trainable、W48 2/16、W64 2/16；
- group 5：W32 1/16、W48 零 trainable、W64 2/16；
- group 9：W32 2/16、W48 2/16、W64 1/16；
- group 12：W32 3/16、W48 2/16、W64 4/16。

这说明 replacement 的主要驱动确实是题目难度及同一题目的生成随机性，而不是 worker 数本身。阈值附近也有明显随机波动，例如 W48 的 group 7 和 17 都是 7/16，只差一个 session 即可接受；W64 的 group 8 为 6/16。

## 丢弃统计

### Group 级

| Setting | Low complete fraction | Zero trainable tokens | 总丢弃 group |
|---|---:|---:|---:|
| W32 | 4（66.7%） | 2（33.3%） | 6 |
| W48 | 9（81.8%） | 2（18.2%） | 11 |
| W64 | 5（83.3%） | 1（16.7%） | 6 |
| 合计 | 18（78.3%） | 5（21.7%） | 23 |

本轮没有因下列原因丢弃 group：

- task status 非 completed；
- task results 为空；
- 返回 session 数与 16 不一致；
- 转换后 samples 为空；
- loss mask 或 rollout logprob 缺失/长度不一致；
- callback、HTTP 或其他通用 task failure；
- policy staleness 超限。

### Trace 级

显式 trace drop 共 70 次：

- `prompt_ids + response_ids > 67584`：70（100%）；
- prompt 或 response token 为空：0；
- 其他显式 trace drop：0。

如果一个 session 的所有 trace 都被丢弃，adapter 会生成一个 fully masked dummy placeholder。它不会单独使整个 batch 崩溃；只有当整个 group 零 trainable token，或 completed trainable session 少于 8/16 时，group 才被丢弃和 replacement。

### Session 终止/屏蔽

三组共尝试 752 个 session：

| Session 结果 | 数量 | 占比 | 训练处理 |
|---|---:|---:|---|
| 正常 completed | 333 | 44.3% | 有合法 mask/logprob 时可训练 |
| 达到 `max_steps=160` | 187 | 24.9% | 原实验为 TIMEOUT/ABORTED；2026-07-24 修复后按 TRUNCATED 训练 |
| 达到 early-stop 12/16 后主动取消 | 159 | 21.1% | ERROR/FAILED，loss mask 清零；这是预期控制流 |
| `session execution timeout=900s` | 71 | 9.4% | TIMEOUT/ABORTED，loss mask 清零 |
| code 137 / SIGKILL | 2 | 0.27% | ERROR/FAILED；保留证据并继续定位 |

early-stop 的 159 个取消 session 不代表 159 个坏 trace。它们是在 group 已达到 12 个最低可用 session 后，为避免等待剩余长尾 session 而主动终止；该 group 仍可接受。

### Max-steps trainability 修复与验证

2026-07-24 对 `src/slime_bridge/adapter.py` 做了最小状态映射修复：仅当
`SessionResult.status=TIMEOUT` 且 `metadata.termination_reason=max_steps` 时，
sample 映射为 `TRUNCATED` 并保留原 loss mask/logprob。真正的
`session_timeout` 仍为 `ABORTED` 并清零 loss mask。

- 单测：focused adapter/acceptance `11 passed`；完整 `tests/slime_bridge` `31 passed`。
- 真实验证：job `5380681`，2 nodes，8×16，`max_steps=2`，W64。
- Job `COMPLETED (0:0)`，总 rollout 19.22s，8/8 groups 接受，无 replacement。
- 96 个 max-steps session 全部转换为 trainable `TRUNCATED` sample，共 21,850 个 trainable tokens；96/96 的 response token 与 rollout logprob 长度对齐。
- 32 个 early-stop 取消 session 仍为 fully masked placeholder。
- 128/128 session DELETE 返回 200；无 SIGKILL、code 137、OOM、Traceback 或 instance stop failure。
- 超过 67,584 tokens 或缺 token 的 trace 仍按原有容量检查丢弃。

## SIGKILL 证据

W48 和 W64 各出现一次 code 137：

- 总发生率 2/752 session（0.27%）；
- 两次均由运行时记录 `returncode=-9` 和 Apptainer inner command SIGKILL；
- cgroup `oom=0`、`oom_kill=0`，宿主内存也充足，因此不是 Slurm cgroup OOM；
- 运行时最终记录 `final_members=[]`；
- session DELETE 返回 200；
- gateway 无重启，job 正常完成，没有发现遗留 agent/container。

W64 的 affected instance 在 SIGKILL 前约 30 秒已经采样到 `members=[]`，随后 Apptainer exec wrapper 才返回 `-9`。这更像 instance 内命令/namespace 已退出后 wrapper 收到 SIGKILL，而不是 agent 子进程泄漏或宿主内存压力。现有证据仍无法确定 SIGKILL 的发送者，因此不把它改写成 timeout，也没有通过降低 worker 数绕过。

## Timeout

正常 completed session 的 run-time：

| Setting | Mean | P50 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| W32 | 146.6s | 75.4s | 393.4s | 493.1s | 566.0s |
| W48 | 168.1s | 89.7s | 448.9s | 563.4s | 758.8s |
| W64 | 144.0s | 85.3s | 322.6s | 427.4s | 605.6s |

800 秒可以节省 execution-timeout session 最后的 100 秒，并覆盖本轮所有成功 session；但 W48 已观察到 758.8 秒的成功样本，只剩约 41 秒余量。为避免 gateway sweep 同时改变两个变量，当前推荐基线仍保留 900 秒。若继续优化，下一次应单独比较 W64 下的 800s 与 900s。

## 当前建议

后续 2-node 多-step rollout 和 4-node 扩展优先采用 W64。不要根据 W48 单次较慢就降低 worker；日志显示 W48 的基础设施 latency 实际优于 W32，其 wall-time 回退来自 replacement 方差。正式训练前仍应关注：

1. 多 step 下 W64 是否持续保持低 queue 和较高 effective throughput；
2. code 137 是否仍约为孤立的 0.3%，以及 instance wrapper 的信号来源；
3. 800 秒 timeout 是否能减少长尾而不提高 low-complete replacement。


## Num-samples 16 vs 32（保留 max-step trace）

2026-07-24 在相同的 2 nodes、相同的 W64（每个 gateway 64 run workers）和相同模型上顺序运行。除 `num_samples` 及其按比例派生的阈值外，其余设置保持一致：8 groups/step、early-stop 75%、min-complete 50%、`max_steps=160`、session timeout 900s。两个 setting 各运行 2 个 rollout steps，均使用修复后的 max-step 状态映射；`TRUNCATED` 比例非零，说明达到 max steps 的有效 trace 已进入训练 batch。

| num_samples | Step 0 | Step 1 | 平均/step |
|---:|---:|---:|---:|
| 16 | 1343.283s（22.39 min） | 1400.540s（23.34 min） | **1371.912s（22.87 min）** |
| 32 | 1926.393s（32.11 min） | 2037.485s（33.96 min） | **1981.939s（33.03 min）** |

`num_samples=32` 的单 step wall time 比 16 高 44.5%。固定 128 个 gateway run workers 时，每 step session 数从 128 增至 256，平均 init queue 从 5.32s 增至 124.76s，说明 W64 已出现明显排队。但按请求 session 数计算的名义吞吐仍从 0.0933 session/s 提升到 0.1292 session/s（+38.4%）；代价是训练每一步更新时间明显变慢、长尾和 group replacement 的等待更重。

两组 rollout success rate 平均分别为 71.1% 和 69.1%，effective tokens/GPU/s 平均分别为 57.58 和 58.45，未显示 `num_samples=32` 带来明显的有效 token 吞吐收益。两项作业均 `COMPLETED (0:0)`；日志无 SIGKILL、code 137、OOM 或 Traceback。ns16/ns32 gateway 分别记录 159/346 次 session DELETE 200，非 200 为 0，并执行了 service cleanup。

本轮建议继续使用 `num_samples=16`：它将每个训练 step 的 rollout 等待缩短约 10.17 分钟，同时稳定性和有效 token 吞吐与 32 相当。若目标只看单位 wall time 产生的请求 session 数，32 更高；但对同步训练 step latency 和后续 4-node DPPO，16 更均衡。

