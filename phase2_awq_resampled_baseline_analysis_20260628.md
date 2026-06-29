# Phase 2 AWQ Resampled Baseline Analysis

更新时间：2026-06-28

## 1. 对应批次

- `batch_run_id`: `phase2-awq-baseline-resampled-20260628T221900Z`
- 原始 benchmark 目录：`results/raw/benchmark/phase2-awq-baseline-resampled-20260628T221900Z/`
- 原始服务侧采样目录：`results/raw/prometheus/phase2-awq-baseline-resampled-20260628T221900Z/`
- 聚合结果目录：`results/batches/phase2-awq-baseline-resampled-20260628T221900Z/`
- 关键 CSV：
  - `results/batches/phase2-awq-baseline-resampled-20260628T221900Z/baseline_metrics.csv`
  - `results/batches/phase2-awq-baseline-resampled-20260628T221900Z/baseline_service_metrics.csv`

## 2. 结论先行

这批 AWQ baseline 最有价值的结论不是“服务侧有没有排队”，而是：

**当前 workload 下，AWQ 7B 的主要压力来自输出长度和解码阶段；输入长度主要推高 TTFT；并发提升能显著提高吞吐，但到 `concurrency=16` 已经明显低于线性扩展。**

可以直接写进报告的判断：

- AWQ baseline 链路稳定：`48/48` 个 case 完成，`failed_cases=0`，请求侧与服务侧成功请求数可以对齐。
- 峰值请求吞吐在 `concurrency=16, input=128, output=128`，约 `22.37 req/s`。
- 最差 tail latency 在 `concurrency=16, input=2048, output=512`，`P95 E2EL` 约 `3026.79 ms`。
- 输出从 `128` 增加到 `512` 后，平均 `P95 E2EL` 从约 `633.78 ms` 增加到 `2252.93 ms`，约 `3.55x`。
- 输入从 `128` 增加到 `2048` 后，平均 `P95 TTFT` 从约 `62.22 ms` 增加到 `177.81 ms`，约 `2.86x`。
- 并发从 `1` 增加到 `16` 后，平均 QPS 从约 `1.20` 增加到 `12.48 req/s`，约 `10.4x`，不是线性 `16x`。
- 当前矩阵没有打出明显 KV cache 压力：`kv_cache_usage_perc_during_run_max` 最高约 `0.107`。
- 当前矩阵没有观测到 vLLM 暴露的 waiting queue：`num_requests_waiting_during_run_max` 全部为 `0.0`。

不能写成报告结论的判断：

- 不能说“系统完全没有排队”，只能说“没有在 `vllm:num_requests_waiting` 上观测到等待队列”。
- 不能说“KV cache 永远不是瓶颈”，只能说“当前 baseline 矩阵没有触发明显 KV cache 压力”。
- 不能用 `server_load` 解释性能变化；本批 `server_load_during_run_*` 全部为 `0.0`，在这套栈上没有分析价值。

## 3. 关键证据

### 3.1 输出长度是最强的延迟放大器

按输出长度聚合：

| output_tokens | 平均 QPS | 平均 P95 E2EL | 平均 P95 TTFT | 平均 P95 TPOT | 平均输出吞吐 |
|---:|---:|---:|---:|---:|---:|
| 128 | 10.36 req/s | 633.78 ms | 111.57 ms | 4.18 ms | 1326.11 tok/s |
| 512 | 2.86 req/s | 2252.93 ms | 106.22 ms | 4.22 ms | 1462.24 tok/s |

这个对比说明：

- 输出长度从 `128` 到 `512` 是 `4x`，请求 QPS 从 `10.36` 降到 `2.86`，约为原来的 `27.6%`。
- `P95 E2EL` 上升约 `3.55x`，非常接近输出 token 增长带来的解码时间放大。
- `P95 TTFT` 基本没有随输出长度上升，说明输出长度主要影响的是 decode 阶段，而不是首 token 前的 prefill 阶段。
- 输出吞吐 `tok/s` 反而略高，说明长输出 case 更能摊薄固定开销；但用户视角的单请求等待时间会明显变长。

因此，这批数据最稳的性能解释是：**端到端延迟主要被 decode token 数拉长，而不是被服务侧 waiting queue 拉长。**

### 3.2 输入长度主要影响 TTFT，并温和压低吞吐

按输入长度聚合：

| input_tokens | 平均 QPS | 平均 P95 E2EL | 平均 P95 TTFT | 平均 P95 TPOT | 最高 KV cache usage |
|---:|---:|---:|---:|---:|---:|
| 128 | 7.35 req/s | 1336.05 ms | 62.22 ms | 4.01 ms | 0.026 |
| 512 | 6.91 req/s | 1392.51 ms | 86.67 ms | 4.10 ms | 0.041 |
| 2048 | 5.57 req/s | 1601.50 ms | 177.81 ms | 4.48 ms | 0.107 |

这个对比说明：

- 输入从 `128` 增加到 `2048`，`P95 TTFT` 约 `2.86x`，这是 prefill 成本增加的直接表现。
- 同一变化下，平均 QPS 只从 `7.35` 降到 `5.57`，约下降 `24%`，没有输出长度那么剧烈。
- `P95 TPOT` 从 `4.01 ms` 到 `4.48 ms`，只小幅上升，说明长输入对 decode 单 token 成本有影响，但不是主导项。
- KV cache usage 随输入长度上升，但最高仍只有约 `0.107`，没有接近容量压力。

因此，长输入会让用户更久看到首 token，也会吃掉一部分吞吐，但它不是这批 baseline 中最主要的端到端延迟来源。

### 3.3 并发提升有效，但扩展开始变钝

按并发聚合：

| concurrency | 平均 QPS | 平均 P95 E2EL | 平均 P95 TTFT | 平均 P95 TPOT | 最高 running |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.20 req/s | 1282.27 ms | 48.64 ms | 3.87 ms | 1 |
| 4 | 4.52 req/s | 1345.78 ms | 72.04 ms | 4.00 ms | 4 |
| 8 | 8.24 req/s | 1463.86 ms | 121.75 ms | 4.22 ms | 8 |
| 16 | 12.48 req/s | 1681.51 ms | 193.16 ms | 4.69 ms | 16 |

这个对比说明：

- 从 `1` 到 `8`，QPS 增长接近并发增长，服务仍能有效吃满更多并行请求。
- 从 `8` 到 `16`，QPS 只从 `8.24` 到 `12.48`，约 `1.51x`，已经不是翻倍。
- `P95 TTFT` 从 `48.64 ms` 增加到 `193.16 ms`，说明高并发下 prefill/scheduler 竞争变明显。
- `num_requests_running_during_run_max` 正好能达到对应并发，说明采样确实捕到了服务运行中状态。

这里的重点不是“没有排队所以没有压力”，而是：**压力表现为 running 中请求的竞争和 TTFT/TPOT 上升，而不是 vLLM waiting gauge 非零。**

## 4. 服务侧指标应该怎么解释

### 4.1 请求侧为什么没问题

请求侧指标来自 benchmark 对每个请求生命周期的直接记录：请求什么时候发出、什么时候拿到首 token、什么时候完成、完成了多少请求、失败了多少请求。

所以这些指标天然适合分析：

- `request_throughput_qps`
- `output_throughput_tps`
- `mean/median/p95 TTFT`
- `mean/median/p95 TPOT`
- `mean/median/p95 E2EL`
- `completed/failed/error_rate`

它们不依赖“采样时刚好看到某个瞬间”，因此不会因为前后快照落在空闲边界而失真。

### 4.2 服务侧为什么之前有问题

服务侧指标分两类：

- `counter`：累计计数，比如 prompt token、generation token、成功请求数。
- `gauge`：瞬时状态，比如 running、waiting、KV cache usage、server load。

`counter` 用 `after - before` 是合理的，因为只要 case 期间发生过，请求结束后累计值就会增加。

`gauge` 不能只看 `before/after`，因为 case 开始前和结束后很可能都已经空闲。之前那批结果里，waiting/running/KV 这类字段很多为 `0`，核心问题就是只采了两个边界点。

这次已经改成运行中周期采样，所以解释力明显变强：

- `service_metrics_during_run_sample_count` 在 `48` 个 case 全部非零，采样链路有效。
- `num_requests_running_during_run_max` 范围为 `1` 到 `16`，能捕捉到实际运行并发。
- `kv_cache_usage_perc_during_run_max` 全部非零，能捕捉到 KV cache 过程占用。
- `num_requests_waiting_during_run_max` 仍全为 `0.0`，这次可以解读为“没有观测到 vLLM waiting queue”，而不是“采样方式完全没捕到过程状态”。

### 4.3 仍然不能过度解释的服务侧字段

`server_load_during_run_*` 本批全为 `0.0`，不要用于报告结论。它在当前 vLLM 版本和启动方式下没有给出有效负载信号。

`num_requests_waiting_during_run_max=0.0` 也不能被写成“完全没有任何等待”。更准确的说法是：

**当前 benchmark 的请求进入 vLLM 后，没有在 `vllm:num_requests_waiting` 这个暴露指标上形成可观测等待队列；但这不排除 scheduler 内部竞争、GPU decode 竞争、客户端并发限制或其他未暴露的等待。**

## 5. 对 Phase 3 的实际指导

如果 Phase 3 继续做 AWQ 调参，优先级建议如下：

1. 先围绕长输出和高并发调：`concurrency=16, input=2048, output=512` 是当前最差 tail case，`concurrency=16, input=128, output=128` 是当前 peak QPS case。
2. 优先评估 `max-num-seqs`、`max-num-batched-tokens`、调度相关参数对 TTFT/TPOT 的影响，而不是一开始就把重点放在 KV cache。
3. 如果目标是证明 KV cache 或长上下文瓶颈，需要新增更强 stress case，例如更长输入、更高并发、更大 `max-model-len` 或更多请求数。
4. 如果只需要进入 Phase 3，不需要因为 `server_load=0` 卡住；当前结果已经足够作为 AWQ baseline。

Phase 3 最小建议矩阵：

- 快路径吞吐点：`concurrency=16, input=128, output=128`
- 长输出 tail 点：`concurrency=16, input=2048, output=512`
- 中高并发长输出点：`concurrency=8, input=2048, output=512`
- 中等上下文长输出点：`concurrency=16, input=512, output=512`

这四个点比全量矩阵更适合快速判断调参是否真的改善了关键路径。

## 6. 可以写进阶段结论的版本

本次 `phase2-awq-baseline-resampled-20260628T221900Z` 完整跑完 `48` 个 AWQ baseline case，所有请求侧 case 均成功，服务侧 counter 与运行中 gauge 采样也已落盘。数据表明，在当前 RTX 5090 + vLLM 0.23.0 + Qwen2.5-7B-AWQ 栈上，baseline 性能主要受输出长度和并发竞争影响：输出从 `128` 到 `512` 会把平均 `P95 E2EL` 放大到约 `3.55x`，输入从 `128` 到 `2048` 主要体现为 `P95 TTFT` 上升到约 `2.86x`。当前矩阵未观测到 vLLM waiting queue，也未触发明显 KV cache 压力，因此后续 Phase 3 应优先围绕长输出、高并发和调度参数做对比；若要研究 KV cache，需要新增更强的长上下文或更高并发 stress workload。
