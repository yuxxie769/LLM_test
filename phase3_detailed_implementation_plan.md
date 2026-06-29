# Phase 3 详情实施计划

## 当前执行结果（2026-06-29）

Phase 3 的两条主线已经有正式产物：

- 双模型 baseline compare 已完成，输入批次为 normal 7B `phase2-7b-baseline-resampled-20260628T162953Z` 和 AWQ `phase2-awq-baseline-resampled-20260628T221900Z`。产物位于 `results/model_compare/`，详细分析见 [model_compare_detailed_analysis.md](./results/model_compare/model_compare_detailed_analysis.md)。
- AWQ 参数 sweep 已完成并校验通过，正式 warmup 后统计批次为 `phase3-formal-awq-gpu-warmup-20260628T1556Z`。产物位于 `results/param_tuning/`，最终调用建议见 [awq_full_gpu_vs_phase2_analysis.md](./results/param_tuning/awq_full_gpu_vs_phase2_analysis.md)。
- 长上下文补实验已完成，批次为 `phase3-long-context-awq-20260629T065835Z`，单独产物位于 `results/param_tuning_long_context/`，并已合并回主 `results/param_tuning/` 形成 `25` 行汇总结果。
- 详细显存分解 baseline 已完成，批次为 normal `phase2-7b-baseline-detailed-20260629T052330Z` 与 AWQ `phase2-awq-baseline-detailed-20260629T143227Z`。两者都记录了启动前 GPU memory、服务健康后 GPU memory、停止后 GPU memory，以及 vLLM startup log 中的 model weights memory、available KV cache memory、GPU KV cache size、CUDA graph / allocator 线索。
- 当前结论仍然成立：AWQ 在同 workload 下 `24/24` 个 case 吞吐高于 normal 7B，`24/24` 个 case P95 latency 更低；后续默认调用配置仍建议为 `max_model_len=3072`、`max_num_batched_tokens=4096`、`max_num_seqs=32`。
- 长上下文补实验把默认值结论补到了 `input=2048`。在 `c8` 下，`max_model_len=3072/4096` 差异很小，`max_num_batched_tokens=8192` 没有带来稳定收益；在 `c16` 下，`max_model_len=4096` 明显改善 QPS、TTFT 和 P95，尤其是 `out512` 时最优点稳定落在 `4096/4096`。
- 显存分解结论也已经闭环：AWQ 权重显存从 normal 的 `14.19 GiB` 降到 `5.2 GiB`，但 vLLM 同时把可用 KV cache 从 `12.7 GiB` 扩到 `22.26 GiB`，所以服务健康后的 `nvidia-smi used` 没有按权重比例下降。不要再把 AWQ 推荐依据写成“显著节省最终显存占用”。

更新时间：2026-06-29

## 兼容性补充（2026-06-28）

本阶段已按最新执行顺序调整为：**跳过独立 `Phase 4`，直接进入 `Phase 3`**。

这意味着：

- 模型对比不再维护独立阶段，而是直接复用 `Phase 2` 的 baseline 结果链路
- `Phase 3` 分成两段：
  - 先做双模型 baseline 对比
  - 再在选定主模型上做单变量参数 sweep
- 早期 `Tesla V100S 32GB (sm_70)` 只能承担非量化 7B 与脚本开发，不能承担 AWQ 正式本机结论。
- 当前正式执行机器已切换为 `NVIDIA GeForce RTX 5090 (sm_120, 32 GiB)`，AWQ baseline、normal 7B baseline 与 AWQ full-GPU sweep 都已在该机器上跑完。
- 因此本文后续的正式结论以 RTX 5090 执行结果为准，旧 V100S 内容只作为历史兼容性背景。

因此本阶段固定两条原则：

1. 双模型对比直接建立在 **同一套 `Phase 2` workload** 上
2. 参数调优坚持 **单变量 sweep + 固定 workload**

补充执行约定：

- 如果 GPU 上已有常驻模型服务，不强制抢占或停止原服务
- 允许通过 `PHASE3_GPU_MEMORY_UTILIZATION` 覆盖默认档位，在共享 GPU 上用更低显存预算启动独立 sweep 服务
- 允许把 3 个 tuning target 拆成多次 sweep 分批执行，最后再合并多个 manifest 生成正式 `param_tuning.csv`

## 1. 目标

本文件用于把 [experience1_implementation_plan_v2.md](./experience1_implementation_plan_v2.md) 中的 `Phase 3` 收敛成可直接执行的仓库实施方案。

本阶段只回答五类问题：

1. 两个候选模型在同一套 baseline workload 下，`QPS / P95 / tokens/s / 显存` 有什么直接差异
2. 哪个模型更适合作为后续主报告对象
3. `max_model_len` 调大后，长上下文预算和显存压力之间怎么取舍
4. `max_num_batched_tokens` 调大后，吞吐是否改善，尾延迟是否恶化
5. `max_num_seqs` 调大后，排队是否减少，以及这种减少是否真的转化成更好的用户延迟

本阶段不做：

- 独立维护 `Phase 4` 代码分支
- 全参数笛卡尔积 sweep 作为常规默认路径
- `gpu_memory_utilization` 主线调参
- 量化算法或内核级优化
- 多 workload 同时 sweep 作为快速调参默认路径

例外：收尾阶段必须增加一个受控长上下文交叉网格，只覆盖 `max_model_len` 与 `max_num_batched_tokens` 两个参数，用于验证 Phase 2 最差压力点附近的 prefill 与 batching 行为。

## 2. 当前基线

### 2.1 已完成前置

当前仓库已经具备：

- `Phase 1`：可运行的 `vLLM + FastAPI` 服务
- `Phase 2`：baseline workload、请求侧 / 服务侧指标采集、CSV / summary / plot / validate 链路
- 多组真实 `Phase 2` batch 结果，可直接作为双模型 compare 输入

当前可直接复用的代码入口：

- 单 case 执行：[bench/run_single_case.py](./bench/run_single_case.py)
- 矩阵执行：[bench/run_matrix.py](./bench/run_matrix.py)
- 基础结果聚合：[analysis/aggregate_results.py](./analysis/aggregate_results.py)
- 基础图表：[analysis/plot_baseline.py](./analysis/plot_baseline.py)
- 基础 summary：[analysis/render_baseline_summary.py](./analysis/render_baseline_summary.py)
- 基础 batch 校验：[analysis/validate_batch.py](./analysis/validate_batch.py)
- 当前服务启动脚本：[scripts/run_vllm_local.sh](./scripts/run_vllm_local.sh)

### 2.2 当前推荐主模型策略

正式建议是：

- 优先先完成双模型 compare
- 再用 compare 结果选择 `Phase 3 sweep` 主模型
- 如果第二个模型暂时还不能在当前机器上正式复现，就先用当前最稳定的：
  - `Qwen2.5-7B-Instruct`
  - `MAX_MODEL_LEN=3072`
  - `VLLM_MAX_NUM_BATCHED_TOKENS=4096`
  - `VLLM_MAX_NUM_SEQS=32`
  - `GPU_MEMORY_UTILIZATION=0.9`

也就是说，`Phase 3` 的调优起点不是随机配置，而是已经跑通过 baseline 的稳定档位。

### 2.3 当前缺口

当前仓库还缺少以下 `Phase 3` 交付物：

- 直接复用 `Phase 2` batch 输出的模型 compare 聚合脚本
- 模型 compare summary / plot / validate 链路
- 专用 `Phase 3` sweep 配置
- 单变量 sweep 运行脚本
- sweep 结果聚合、summary、plot、validate 链路
- 多个 split sweep manifest 的合并能力
- `results/model_compare/` 与 `results/param_tuning/` 两套正式产物目录

## 3. Phase 3 最终范围

### 3.1 第一段：双模型 baseline 对比

执行结构：

1. 分别完成两个模型的 `Phase 2` baseline batch
2. 读取两个 batch 的 `baseline_metrics.csv` 与 `baseline_service_metrics.csv`
3. 只保留两个模型都有结果的重叠 case
4. 输出统一 compare 宽表、summary 和图表

这一段不重新定义 workload，而是**直接吃 `Phase 2` 已有矩阵结果**。

### 3.2 第二段：参数 sweep

执行结构：

1. **服务启动层**
   - 每轮 sweep 用一组明确参数启动 `vLLM`
   - 所有配置都显式记录
   - 记录启动前、服务健康后、停止后的 GPU memory，并保留 vLLM startup log 显存相关解析结果
2. **workload 执行层**
   - 快速调参仍跑固定短 workload
   - 长上下文补实验跑受控网格 workload
   - 沿用 `Phase 2` 的请求侧 / 服务侧指标采集
3. **参数对比层**
   - 对同一 workload 下不同参数结果做横向比较
   - 输出宽表、图表和 summary

### 3.3 第一版只扫 3 个参数

| 参数 | 作用 | 第一版建议取值 |
|---|---|---|
| `max_model_len` | 单请求上下文预算上限 | `2048, 3072, 4096` |
| `max_num_batched_tokens` | 每个 batch 可打包的 token 上限 | `2048, 4096, 8192` |
| `max_num_seqs` | 调度器允许的同时序列数 | `16, 32, 64` |

补充说明：

- `3072` 是当前稳定 baseline 档位
- `gpu_memory_utilization` 暂时不进主线 sweep
- 第一版不做跨参数笛卡尔积

### 3.4 Workload 设计

#### 3.4.1 快速单变量 workload

快速调参仍固定为：

| 维度 | 取值 |
|---|---|
| 并发数 | `8` |
| 输入长度 | `512` |
| 输出长度 | `128` |
| warmup_repeats | `1` |
| measured repeat | `2` |
| num_prompts | `40` |

原因：

- 该 case 已经被 `Phase 2` 基线矩阵覆盖，验证更顺手
- 相比 `output=256`，`output=128` 更利于缩短 sweep 单轮耗时
- 仍然足以暴露并发、调度和排队差异
- 每个参数档位先执行 `1` 次 warmup，再统计 `2` 次正式 repeat，降低首轮 JIT / graph capture 对结论的污染

#### 3.4.2 长上下文补实验 workload

收尾阶段必须补一组长上下文网格：

| 维度 | 取值 |
|---|---|
| 并发数 | `8, 16` |
| 输入长度 | `2048` |
| 输出长度 | `128, 512` |
| `max_model_len` | `3072, 4096` |
| `max_num_batched_tokens` | `4096, 8192` |
| `max_num_seqs` | 固定 `32` |
| warmup_repeats | `1` |
| measured repeat | `2` |
| num_prompts | `40` |

这组不是常规全参数 sweep，而是针对 Phase 2 最差压力点的补实验。它要回答：

- `input=2048` 下，`max_model_len=3072/4096` 是否影响 prefill、TTFT 和 KV cache 预算
- `output=512` 下，`max_num_batched_tokens=4096/8192` 是否改善 batching 吞吐，还是只抬高尾延迟 / 显存预留
- `concurrency=16` 下，短 workload 得出的默认值是否仍然成立

### 3.5 单变量 sweep 规则

本阶段严格遵守：

1. 一次只 sweep 一个参数
2. 另外两个参数固定在当前已知稳定值
3. 每个参数档位先跑 warmup，再落正式 measured repeats
4. 聚合与 summary 只统计 warmup 之后的 measured repeats
5. 先扫 `max_model_len`，再扫 `max_num_batched_tokens`，最后扫 `max_num_seqs`

## 4. 指标设计

### 4.1 双模型 compare 关注指标

固定看：

- `request_throughput_qps`
- `p95_e2el_ms`
- `output_throughput_tps`
- `error_rate`
- `gpu_memory_used_mb_after`
- `num_requests_waiting_observed`
- `kv_cache_usage_perc_observed`

### 4.2 参数调优关注指标

固定看：

- `request_throughput_qps`
- `p50_e2el_ms`
- `p95_e2el_ms`
- `mean_ttft_ms`
- `mean_tpot_ms`
- `output_throughput_tps`
- `error_rate`
- `num_requests_waiting_peak`
- `kv_cache_usage_perc_peak`
- `gpu_memory_used_mb_after`

长上下文补实验额外关注：

- `input=2048` 相比短 workload 的 TTFT / prefill 放大
- `output=512` 下 TPOT、P95 E2EL 和 output throughput 的变化
- `max_num_batched_tokens=8192` 是否真正提高吞吐，还是只提高显存预留或尾延迟

### 4.3 显存分解指标

为了避免把 AWQ 显存结论写错，每轮服务启动需要记录：

- `gpu_memory_used_mb_before_start`：vLLM 进程启动前的 GPU memory
- `gpu_memory_used_mb_after_health`：服务 `/health` 可用后的 GPU memory，可近似覆盖模型加载、KV cache 初始化和运行时预留后的状态
- `gpu_memory_used_mb_after_stop`：服务停止后的 GPU memory，用于判断是否有残留占用
- `vllm_model_weights_memory_gb`：vLLM startup log 中的 model weights memory
- `vllm_gpu_kv_cache_size_tokens`：vLLM startup log 中的 GPU KV cache size
- `vllm_num_gpu_blocks` / `vllm_num_cpu_blocks`：vLLM startup log 中的 block 数
- `vllm_startup_cuda_graph_line_count` / `vllm_startup_allocator_line_count`：CUDA graph 与 allocator 相关日志线索

报告中应写成：**AWQ 权重显存会降低，但 vLLM 可能把释放出的空间继续用于 KV cache、CUDA graph 或 runtime allocator 预留；因此最终 `nvidia-smi used memory` 不等价于权重显存。**

### 4.4 结论判读规则

第一版只需要会看以下几种模式：

1. `QPS` 上升，`P95` 也明显上升
   - 说明吞吐提升是拿尾延迟换来的
2. `num_requests_waiting` 下降，但 `P95` 没改善
   - 说明减少排队不一定等于更好的用户体验
3. `gpu_cache_usage_perc` 显著变高，但吞吐没有提升
   - 说明参数把显存 / cache 压力拉高了，但没换来收益
4. 第二个模型显存更低，但 `P95` 更差
   - 说明模型切换带来的收益不一定是免费午餐

## 5. 仓库交付物

本阶段完成后，仓库内应至少新增：

```text
bench/
├── phase3_sweep.yaml
└── run_phase3_sweep.py

analysis/
├── phase3_common.py
├── aggregate_model_compare.py
├── plot_model_compare.py
├── render_model_compare_summary.py
├── validate_model_compare.py
├── aggregate_param_tuning.py
├── plot_param_tuning.py
├── render_param_tuning_summary.py
└── validate_param_tuning.py

scripts/
├── run_phase3_model_compare.sh
├── run_phase3_compare.sh
└── run_phase3_sweep.sh
```

正式运行后应至少产出：

```text
results/model_compare/
├── manifest.json
├── model_compare.csv
├── model_compare_long.csv
├── model_compare_summary.md
└── plots/

results/param_tuning/
├── raw/
├── manifest.json
├── param_tuning.csv
├── param_tuning_summary.md
└── plots/
```

## 6. 结果结构设计

### 6.1 模型 compare 宽表

`model_compare.csv` 每行对应一个重叠 case，至少包含：

- `reference_model`
- `candidate_model`
- `concurrency`
- `input_tokens`
- `output_tokens`
- `request_throughput_qps_reference`
- `request_throughput_qps_candidate`
- `request_throughput_qps_delta`
- `p95_e2el_ms_reference`
- `p95_e2el_ms_candidate`
- `p95_e2el_ms_delta`
- `gpu_memory_used_mb_after_reference`
- `gpu_memory_used_mb_after_candidate`
- `gpu_memory_used_mb_after_delta`

### 6.2 参数调优汇总表

`param_tuning.csv` 每行对应一个参数取值，至少包含：

- `tuning_target`
- `tuning_value`
- `max_model_len`
- `max_num_batched_tokens`
- `max_num_seqs`
- `request_throughput_qps`
- `p50_e2el_ms`
- `p95_e2el_ms`
- `output_throughput_tps`
- `error_rate`
- `num_requests_waiting_peak`
- `kv_cache_usage_perc_peak`
- `gpu_memory_used_mb_before`
- `gpu_memory_used_mb_after`
- `gpu_memory_used_mb_before_start`
- `gpu_memory_used_mb_after_health`
- `gpu_memory_used_mb_after_stop`
- `vllm_model_weights_memory_gb`
- `vllm_gpu_kv_cache_size_tokens`
- `vllm_num_gpu_blocks`
- `vllm_startup_cuda_graph_line_count`
- `vllm_startup_allocator_line_count`

## 7. 实现拆分

### 7.1 Compare 侧

新增：

- `analysis/aggregate_model_compare.py`
- `analysis/render_model_compare_summary.py`
- `analysis/plot_model_compare.py`
- `analysis/validate_model_compare.py`
- `scripts/run_phase3_model_compare.sh`

职责：

- 聚合两个 `Phase 2` batch 的重叠 case
- 生成对比宽表与长表
- 生成 summary 和图表
- 校验结果完整性

### 7.2 Sweep 运行侧

新增 `bench/run_phase3_sweep.py` 与 `scripts/run_phase3_sweep.sh`：

- 读取 `bench/phase3_sweep.yaml`
- 支持用 `PHASE3_GPU_MEMORY_UTILIZATION` 覆盖默认显存利用率
- 为每个参数取值显式启动服务
- 每个参数档位先执行 warmup，再执行 measured repeats
- 运行固定 workload
- 记录 sweep manifest、warmup raw 结果和 measured raw 结果

### 7.3 Sweep 聚合侧

新增：

- `analysis/aggregate_param_tuning.py`
- `analysis/render_param_tuning_summary.py`
- `analysis/plot_param_tuning.py`
- `analysis/validate_param_tuning.py`

职责：

- 读取单个或多个 sweep manifest 对应的结果
- 聚合成 `param_tuning.csv`
- 明确区分 warmup repeat 与 measured repeat，只汇总 measured repeat
- 输出调优结论与图表
- 校验 split sweep 合并后的结果完整性

## 8. 开发任务列表

按顺序执行：

1. 修订主计划，把独立 `Phase 4` 并入 `Phase 3`
2. 补齐双模型 compare 聚合链
3. 新建 `Phase 3` sweep 配置
4. 新建单变量 sweep 脚本
5. 新建 tuning summary / plot / validate 脚本
6. 用真实 `Phase 2` batch 验证 compare 链
7. 用 `--dry-run` 和语法检查验证 sweep 链

## 9. 开发前检查项

只有以下条件满足，才进入正式 `Phase 3` 运行：

- [ ] `Phase 2` baseline 结果可用
- [ ] 至少有两个候选模型的 `Phase 2` batch 输出，或已决定先只推进单模型 sweep
- [ ] `.venv` 可正常激活
- [ ] `vLLM` 服务可正常启动
- [ ] `/metrics` 可稳定抓取
- [ ] `results/`、`logs/` 可写

如果只是先做仓库开发，最小条件为：

- [ ] Python 脚本可通过语法检查
- [ ] compare 链可基于现有 batch 输出跑通
- [ ] sweep 链至少可通过 `--dry-run` 输出计划清单

## 10. 验收标准

只有以下条件全部满足，`Phase 3` 才算完成：

- [ ] 双模型 compare 结果完整落盘
- [ ] 生成 `results/model_compare/model_compare.csv`
- [ ] 生成 `results/model_compare/model_compare_summary.md`
- [ ] 参数 sweep 结果完整落盘
- [ ] 生成 `results/param_tuning/param_tuning.csv`
- [ ] `param_tuning.csv` 明确记录 warmup repeat 与 measured repeat 信息
- [ ] 生成 `results/param_tuning/param_tuning_summary.md`
- [ ] summary 明确声明正式统计是否已排除 warmup
- [ ] 两条链路合计至少生成 3 张图表
- [ ] 至少形成 2 到 3 条可信的 trade-off 结论
- [ ] 结论建立在同 workload compare、快速单变量 sweep 与长上下文补实验之上
- [ ] 长上下文补实验覆盖 `c8/c16, input=2048, output=128/512, max_model_len=3072/4096, max_num_batched_tokens=4096/8192`
- [ ] 显存结论包含启动前、服务健康后、停止后 GPU memory，以及 vLLM startup log 中的 weights / KV cache / GPU blocks / CUDA graph / allocator 字段

## 11. 当前最小启动条件

如果只问“现在还差什么才能开始 `Phase 3`”，答案很简单：

1. 直接用两个 `Phase 2` batch 产出一份统一 compare 报告
2. 把 sweep 范围收窄到 `max_model_len / max_num_batched_tokens / max_num_seqs`
3. 把快速 workload 锁成 `8 / 512 / 128`
4. 额外补长上下文网格 `c8/c16, input=2048, output=128/512, max_model_len=3072/4096, max_num_batched_tokens=4096/8192`
5. 把 `param_tuning.csv + summary + plot + validate` 这条链补齐，并让 summary 明确区分快速 sweep 与长上下文补实验
6. 补显存分解字段，避免用最终 `nvidia-smi` 占用直接解释 AWQ 权重显存

这些项补齐后，`Phase 3` 才能从“方向性想法”变成可直接支撑报告结论的实验闭环。
