# 经历一：LLM 推理服务压测与性能分析平台 — 实现计划 v2

## 目标

这个项目的目标不是“证明我会做底层 CUDA 优化”或“从零造一个 benchmark 框架”，而是基于现有 **LLM serving / benchmark / observability** 工具，建立一套**可运行、可压测、可对比、可复现、可写进简历**的大模型推理服务实验闭环，最终支撑这 4 条简历表述：

1. 基于 **vLLM** 部署 Qwen 模型服务，并在前面增加轻量 FastAPI 网关  
2. 基于原生 benchmark 与服务侧指标，对不同并发、输入长度、输出长度场景做压测，采集 **QPS / P95 / TTFT / TPOT / ITL / error rate / GPU 指标**  
3. 对关键 serving 参数做对比实验，总结吞吐与延迟 trade-off  
4. 复用 **Phase 2** 基线链路，对两个候选模型在显存、吞吐与延迟上的差异做直接对比

---

## 简历 Bullet 对应关系

| # | Bullet | 核心产出 |
|---|--------|---------|
| 1 | vLLM 部署 + FastAPI 网关 | 可运行的推理服务 |
| 2 | 原生 benchmark + 指标采集 | 矩阵编排脚本 + 两层指标采集 + baseline 报告 |
| 3 | 参数调优 | 参数对比实验 + 调优结论 |
| 4 | 双模型基线对比 | 模型对比报告 |

---

## 总体顺序

推荐执行顺序：

```text
Phase 0 → Phase 1 → Phase 2 → Phase 3
```

原因：

- `Phase 1` 是服务基线，后面都依赖它
- `Phase 2` 先建立 baseline 指标
- `Phase 3` 先直接复用 `Phase 2` 基线链路完成双模型对比，再在选定主模型上做参数 sweep
- 这样可以避免维护一个独立 `Phase 4` 分支，同时保留原本需要的模型对比结论

## 当前机器兼容性补记

原始计划按“24GB+ GPU 即可完成 `FP16/BF16 vs AWQ-INT4` 对比”来写，但当前实际落地机器是 `Tesla V100S 32GB`，计算能力只有 `sm_70`。这会带来两类必须前置记录的兼容性约束：

- 当前机器不能继续使用原先尝试过的 `torch 2.11.0+cu130`，否则会报 `CUDA error: no kernel image is available for execution on the device`；本仓库现已切到 `torch 2.6.0+cu124 + vllm 0.8.5.post1`。
- `Qwen2.5-7B-Instruct-AWQ` 在这台机器上不是“性能较差”或“需要调参”，而是硬件能力不满足。实际报错是：`The quantization method awq is not supported for the current GPU. Minimum capability: 75. Current capability: 70.`，因此本机不能完成 `AWQ` 基线与 Phase 4 的本地对比。
- 当前机器可以稳定运行非量化 `Qwen2.5-7B-Instruct`，并已据此继续推进 `Phase 1/2` 的本地部署与 baseline 压测。

因此，后续解释计划产出时要区分两层结论：

- `Phase 1/2` 的本地链路验证、baseline 指标采集、脚本兼容性适配，已经可以在这台 `V100S 32GB` 上完成。
- 如果双模型对比中的第二个模型依赖 AWQ，那么正式对比仍需要切回 `sm_75+` 的 GPU，或者更换为支持 AWQ 的机器执行。

---

## 前置条件

### 模型选择

第一版固定使用：

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct-AWQ`

选择原因：

- 中文效果自然，方便后续用中文 prompt 做压测
- 官方提供可直接使用的 AWQ checkpoint
- 变量少，便于收敛和复现

第一版**不建议**同时引入第二个模型家族，例如 Llama。

### GPU 服务器

不是必须 A100 40G。

推荐分级：

- **最低可行**：24GB GPU
  - 适合完成 MVP，模型固定为 7B，适当控制上下文长度与并发
- **推荐配置**：48GB GPU
  - 更适合做 FP16/BF16 与 AWQ 对比，以及后续参数 sweep

优先级建议：

1. `A40 48GB`
2. `RTX A6000 48GB`
3. `A5000 24GB`
4. `L4 24GB`

### 环境

- Python 3.10+
- CUDA 12.x
- Linux
- 如果是本地 Windows 机器，优先走 `WSL2 + Ubuntu`；进入依赖安装前先确认 WSL 内 `/dev/dxg` 与 `nvidia-smi` 正常，并把项目放在 WSL Linux 文件系统而不是 `/mnt/<盘符>/...`
- 在受限沙箱、代理终端或某些自动化会话里，`nvidia-smi` 可能出现假阴性；这类结果需要回到正常 WSL 终端复核

---

## Phase 0：冻结实验定义

### 0.1 固定实验对象

- 模型：`Qwen2.5-7B-Instruct`
- 量化模型：`Qwen2.5-7B-Instruct-AWQ`
- 接口：OpenAI-compatible `/v1/chat/completions`
- 主 benchmark 工具：优先 `vLLM` 原生 benchmark，必要时用 `SGLang` 或 `GenAI-Perf` 做交叉验证
- `Locust` 仅作为网关链路验证和补充性黑盒压测工具

### 0.2 固定第一版指标

第一版指标分成两层：

1. **请求侧指标**
   - `QPS`
   - `P50 latency`
   - `P95 latency`
   - `TTFT`
   - `TPOT`
   - `ITL`
   - `error rate`
2. **服务侧指标**
   - `GPU memory used`
   - `num_requests_running`
   - `num_requests_waiting`
   - `gpu_cache_usage_perc`
   - `prompt throughput`
   - `generation throughput`

### 0.3 固定第一版 workload

第一版建议只做以下矩阵：

| 维度 | 取值 |
|------|------|
| 并发数 | 1, 4, 8, 16 |
| 输入长度 | 128, 512, 2048 tokens |
| 输出长度 | 128, 512 |

总共 `4 x 3 x 2 = 24` 组，已经足够形成可写简历的实验结果。

### 0.4 验收标准

- [ ] 固定模型、指标与 workload，不再反复修改
- [ ] 明确第一版不做多模型横向对比
- [ ] 明确第一版不自己实现量化算法

---

## Phase 1：vLLM 部署 + FastAPI 网关

### 1.1 vLLM 服务启动

建议使用更接近新版本文档的写法：

```bash
pip install vllm

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen-7b \
  --host 0.0.0.0 \
  --port 8100 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

说明：

- 先以单模型、单卡、默认配置跑通
- 第一版不要急着改太多 serving 参数
- 如果是本地 WSL 环境，先做 GPU 预检；`nvidia-smi` 不通时，不要继续进入 `pip install vllm` 和模型启动排查

### 1.2 FastAPI 网关

第一版建议做**轻量网关**，避免过度工程化。

推荐目录：

```text
llm-bench/
├── gateway/
│   ├── main.py
│   ├── schemas.py
│   ├── logger.py
│   └── config.py
├── bench/
├── analysis/
└── results/
```

第一版网关职责只做：

- 参数校验
- 异常标准化
- `request_id` 注入
- 请求日志记录
- 简单 Bearer token 校验

第一版**不需要**一开始拆出很多 middleware 和复杂模块。

### 1.3 日志字段

建议记录：

- `timestamp`
- `request_id`
- `input_tokens`
- `max_tokens`
- `latency_ms`
- `status_code`
- `error_type`

如果返回体里能稳定拿到 usage，再补：

- `output_tokens`

### 1.4 验收标准

- [ ] `/v1/chat/completions` 通过网关正常可用
- [ ] 无效请求返回结构化 4xx
- [ ] vLLM 异常能被网关捕获并转成统一错误格式
- [ ] 至少落一份结构化日志样本

---

## Phase 2：原生 Benchmark + 矩阵编排 + 指标采集

### 2.1 工具选择

第一版建议：

- **vLLM 原生 benchmark**：主方案
- **SGLang benchmark**：可选交叉验证
- **GenAI-Perf**：可选专业 benchmark 对照
- **Prometheus metrics**：服务侧指标主采集来源
- **Locust / hey**：仅用于网关链路验证和补充性黑盒测试

第一版先**不使用 `wrk`**，避免维护多套压测逻辑。

### 2.2 压测模式

分两层：

1. **benchmark 执行层**
   - 负责执行单个固定 case
   - 由 `vLLM` 原生 benchmark 或其他现成工具完成
2. **实验编排层**
   - 负责展开 workload 矩阵
   - 逐组调用 benchmark 命令
   - 同步抓取 `/metrics`
   - 统一落盘原始结果和汇总结果

不要把“多维矩阵”理解成“自己重写 benchmark 工具”。第一版只需要实现一个很薄的实验编排层。

### 2.3 workload 矩阵

第一版矩阵：

| 维度 | 取值 |
|------|------|
| 并发数 | 1, 4, 8, 16 |
| 输入长度 | 128, 512, 2048 |
| 输出长度 | 128, 512 |

建议 prompt 模板固定，减少随机波动。

补充说明：

- 非流式矩阵用于 `QPS / latency / error rate`
- 流式子矩阵用于 `TTFT / TPOT / ITL`
- 不要把 workload 矩阵、模型对比和参数 sweep 做全笛卡尔积

### 2.4 两层指标定义

#### 请求侧指标

| 指标 | 说明 |
|------|------|
| `QPS` | 总请求数 / 总测试时间 |
| `P50 latency` | 请求总延迟的 50 分位 |
| `P95 latency` | 请求总延迟的 95 分位 |
| `TTFT` | 流式模式下首 token 返回时间 |
| `TPOT` | 每输出 token 的平均时间 |
| `ITL` | 相邻 token 间延迟 |
| `error rate` | 非 200 响应占比 |

#### 服务侧指标

| 指标 | 说明 |
|------|------|
| `GPU memory used` | 显存占用 |
| `num_requests_running` | 正在执行的请求数 |
| `num_requests_waiting` | 排队请求数 |
| `gpu_cache_usage_perc` | KV cache / GPU cache 使用率 |
| `prompt throughput` | prompt token 吞吐 |
| `generation throughput` | generation token 吞吐 |

说明：

- `TTFT / TPOT / ITL` 建议优先使用原生 benchmark 输出
- 服务侧指标从 `/metrics` 抓取，而不是靠人工看 `nvidia-smi`
- 结论需要同时参考请求侧和服务侧两层指标

### 2.5 第一版结果产物

- `results/raw/benchmark/`
- `results/raw/prometheus/`
- `results/raw/benchmark/<batch_run_id>/manifest.json`
- `results/baseline_metrics.csv`
- `results/baseline_service_metrics.csv`
- `results/baseline_summary.md`
- `batch validation summary`
- 2 张图：
  - 并发数 vs `QPS`
  - 并发数 vs `P95 latency`

### 2.6 验收标准

- [ ] 24 组 baseline 结果完整落盘
- [ ] 请求侧与服务侧指标都能落盘
- [ ] 至少能稳定复现 2 次
- [ ] 能给出一段文字总结：不同并发、不同上下文长度下的主要变化趋势
- [ ] 能结合服务侧指标解释性能变化原因

---

## Phase 3：双模型基线对比 + 参数调优实验

### 3.1 前提

只在以下条件满足后再开始：

- baseline 压测已完成
- 至少已经有两个候选模型的 `Phase 2` batch 输出，或者已经决定先用当前稳定主模型完成 sweep 开发

### 3.2 第一段：直接复用 Phase 2 做双模型对比

这一步不再单独开一个 `Phase 4`，而是直接复用 `Phase 2` 已经验证过的基线矩阵和聚合链路。

目标：

- 让两个模型在同一组 `Phase 2` workload 下直接对齐
- 输出一份统一的模型对比表和结论摘要
- 用这份结果决定 `Phase 3` 第二段的主模型

对比维度：

- `QPS`
- `P95 latency`
- `tokens/s`
- `error rate`
- `GPU memory used`

产物：

- `results/model_compare/model_compare.csv`
- `results/model_compare/model_compare_summary.md`

验收标准：

- [ ] 两个模型有重叠 workload 的可比结果
- [ ] 输出统一对比表和摘要
- [ ] 能解释模型切换带来的吞吐、延迟和显存差异

### 3.3 第二段：在选定主模型上做参数 sweep

重点关注：

- 并发容量
- 排队情况
- 显存压力
- `QPS / P95 / tokens/s`

### 3.4 重点参数

第一版只扫 3 个：

| 参数 | 实验取值 |
|------|---------|
| `--max-model-len` | 2048, 3072, 4096 |
| `--max-num-batched-tokens` | 2048, 4096, 8192 |
| `--max-num-seqs` | 16, 32, 64 |

`--gpu-memory-utilization` 暂时不放第一版 sweep 主线，除非你后面时间充裕。

### 3.5 固定 workload

统一使用：

- 并发：`8`
- 输入长度：`512`
- 输出长度：`128`
- warmup：`1` 次
- 正式 measured repeat：`2` 次

原因：

- 该 workload 已经在 `Phase 2` 基线矩阵中被覆盖
- 本地开发验证更快，且仍然足以暴露调度和排队差异
- 能减少因为输出过长导致的 sweep 单轮耗时
- 每个参数档位先做 1 次 warmup，再统计 2 次正式 repeat，降低首轮 JIT / graph capture 对调优结论的污染

先保持 workload 不变，避免多变量同时变化。正式 `param_tuning.csv` 与 summary 只汇总 warmup 之后的 measured repeats。

如果 GPU 上已经有常驻模型服务，可以通过更低的 `gpu_memory_utilization` 分批执行 3 组 sweep，再在离线聚合阶段合并多个 manifest。

### 3.6 观测指标

除了压测指标，还可结合 `/metrics` 关注：

- `num_requests_waiting`
- `num_requests_running`
- `gpu_cache_usage_perc`
- `avg_prompt_throughput_toks_per_s`
- `avg_generation_throughput_toks_per_s`

### 3.7 产物

- `results/model_compare/model_compare.csv`
- `results/model_compare/model_compare_summary.md`
- `results/param_tuning/param_tuning.csv`
- `results/param_tuning/param_tuning_summary.md`
- `results/param_tuning/awq_full_gpu_vs_phase2_analysis.md`

### 3.8 合理结论示例

适合写的结论是：

- `max_model_len` 增大后，长上下文支持更强，但显存压力更高，并发容量下降
- `max_num_batched_tokens` 提升后，吞吐可能改善，但高并发下尾延迟也可能上升
- `max_num_seqs` 过高时，排队减少未必能转化为更低延迟，需结合实际 workload 取舍
- 两个模型在同一组 `Phase 2` case 下即使吞吐接近，显存占用和尾延迟也可能体现不同 trade-off

不建议轻易写：

- “KV cache 命中率提升”

除非你确实采到了对应指标并能解释定义。

### 3.9 验收标准

- [ ] 双模型对比结果完整落盘
- [ ] 参数 sweep 结果完整落盘
- [ ] 至少形成 2 到 3 条可信的 trade-off 结论
- [ ] 能解释参数变化为什么会影响吞吐、延迟和显存

---

## 量化策略建议

### 第一版

- 使用官方 AWQ checkpoint
- 目标是完成**量化模型服务对比实验**

### 第二版可选加分

如果主线顺利完成，可以增加：

- 自己复现一次 AWQ 量化流程

但这属于**可选加分项**，不是第一版必做项。

第一版**不建议**：

- 自己实现量化算法
- 自己写量化 kernel
- 上来就做 TensorRT-LLM / 多卡量化部署

---

## 最终成果清单

### 必须产出

- [ ] 可运行的 `vLLM + FastAPI` 服务
- [ ] baseline 压测结果
- [ ] 双模型基线对比结果
- [ ] 主模型参数调优结果
- [ ] 一份可写进简历和面试材料的总结文档

### 最终简历可落的表述

- 基于 vLLM 部署 Qwen 系列模型服务，封装 OpenAI-compatible API，并通过 FastAPI 实现参数校验、异常处理与结构化日志记录
- 基于原生 benchmark 工具与服务侧 metrics 构建可复现实验流程，对不同并发、输入长度和输出长度场景进行压测，统计 QPS、P95 latency、TTFT、TPOT、ITL 与错误率等指标
- 对比 `max_model_len`、`max_num_batched_tokens` 等配置对吞吐、延迟与显存压力的影响，形成推理服务调优结论
- 复用同一套 baseline workload 直接对比两个候选模型的显存占用、吞吐与尾延迟表现，并据此选择后续调优主模型

---

## 执行建议

- GPU 机只跑模型服务与监控
- benchmark runner 与结果聚合可以跑在本地机器或另一台 CPU 机上
- 所有结论必须以实测为准，不要预设“应该提升多少”
- 第一版先做完闭环，再考虑扩展到第二模型、第二量化方案或更复杂参数
