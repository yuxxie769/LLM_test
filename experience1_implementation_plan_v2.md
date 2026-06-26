# 经历一：LLM 推理服务压测与性能分析平台 — 实现计划 v2

## 目标

这个项目的目标不是“证明我会做底层 CUDA 优化”，而是建立一套**可运行、可压测、可对比、可写进简历**的大模型推理服务实验闭环，最终支撑这 4 条简历表述：

1. 基于 **vLLM** 部署 Qwen 模型服务，并在前面增加轻量 FastAPI 网关  
2. 对不同并发、输入长度、输出长度场景做压测，采集 **QPS / P95 / TTFT / tokens/s / error rate**  
3. 对关键 serving 参数做对比实验，总结吞吐与延迟 trade-off  
4. 对比 **FP16/BF16** 与 **AWQ-INT4** 模型在显存、吞吐与延迟上的差异

---

## 简历 Bullet 对应关系

| # | Bullet | 核心产出 |
|---|--------|---------|
| 1 | vLLM 部署 + FastAPI 网关 | 可运行的推理服务 |
| 2 | Locust 压测 | 压测脚本 + 指标采集 |
| 3 | 参数调优 | 参数对比实验 + 调优结论 |
| 4 | FP16/BF16 vs AWQ-INT4 对比 | 量化对比报告 |

---

## 总体顺序

推荐执行顺序：

```text
Phase 0 → Phase 1 → Phase 2 → Phase 4 → Phase 3
```

原因：

- `Phase 1` 是服务基线，后面都依赖它
- `Phase 2` 先建立 baseline 指标
- `Phase 4` 只需切换模型重跑，更容易先产出结果
- `Phase 3` 需要基于 baseline 和模型选择做参数 sweep，放最后最稳

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

---

## Phase 0：冻结实验定义

### 0.1 固定实验对象

- 模型：`Qwen2.5-7B-Instruct`
- 量化模型：`Qwen2.5-7B-Instruct-AWQ`
- 接口：OpenAI-compatible `/v1/chat/completions`
- 主压测工具：`Locust`

### 0.2 固定第一版指标

- `QPS`
- `P95 latency`
- `TTFT`
- `tokens/s`
- `error rate`
- `GPU memory used`

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

## Phase 2：压测脚本与指标采集

### 2.1 工具选择

第一版建议：

- **Locust**：主压测工具
- **hey**：仅用于 smoke test

第一版先**不使用 `wrk`**，避免维护多套压测逻辑。

### 2.2 压测模式

分两类：

1. **非流式压测**
   - 用来采集 `QPS / P95 / error rate / 请求级 tokens/s`
2. **流式压测**
   - 用来单独采集 `TTFT`

不要试图一套脚本同时把所有指标做得特别完美，第一版分开测更稳。

### 2.3 Locust workload

第一版矩阵：

| 维度 | 取值 |
|------|------|
| 并发数 | 1, 4, 8, 16 |
| 输入长度 | 128, 512, 2048 |
| 输出长度 | 128, 512 |

建议 prompt 模板固定，减少随机波动。

### 2.4 指标定义

| 指标 | 说明 |
|------|------|
| `QPS` | 总请求数 / 总测试时间 |
| `P95 latency` | 请求总延迟的 95 分位 |
| `TTFT` | 流式模式下首 token 返回时间 |
| `tokens/s` | `completion_tokens / 总响应时间` 的汇总统计 |
| `error rate` | 非 200 响应占比 |

说明：

- `TTFT` 应单独在 `stream=True` 下测
- `tokens/s` 第一版按请求级近似统计即可，不必一开始过度细化到 TPOT

### 2.5 第一版结果产物

- `results/baseline_metrics.csv`
- `results/baseline_summary.md`
- 2 张图：
  - 并发数 vs `QPS`
  - 并发数 vs `P95 latency`

### 2.6 验收标准

- [ ] 24 组 baseline 结果完整落盘
- [ ] 至少能稳定复现 2 次
- [ ] 能给出一段文字总结：不同并发、不同上下文长度下的主要变化趋势

---

## Phase 4：FP16/BF16 vs AWQ-INT4 对比

### 4.1 目标

这一步的目标是证明：

- 量化模型显存占用是否明显下降
- 吞吐和延迟是否出现可观察的 trade-off

这一步**不是**证明你自己实现了量化算法。

### 4.2 模型启动方式

```bash
# baseline
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen-7b \
  --dtype float16 \
  --port 8100

# quantized
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen-7b-awq \
  --quantization awq \
  --dtype float16 \
  --port 8100
```

### 4.3 对比维度

- 模型加载显存
- 压测期间峰值显存
- `QPS`
- `P50 / P95 latency`
- `tokens/s`
- `error rate`

### 4.4 实验设计

固定 workload：

- 并发：`1 / 4 / 8 / 16`
- 输入长度：`512`
- 输出长度：`256`

流程：

1. 跑 baseline 模型
2. 记录显存与压测结果
3. 切到 AWQ 模型
4. 记录相同 workload 下结果
5. 输出对比表

### 4.5 产物

- `results/quant_compare.csv`
- `results/quant_compare_summary.md`

### 4.6 注意事项

- 不要在计划里预先写死“显存下降 65%”这类数字
- 所有结论以实测为准
- 简历上可以写“完成 FP16/BF16 与 AWQ-INT4 模型服务对比实验”

### 4.7 验收标准

- [ ] 完成 baseline 与 AWQ 的同 workload 对比
- [ ] 输出对比表和 1 段结论
- [ ] 能解释“显存节省”和“吞吐/延迟变化”之间的关系

---

## Phase 3：参数调优实验

### 3.1 前提

只在以下条件满足后再开始：

- baseline 压测已完成
- AWQ 对比已完成
- 你已经知道哪种模型形态更适合作为主报告对象

### 3.2 观测目标

重点关注：

- 并发容量
- 排队情况
- 显存压力
- `QPS / P95 / tokens/s`

### 3.3 重点参数

第一版只扫 3 个：

| 参数 | 实验取值 |
|------|---------|
| `--max-model-len` | 2048, 4096, 8192 |
| `--max-num-batched-tokens` | 2048, 4096, 8192 |
| `--max-num-seqs` | 16, 32, 64 |

`--gpu-memory-utilization` 暂时不放第一版 sweep 主线，除非你后面时间充裕。

### 3.4 固定 workload

统一使用：

- 并发：`8`
- 输入长度：`512`
- 输出长度：`256`

先保持 workload 不变，避免多变量同时变化。

### 3.5 观测指标

除了压测指标，还可结合 `/metrics` 关注：

- `num_requests_waiting`
- `num_requests_running`
- `gpu_cache_usage_perc`
- `avg_prompt_throughput_toks_per_s`
- `avg_generation_throughput_toks_per_s`

### 3.6 产物

- `results/param_tuning.csv`
- `results/param_tuning_summary.md`

### 3.7 合理结论示例

适合写的结论是：

- `max_model_len` 增大后，长上下文支持更强，但显存压力更高，并发容量下降
- `max_num_batched_tokens` 提升后，吞吐可能改善，但高并发下尾延迟也可能上升
- `max_num_seqs` 过高时，排队减少未必能转化为更低延迟，需结合实际 workload 取舍

不建议轻易写：

- “KV cache 命中率提升”

除非你确实采到了对应指标并能解释定义。

### 3.8 验收标准

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
- [ ] FP16/BF16 vs AWQ-INT4 对比结果
- [ ] 参数调优结果
- [ ] 一份可写进简历和面试材料的总结文档

### 最终简历可落的表述

- 基于 vLLM 部署 Qwen 系列模型服务，封装 OpenAI-compatible API，并通过 FastAPI 实现参数校验、异常处理与结构化日志记录
- 使用 Locust 对不同并发、输入长度和输出长度场景进行压测，统计 QPS、P95 latency、TTFT、tokens/s 与错误率等指标
- 对比 `max_model_len`、`max_num_batched_tokens` 等配置对吞吐、延迟与显存压力的影响，形成推理服务调优结论
- 完成 FP16/BF16 与 AWQ-INT4 模型服务对比实验，分析量化对显存占用、吞吐与延迟表现的影响

---

## 执行建议

- GPU 机只跑模型服务与监控
- Locust 尽量跑在本地机器或另一台便宜的 CPU 机上
- 所有结论必须以实测为准，不要预设“应该提升多少”
- 第一版先做完闭环，再考虑扩展到第二模型、第二量化方案或更复杂参数
