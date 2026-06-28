# Phase 2 详情实施计划

更新时间：2026-06-27

## 兼容性补充（2026-06-28）

在当前 `Tesla V100S 32GB` 机器上，`Phase 2` 的主要兼容性问题不是链路设计，而是 `vllm bench serve` 的 CLI 代际差异。当前已确认：

- 仓库原始实现按较新的 `vllm bench serve` 参数构造命令：`--backend`、`--input-len`、`--output-len`、`--num-warmups`、`--temperature`。
- 本机实际可用的 `vllm 0.8.5.post1` 只支持旧参数集：
  - `--endpoint-type openai-comp`
  - `--random-input-len`
  - `--random-output-len`
  - endpoint 也必须走 `/v1/completions`
- 因此 [bench/benchmark_backends/vllm_bench.py](./bench/benchmark_backends/vllm_bench.py) 已增加版本感知兼容层：
  - `vllm < 0.9` 时自动切旧 CLI
  - 新版 `vllm` 仍保留原命令路径
- 同时 [bench/config.py](./bench/config.py) 已改为优先使用本机存在的 `/root/autodl-tmp/qwen2.5-0.5b`，避免 `run_matrix.py` 回落到失效的 `/root/models/...` 默认值。
- 当前这台 `V100S 32GB` 上，如果要覆盖 `baseline` 矩阵里的 `input=2048, output=512` 组合，服务侧 `MAX_MODEL_LEN` 不能再停在 `2048`。当前正式 `7B` baseline 已改用 `MAX_MODEL_LEN=3072`、`GPU_MEMORY_UTILIZATION=0.9` 的正常档位。
- [bench/run_single_case.py](./bench/run_single_case.py) 已增加长度预算保护：传入 `SERVICE_MAX_MODEL_LEN` 或 `MAX_MODEL_LEN` 时，若 `input_tokens + output_tokens` 超过该预算会直接报错，避免跑半天才发现服务配置覆盖不了矩阵。
- [bench/run_matrix.py](./bench/run_matrix.py) 现在支持按 `batch_run_id` 断点续跑：若结果目录里已存在对应 case 的 `.combined.json`，再次执行时会自动跳过已完成 case，并在 manifest 中记录 `skipped_existing_cases`。
- `Qwen2.5-7B-Instruct-AWQ` 在当前 `sm_70` 机器上已确认是硬件不支持，而不是显存调优问题。真实报错是：`The quantization method awq is not supported for the current GPU. Minimum capability: 75. Current capability: 70.`


## 1. 目标

本文件用于把 [experience1_implementation_plan_v2.md](./experience1_implementation_plan_v2.md) 中的 `Phase 2` 从方向性目标收敛成可直接开发的执行文档。

本阶段的正确定位不是“自己写一个 LLM benchmark 系统”，而是：

- 选用现成 benchmark 工具跑单个测试点
- 自己实现很薄的矩阵编排层
- 同步采集请求侧和服务侧两层指标
- 输出可复现、可对比、可解释的 baseline 结果

本阶段只做四件事：

1. 封装原生 benchmark 命令
2. 落地 workload 矩阵编排
3. 落地两层指标采集与结果落盘
4. 产出 baseline 汇总表、图表和结论摘要

本阶段不做：

- 自写主 benchmark 引擎
- `FP16/BF16 vs AWQ-INT4` 模型对比
- serving 参数 sweep
- 完整 Grafana 大盘建设
- 分布式压测平台

## 2. 当前基线

### 2.1 已完成前置

`Phase 1` 已完成，当前仓库内已具备：

- 本地 `vLLM` 启动脚本：[scripts/run_vllm_local.sh](./scripts/run_vllm_local.sh)
- 本地网关启动脚本：[scripts/run_gateway_local.sh](./scripts/run_gateway_local.sh)
- 端到端验收脚本：[scripts/verify_phase1_local.sh](./scripts/verify_phase1_local.sh)
- 网关实现：[gateway/main.py](./gateway/main.py)
- Phase 1 验收说明：[phase1_runbook.md](./phase1_runbook.md)

当前默认运行基线：

- 正式 `Phase 2 baseline`：`/root/autodl-tmp/qwen2.5-7b`
- 模型名：`qwen-7b-local`
- `vLLM`：`127.0.0.1:19100`
- `gateway`：`127.0.0.1:18080`
- 正常档位：`MAX_MODEL_LEN=3072`、`gpu_memory_utilization=0.9`
- `Qwen2.5-7B-Instruct-AWQ` 仍保留为目标对比模型，但在当前 `sm_70` 机器上不可运行

### 2.2 当前缺口

当前仓库还没有以下 `Phase 2` 交付物：

- `bench/` benchmark 编排目录
- `analysis/` 汇总分析目录
- matrix 配置文件
- benchmark runner
- `/metrics` 抓取脚本
- baseline 汇总 CSV 和摘要文档

### 2.3 当前环境阻碍

当前还没有确认以下能力已经就绪：

- `.venv` 中是否可直接使用 `vllm bench serve`
- `vLLM` 当前启动方式下 `/metrics` 是否可稳定抓取
- 用于结果汇总的 `pandas / matplotlib / seaborn` 是否已安装

已确认还未安装的 Python 依赖：

- `pandas`
- `matplotlib`
- `seaborn`

补充说明：

- `Locust` 不再是 `Phase 2` 主依赖
- 当前仓库仍在 `/mnt/d/LLM_test/LLM_test`，可以继续开发编排脚本，但正式长时间压测更建议迁到 WSL Linux 文件系统
- 当前本地 GPU 为 `RTX 4080 16GB`，更适合先完成 `AWQ` baseline；后续 `Phase 4` 的非量化对比不应默认放在这台机器上做正式结论

## 3. Phase 2 最终范围

### 3.1 执行结构

本阶段采用两层结构：

1. **benchmark 执行层**
   - 负责执行单个固定测试点
   - 优先使用 `vLLM` 原生 benchmark
   - 必要时可引入 `SGLang` 或 `GenAI-Perf` 做交叉验证
2. **实验编排层**
   - 负责展开多维矩阵
   - 逐组调用 benchmark 命令
   - 同步抓取 `/metrics`
   - 统一落盘原始结果和汇总结果

这意味着：

- 多维压测矩阵仍然保留
- 但你自己写的是“实验调度器”，不是“benchmark 引擎”

### 3.2 压测入口

本阶段主压测入口优先定为服务本体，而不是自定义网关：

- 主入口：`vLLM` OpenAI-compatible API
- 地址：`http://127.0.0.1:19100`

原因：

- 原生 benchmark 与服务本体对齐最直接
- 指标语义和官方文档一致
- 先减少网关层附加变量，baseline 更干净

补充约定：

- `gateway` 仍然保留，用于链路验证和后续统一入口
- 如果后面要评估网关额外开销，再单独做一组 A/B 对比

### 3.3 两层指标

本阶段固定采两层指标。

#### 请求侧指标

这层反映“用户看到的表现”，由 benchmark 输出：

- `QPS`
- `P50 latency`
- `P95 latency`
- `TTFT`
- `TPOT`
- `ITL`
- `error rate`

#### 服务侧指标

这层反映“服务内部状态”，由 `vLLM /metrics` 抓取：

- `GPU memory used`
- `num_requests_running`
- `num_requests_waiting`
- `gpu_cache_usage_perc`
- `avg_prompt_throughput_toks_per_s`
- `avg_generation_throughput_toks_per_s`

这两层必须一起看。否则你只能看到“变慢了”，却解释不了为什么变慢。

### 3.4 workload 矩阵

正式 baseline 仍沿用主计划中的 24 组矩阵：

| 维度 | 取值 |
|---|---|
| 并发数 | `1, 4, 8, 16` |
| 输入长度 | `128, 512, 2048` |
| 输出长度 | `128, 512` |

总计：

- `4 x 3 x 2 = 24` 组非流式 baseline

流式指标建议使用单独子矩阵：

| 维度 | 取值 |
|---|---|
| 并发数 | `1, 4, 8` |
| 输入长度 | `128, 2048` |
| 输出长度 | `128` |

原因：

- `TTFT / TPOT / ITL` 更依赖流式模式
- 第一版先形成趋势，不需要把所有维度全量笛卡尔积

### 3.5 分层执行策略

为了适配当前机器条件，`Phase 2` 分成两层：

1. 本地开发验证层
   - 目标：命令正确、结果格式正确、目录结构正确
   - 推荐规模：`并发 1/4 + 输入 128/512 + 输出 128`
2. 正式 baseline 产出层
   - 目标：测满 24 组并至少复现 2 次
   - 前提：服务稳定、显存稳定、长时间运行不中断

## 4. 仓库交付物

本阶段完成后，仓库内应至少新增：

```text
bench/
├── matrix.yaml
├── config.py
├── prompts.py
├── run_matrix.py
├── run_single_case.py
├── benchmark_backends/
│   ├── __init__.py
│   ├── vllm_bench.py
│   └── sglang_bench.py
└── collect_metrics.py

analysis/
├── aggregate_results.py
├── plot_baseline.py
└── render_baseline_summary.py

results/
├── raw/
│   ├── benchmark/
│   └── prometheus/
├── baseline_metrics.csv
├── baseline_service_metrics.csv
└── baseline_summary.md
```

文件职责如下：

- `bench/matrix.yaml`
  - 定义 baseline 和流式子矩阵
- `bench/config.py`
  - 统一读取环境变量和路径配置
- `bench/prompts.py`
  - 生成固定 token 长度 prompt
- `bench/run_single_case.py`
  - 跑单个 case
- `bench/run_matrix.py`
  - 展开矩阵并串行执行全部 case
- `bench/benchmark_backends/vllm_bench.py`
  - 封装 `vLLM` 原生 benchmark 调用
- `bench/benchmark_backends/sglang_bench.py`
  - 预留可选交叉验证实现
- `bench/collect_metrics.py`
  - 抓取并解析 `/metrics`
- `analysis/aggregate_results.py`
  - 汇总 benchmark 输出与服务侧指标
- `analysis/plot_baseline.py`
  - 产出基础图表
- `analysis/render_baseline_summary.py`
  - 产出摘要文档

## 5. 任务清单

- [ ] `T1` 确认主 benchmark 工具及命令参数
- [ ] `T2` 安装分析依赖并固化安装命令
- [ ] `T3` 新建 `bench/` 与 `analysis/` 目录
- [ ] `T4` 实现固定长度 prompt 生成逻辑
- [ ] `T5` 实现单 case benchmark runner
- [ ] `T6` 实现 matrix 配置与矩阵执行器
- [ ] `T7` 实现 `/metrics` 抓取与解析
- [ ] `T8` 实现请求侧 + 服务侧结果聚合
- [ ] `T9` 生成基线图表与摘要
- [ ] `T10` 小矩阵验证一次，再跑正式 baseline

## 6. 具体实施规划

### 6.1 T1 benchmark 工具确认

第一版主方案建议固定为：

- `vLLM` 原生 benchmark

理由：

- 与当前 serving 后端一致
- 指标定义最贴近官方实现
- 少引入额外变量

可选交叉验证工具：

- `SGLang bench_serving`
- `GenAI-Perf`

但它们不应进入第一版主线。

### 6.2 T2 依赖安装

需要先补装分析依赖：

```bash
source .venv/bin/activate
python -m pip install pandas matplotlib seaborn
```

如果后续确认 `vllm bench serve` 当前环境不可直接使用，再补充对应依赖或运行方式说明。

第一版不建议一开始引入：

- `jupyter`
- `plotly`
- 完整 Grafana 部署

### 6.3 T3 目录初始化

新建目录时，保持后续扩展兼容：

```text
bench/
bench/benchmark_backends/
analysis/
results/raw/benchmark/
results/raw/prometheus/
```

要求：

- benchmark 原始输出和服务侧 metrics 分开
- 每次跑批都写独立时间戳目录
- 不把不同轮次结果覆盖写在同一个文件里

建议路径约定：

```text
results/raw/benchmark/<run_id>/
results/raw/prometheus/<run_id>/
```

### 6.4 T4 Prompt 生成

`bench/prompts.py` 不应手写一堆超长文本，而应使用 tokenizer 做长度控制。

建议实现方式：

1. 读取本地 tokenizer：
   - `/mnt/d/models/qwen2.5-7b-awq`
2. 准备一个中文基础段落
3. 循环拼接基础段落
4. 用 tokenizer 截断到目标 token 长度
5. 再 decode 回字符串作为最终 prompt

附加要求：

- prompt 内容固定，不要随机采样
- 每个输入长度只保留一个模板，减少波动

### 6.5 T5 单 case runner

`bench/run_single_case.py` 负责执行一个固定 case，例如：

- 并发 `8`
- 输入 `512`
- 输出 `128`
- 模式 `non_stream`

它内部只做几件事：

1. 读取 case 参数
2. 生成 prompt
3. 调用 benchmark backend
4. 记录 benchmark 原始输出
5. 触发一次 `/metrics` 抓取
6. 保存当前 case 的元数据

这里不要混入矩阵循环逻辑。

### 6.6 T6 benchmark backend 封装

`bench/benchmark_backends/vllm_bench.py` 负责把仓库内部 case 参数转换成实际 benchmark 命令参数。

你真正需要统一的是输入和输出协议：

- 输入：
  - `concurrency`
  - `input_tokens`
  - `output_tokens`
  - `mode`
  - `duration`
- 输出：
  - 原始 benchmark stdout/stderr
  - 标准化后的 benchmark JSON

这样后面即便切到 `SGLang`，也只需要换 backend 适配层。

### 6.7 T7 矩阵编排

`bench/matrix.yaml` 建议采用声明式定义：

```yaml
baseline:
  backend: vllm
  mode: non_stream
  repeat: 2
  dimensions:
    concurrency: [1, 4, 8, 16]
    input_tokens: [128, 512, 2048]
    output_tokens: [128, 512]

stream_latency:
  backend: vllm
  mode: stream
  repeat: 2
  dimensions:
    concurrency: [1, 4, 8]
    input_tokens: [128, 2048]
    output_tokens: [128]
```

`bench/run_matrix.py` 负责：

1. 读取矩阵配置
2. 展开 case 列表
3. 为每个 case 分配 `run_id`
4. 串行调用 `run_single_case.py`
5. 记录成功/失败状态

第一版建议串行执行，不要先上并行调度。

### 6.8 T8 两层指标采集

`bench/collect_metrics.py` 负责抓取 `vLLM /metrics`，并解析出本阶段需要的服务侧字段。

第一版实现要求：

1. benchmark case 开始前抓一次
2. benchmark case 结束后抓一次
3. 对关键指标做：
   - `before`
   - `after`
   - `delta` 或 `peak` 近似记录

说明：

- 第一版不要求完整 time series 监控系统
- 但必须把服务侧指标纳入正式结果，不然“为什么延迟变差”很难解释

### 6.9 T9 结果聚合

`analysis/aggregate_results.py` 负责把两层结果合成标准表。

请求侧结果建议字段：

| 字段 | 说明 |
|---|---|
| `run_id` | 批次编号 |
| `concurrency` | 并发数 |
| `input_tokens` | 输入长度 |
| `output_tokens` | 输出长度 |
| `mode` | `non_stream` / `stream` |
| `qps` | 吞吐 |
| `p50_latency_ms` | 50 分位延迟 |
| `p95_latency_ms` | 95 分位延迟 |
| `ttft_ms` | 首 token 延迟 |
| `tpot_ms` | 平均每 token 时间 |
| `itl_ms` | token 间延迟 |
| `error_rate` | 错误率 |

服务侧结果建议字段：

| 字段 | 说明 |
|---|---|
| `run_id` | 批次编号 |
| `gpu_memory_used_mb` | 显存 |
| `num_requests_running` | 运行中请求 |
| `num_requests_waiting` | 等待请求 |
| `gpu_cache_usage_perc` | cache 使用率 |
| `prompt_throughput_toks_per_s` | prompt 吞吐 |
| `generation_throughput_toks_per_s` | generation 吞吐 |

最终至少产出：

- `results/baseline_metrics.csv`
- `results/baseline_service_metrics.csv`

### 6.10 T10 图表与摘要

`analysis/plot_baseline.py` 第一版只画两张请求侧图：

1. 并发数 vs `QPS`
2. 并发数 vs `P95 latency`

再补一张服务侧图即可：

3. 并发数 vs `num_requests_waiting` 或 `gpu_cache_usage_perc`

`analysis/render_baseline_summary.py` 至少要回答三类问题：

1. 并发升高时吞吐如何变化
2. 长输入下尾延迟如何变化
3. 这些变化在服务侧指标上对应什么现象

## 7. 开发前检查项

只有以下条件满足，才进入正式开发：

- [ ] `Phase 1` 仍可通过 `./scripts/verify_phase1_local.sh`
- [ ] `.venv` 可正常激活
- [ ] 本地模型目录 `/mnt/d/models/qwen2.5-7b-awq` 可读
- [ ] `vllm bench serve` 可调用或已有明确替代运行方式
- [ ] `pandas / matplotlib / seaborn` 已安装
- [ ] `/metrics` 可稳定抓取
- [ ] `logs/` 与 `results/` 可写

正式跑 baseline 前，再追加：

- [ ] 机器空闲显存足够
- [ ] `vLLM` 端口无冲突
- [ ] 长时间运行不受代理或终端超时影响

## 8. 验收标准

只有以下条件全部满足，`Phase 2` 才算完成：

- [ ] 24 组 baseline 请求侧结果完整落盘
- [ ] 流式子矩阵结果完整落盘
- [ ] 服务侧指标结果完整落盘
- [ ] 生成 `results/baseline_metrics.csv`
- [ ] 生成 `results/baseline_service_metrics.csv`
- [ ] 生成 `results/baseline_summary.md`
- [ ] 至少生成 3 张基础图表
- [ ] 同一配置至少复现 2 次，趋势一致
- [ ] 能用服务侧指标解释至少 2 条请求侧性能变化

## 9. 当前最小启动条件

如果只问“现在还差什么才能开始开发 `Phase 2`”，答案很简单：

1. 确认 `vLLM` 原生 benchmark 的实际命令和输出格式
2. 安装 `pandas + matplotlib + seaborn`
3. 新建 `bench/`、`analysis/` 并先实现单 case runner
4. 验证 `/metrics` 能稳定抓到需要的字段

这四项补齐后，就可以正式开工。
