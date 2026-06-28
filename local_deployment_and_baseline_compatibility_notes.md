# 本机部署与 Baseline 兼容性记录

更新时间：2026-06-28

## 1. 结论先行

当前仓库在这台机器上已经完成：

- `qwen2.5-7b` 的本机部署
- `Phase 1` 本地链路验证
- `Phase 2` 正常性能档全量 baseline / matrix
- `aggregate / summary / plot / validate` 全流程产物生成

当前仓库在这台机器上无法完成：

- `qwen2.5-7b-awq` 的正式 baseline / matrix

原因不是参数不足，而是硬件能力不满足。当前 GPU 为 `Tesla V100S 32GB`，计算能力 `sm_70`，而 AWQ 需要至少 `sm_75`。

## 2. 当前机器与最终可用运行栈

### 2.1 硬件事实

- GPU：`Tesla V100S-PCIE-32GB`
- CUDA Compute Capability：`7.0`（`sm_70`）
- 显存：`32 GiB`

### 2.2 最终可用软件栈

- `torch 2.6.0+cu124`
- `vllm 0.8.5.post1`
- `transformers 4.51.3`
- `tokenizers 0.21.1`

## 3. 本次部署与 baseline 过程中遇到的主要坑

### 3.1 误以为当前机器是较新的消费级卡

最初计划与已有文档中，很多假设来自另一台更强或更新的机器，例如：

- 默认认为可以直接跑 `AWQ`
- 默认认为较新的 `torch + vllm` 组合可用
- 默认认为 `24GB+` 意味着可以直接完成量化对比

实际机器是 `V100 sm_70`，这直接影响：

- 可用 `torch` 版本
- 可用 `vllm` 版本
- 是否支持 AWQ
- baseline 的可行配置上限

### 3.2 新版 Torch / CUDA 组合在本机直接不可用

尝试较新的 `torch 2.11.0+cu130` 后，实际报错：

- `CUDA error: no kernel image is available for execution on the device`

根因：

- 当前卡是 `sm_70`
- 新组合并不覆盖这张卡的可执行 kernel

处理方式：

- 降回 `torch 2.6.0+cu124`
- 配套使用 `vllm 0.8.5.post1`

### 3.3 AWQ 不是“慢”或“需要继续调参”，而是硬件直接不支持

对 `/root/autodl-tmp/qwen2.5-7b-awq` 实测启动后，得到明确错误：

- `The quantization method awq is not supported for the current GPU. Minimum capability: 75. Current capability: 70.`

这说明：

- 当前机器不能完成 AWQ baseline
- 这不是显存、并发、`gpu_memory_utilization` 或 `max_model_len` 能解决的问题
- 要继续 AWQ，必须换到 `sm_75+` 的 GPU

### 3.4 旧版 vLLM benchmark CLI 与仓库原实现不兼容

仓库最初的 benchmark 命令构造偏向较新的 `vllm bench serve`：

- `--backend`
- `--input-len`
- `--output-len`
- `--num-warmups`
- `--temperature`
- chat endpoint 路径

但当前机器上最终可用的是 `vllm 0.8.5.post1`，它要求旧参数集：

- `--endpoint-type openai-comp`
- `--random-input-len`
- `--random-output-len`
- endpoint 必须走 `/v1/completions`

不兼容时会出现：

- `unrecognized arguments`
- benchmark 直接失败

处理方式：

- 在 [bench/benchmark_backends/vllm_bench.py](./bench/benchmark_backends/vllm_bench.py) 增加版本感知兼容分支
- `vllm < 0.9` 自动切换到旧 CLI 参数

### 3.5 最初的“正常档”服务参数其实覆盖不了完整矩阵

曾经启动过一版看似“正常”的 7B 服务：

- `MAX_MODEL_LEN=2048`
- `GPU_MEMORY_UTILIZATION=0.8`
- `LOW_VRAM_MODE=0`

但 baseline 矩阵包含：

- `input_tokens=2048`
- `output_tokens=512`

也就是单请求总预算达到 `2560` tokens。

因此：

- `MAX_MODEL_LEN=2048` 实际覆盖不了完整矩阵
- 即使服务能起，也不该把这轮视为真正的全量 baseline

处理方式：

- 正式 baseline 改为 `MAX_MODEL_LEN=3072`
- `GPU_MEMORY_UTILIZATION=0.9`
- 保持 `LOW_VRAM_MODE=0`

### 3.6 需要给矩阵执行加 fail fast，避免“跑了半天才发现配置不对”

在没有额外保护时，如果：

- 服务 `MAX_MODEL_LEN` 小于矩阵某个 case 的 `input + output`

矩阵脚本仍会继续跑到出错点，浪费时间。

处理方式：

- 在 [bench/run_single_case.py](./bench/run_single_case.py) 增加长度预算保护
- 当传入 `SERVICE_MAX_MODEL_LEN` 或 `MAX_MODEL_LEN` 时，若 `input_tokens + output_tokens` 超出预算，立即报错

### 3.7 全量 baseline 是小时级长跑，必须支持断点续跑

48 个 case 的全量 baseline 在 7B 上耗时很长，尤其是：

- `output=512`
- `input=2048`
- 高并发阶段

如果中途断掉又只能重头开始，会极大拖慢验证速度。

处理方式：

- 在 [bench/run_matrix.py](./bench/run_matrix.py) 增加按 `batch_run_id` 断点续跑能力
- 对同一批次再次执行时，已存在 `.combined.json` 的 case 自动跳过

### 3.8 旧版 benchmark 结果字段与聚合逻辑存在兼容缺口

全量 baseline 跑完后发现生成的 summary 有明显错误，例如：

- `Highest observed P95 latency ... 0.0 ms`

这不是实际结果，而是因为：

- 旧版 `vLLM` 结果文件里是 `p95_e2el_ms` 这类平铺字段
- 原聚合逻辑只读 `percentiles_e2el_ms` 这类新结构

另外最重的两个 case 在原始结果里还出现：

- `completed=38`
- `failed` 为空或 `0`
- 但 `generated_texts` 和 `output_lens` 实际长度是 40

说明旧版输出字段本身也有兼容问题。

处理方式：

- 在 [bench/benchmark_backends/vllm_bench.py](./bench/benchmark_backends/vllm_bench.py) 中：
  - 增加 `p50/p95/p99_*` 平铺字段回退读取
  - 对 `completed` 增加基于 `generated_texts` / `errors` 的归一化修正
- 在 [analysis/aggregate_results.py](./analysis/aggregate_results.py) 中：
  - 聚合时优先根据原始 `.benchmark.json` 重新归一化，而不是盲信旧 combined payload
- 在 [analysis/validate_batch.py](./analysis/validate_batch.py) 中：
  - 增加“`completed < 40` 且 `failed = 0`”的异常检查

### 3.9 单独重渲染 summary 时，模型元数据会回落到默认值

手动重跑 `render_baseline_summary.py` 时，如果不显式带上：

- `MODEL_DIR`
- `SERVED_MODEL_NAME`

summary 头部环境信息会回落到默认 `0.5B` 配置。

处理方式：

- 手动重渲染时显式传：
  - `MODEL_DIR=/root/autodl-tmp/qwen2.5-7b`
  - `SERVED_MODEL_NAME=qwen-7b-local`

## 4. 本次实际落地的关键改动

### 4.1 启动脚本与配置

- [scripts/run_vllm_local.sh](./scripts/run_vllm_local.sh)
  - 正常模式默认 `MAX_MODEL_LEN=3072`
  - 增加旧版 `vllm` 的 `--offload-backend` 探测
  - 默认模型目录优先 `/root/autodl-tmp/qwen2.5-0.5b`
- [bench/config.py](./bench/config.py)
  - 默认模型目录优先 `/root/autodl-tmp/qwen2.5-0.5b`

### 4.2 Benchmark 与矩阵执行

- [bench/benchmark_backends/vllm_bench.py](./bench/benchmark_backends/vllm_bench.py)
  - 增加 legacy CLI 兼容
  - 增加 legacy percentile / completed 字段兼容
- [bench/run_single_case.py](./bench/run_single_case.py)
  - 增加长度预算 fail fast
- [bench/run_matrix.py](./bench/run_matrix.py)
  - 增加按 `batch_run_id` 断点续跑

### 4.3 聚合与校验

- [analysis/aggregate_results.py](./analysis/aggregate_results.py)
  - 基于原始 benchmark json 重新归一化
- [analysis/validate_batch.py](./analysis/validate_batch.py)
  - 增加不完整完成数检查

## 5. 本机上最终确认可用的正式 7B baseline 配置

```bash
MODEL_DIR=/root/autodl-tmp/qwen2.5-7b SERVED_MODEL_NAME=qwen-7b-local LOW_VRAM_MODE=0 MAX_MODEL_LEN=3072 GPU_MEMORY_UTILIZATION=0.9 VLLM_CPU_OFFLOAD_GB=0 VLLM_ENFORCE_EAGER=0 VLLM_PORT=19100 bash ./scripts/run_vllm_local.sh
```

跑全量 baseline：

```bash
MODEL_DIR=/root/autodl-tmp/qwen2.5-7b SERVED_MODEL_NAME=qwen-7b-local VLLM_BASE_URL=http://127.0.0.1:19100 SERVICE_MAX_MODEL_LEN=3072 BATCH_RUN_ID=phase2-baseline-7b-normal-3072 bash ./scripts/run_phase2_suite.sh baseline
```

## 6. 最终状态

### 6.1 已完成

- `qwen2.5-7b` 本机部署完成
- `Phase 2` 正常性能档全量 baseline / matrix 完成
- `manifest`、CSV、summary、plots、validate 全部完成

最终批次：

- [results/batches/phase2-baseline-7b-normal-3072](./results/batches/phase2-baseline-7b-normal-3072)
- [results/raw/benchmark/phase2-baseline-7b-normal-3072](./results/raw/benchmark/phase2-baseline-7b-normal-3072)
- [results/raw/prometheus/phase2-baseline-7b-normal-3072](./results/raw/prometheus/phase2-baseline-7b-normal-3072)

### 6.2 当前机器无法完成

- `qwen2.5-7b-awq` 的正式 baseline / matrix

根因：

- 当前 GPU 为 `sm_70`
- AWQ 至少需要 `sm_75`

## 7. 对后续阶段的建议

如果下一阶段是：

- 基于 7B baseline 做参数 sweep、结论分析、报告整理
  - 当前机器可以直接继续
- 做 `FP16/BF16 vs AWQ` 对比
  - 需要先切到支持 AWQ 的 `sm_75+` GPU
