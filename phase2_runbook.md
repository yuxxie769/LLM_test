# Phase 2 Runbook

更新时间：2026-06-27

## 兼容性补充（2026-06-28）

当前 `Phase 2` 已在一台 `Tesla V100S 32GB (sm_70)` 机器上补齐兼容层并重新验证。需要记录的增量结论：

- 本机最终可用栈为：`torch 2.6.0+cu124`、`vllm 0.8.5.post1`、`transformers 4.51.3`、`tokenizers 0.21.1`。
- [bench/benchmark_backends/vllm_bench.py](./bench/benchmark_backends/vllm_bench.py) 已增加版本感知分支：
  - `vllm < 0.9` 时自动改走 `openai-comp + /v1/completions + --random-input-len + --random-output-len`
  - 新版 `vllm` 继续走仓库原本的 chat benchmark 参数
- [bench/config.py](./bench/config.py) 已改为优先本机存在的 `/root/autodl-tmp/qwen2.5-0.5b`。
- 当前这台 `V100S 32GB` 上的正式 `7B` baseline 启动档位已固定为：`LOW_VRAM_MODE=0`、`MODEL_DIR=/root/autodl-tmp/qwen2.5-7b`、`SERVED_MODEL_NAME=qwen-7b-local`、`MAX_MODEL_LEN=3072`、`GPU_MEMORY_UTILIZATION=0.9`、`VLLM_CPU_OFFLOAD_GB=0`、`VLLM_ENFORCE_EAGER=0`。
- [bench/run_single_case.py](./bench/run_single_case.py) 已增加长度预算保护：如果传入 `SERVICE_MAX_MODEL_LEN` 或 `MAX_MODEL_LEN`，当 `input_tokens + output_tokens` 超出该上限时会直接 fail fast，而不是跑到中途才发现服务配置不匹配。
- [bench/run_matrix.py](./bench/run_matrix.py) 现在支持按 `batch_run_id` 断点续跑：若结果目录里已存在对应 case 的 `.combined.json`，再次执行时会自动跳过已完成 case，并在 manifest 中记录 `skipped_existing_cases`。
- `Qwen2.5-7B-Instruct-AWQ` 在当前 `sm_70` 机器上不是“需要进一步调优”，而是启动即报 `The quantization method awq is not supported for the current GPU. Minimum capability: 75. Current capability: 70.`，因此本机不能完成 AWQ baseline。
- 在这组兼容修改后，`Phase 2 smoke` 已重新通过，产物包括：
  - `results/raw/benchmark/phase2smoke-20260627T180045Z/`
  - `results/raw/prometheus/phase2smoke-20260627T180045Z/`
  - `results/batches/phase2smoke-20260627T180045Z/`
- 额外还完成了一次缩小版 `baseline` 矩阵验证：`batch_run_id=phase2-baseline-limit2`，证明 `run_matrix -> aggregate -> render -> plot -> validate` 这条路径也已可用。
- [scripts/run_phase2_suite.sh](./scripts/run_phase2_suite.sh) 已支持通过 `MATRIX_LIMIT` 运行缩小版 suite，用于当前机器上的快速验证，不影响原有全量 baseline 用法。


## 1. Phase 2 当前定位

本阶段不再走“自写 benchmark 框架”路线，而是：

1. 使用 `vLLM` 原生 benchmark 跑单个 case
2. 用仓库内编排脚本展开 workload 矩阵
3. 同时采集请求侧和服务侧两层指标
4. 聚合为 CSV、summary 和图表

## 2. 当前代码入口

- 矩阵定义：[bench/matrix.yaml](./bench/matrix.yaml)
- 单 case 执行：[bench/run_single_case.py](./bench/run_single_case.py)
- 矩阵执行：[bench/run_matrix.py](./bench/run_matrix.py)
- 服务侧指标采集：[bench/collect_metrics.py](./bench/collect_metrics.py)
- 结果聚合：[analysis/aggregate_results.py](./analysis/aggregate_results.py)
- 图表生成：[analysis/plot_baseline.py](./analysis/plot_baseline.py)
- summary 生成：[analysis/render_baseline_summary.py](./analysis/render_baseline_summary.py)
- batch 校验：[analysis/validate_batch.py](./analysis/validate_batch.py)
- smoke 验证脚本：[scripts/verify_phase2_smoke.sh](./scripts/verify_phase2_smoke.sh)
- 正式 suite 跑批脚本：[scripts/run_phase2_suite.sh](./scripts/run_phase2_suite.sh)
- 依赖安装脚本：[scripts/setup_phase2_deps.sh](./scripts/setup_phase2_deps.sh)

## 3. 先装依赖

如果你重建了 `.venv`，先执行：

```bash
cd /mnt/d/LLM_test/LLM_test
source .venv/bin/activate
./scripts/setup_phase2_deps.sh
```

## 4. 最小 smoke

如果你要先验证整条 `Phase 2` 链路：

```bash
cd /mnt/d/LLM_test/LLM_test
source .venv/bin/activate
READINESS_TIMEOUT_SECONDS=600 \
BENCHMARK_NUM_PROMPTS=2 \
BENCHMARK_STREAM_NUM_PROMPTS=1 \
./scripts/verify_phase2_smoke.sh
```

这个 smoke 脚本现在默认会启用一组更保守的本地开发参数：

- `LOW_VRAM_MODE=1`
- `MODEL_DIR=/root/models/qwen2.5-0.5b`
- `SERVED_MODEL_NAME=qwen-05b-local`
- `MAX_MODEL_LEN=256`
- `GPU_MEMORY_UTILIZATION=0.45`
- `DTYPE=half`
- `VLLM_ENFORCE_EAGER=1`
- `VLLM_CPU_OFFLOAD_GB=4`
- `VLLM_MAX_NUM_SEQS=1`
- `VLLM_MAX_NUM_BATCHED_TOKENS=256`

这组默认值已在 `2026-06-27` 的本机 `RTX 4080 16GB` 环境跑通过一次完整 `Phase 2 smoke`。
目的不是拿它做正式 baseline，而是优先提高本地直接运行成功率。
其中 `MODEL_DIR` 默认切到 `/root/models/qwen2.5-0.5b`，是为了绕开 WSL 对 `/mnt/d/...` 的 `9P` 读盘路径；之前正式 `baseline` 启动卡住时，进程等待点就是 `p9_client_rpc`。

成功后会产出：

```text
results/raw/benchmark/<batch_run_id>/
results/raw/prometheus/<batch_run_id>/
results/batches/<batch_run_id>/
```

其中包括：

- `baseline_metrics.csv`
- `baseline_service_metrics.csv`
- `baseline_summary.md`
- `results/raw/benchmark/<batch_run_id>/manifest.json`
- 如果装了 `matplotlib`，还会有 `plots/`
- 自检通过时，脚本会额外打印一份 batch 验证摘要

## 5. 正式 baseline

当前这台 `Tesla V100S 32GB` 上，如果要跑覆盖完整矩阵的正式 `7B` baseline，先单独起 `vLLM`：

```bash
cd /GitHub/LLM_test
source .venv/bin/activate
LOW_VRAM_MODE=0 \
MODEL_DIR=/root/autodl-tmp/qwen2.5-7b \
SERVED_MODEL_NAME=qwen-7b-local \
MAX_MODEL_LEN=3072 \
GPU_MEMORY_UTILIZATION=0.9 \
VLLM_CPU_OFFLOAD_GB=0 \
VLLM_ENFORCE_EAGER=0 \
./scripts/run_vllm_local.sh
```

另一个 shell 跑 baseline：

```bash
cd /GitHub/LLM_test
source .venv/bin/activate
MODEL_DIR=/root/autodl-tmp/qwen2.5-7b \
SERVED_MODEL_NAME=qwen-7b-local \
VLLM_BASE_URL=http://127.0.0.1:19100 \
SERVICE_MAX_MODEL_LEN=3072 \
./scripts/run_phase2_suite.sh baseline
```

如果只想先跑流式子矩阵：

```bash
MODEL_DIR=/root/autodl-tmp/qwen2.5-7b \
SERVED_MODEL_NAME=qwen-7b-local \
VLLM_BASE_URL=http://127.0.0.1:19100 \
SERVICE_MAX_MODEL_LEN=3072 \
./scripts/run_phase2_suite.sh stream_latency
```

低显存 `0.5B` 档位仍然保留，但它现在只用于 smoke 或链路排障，不再作为这台机器的正式 baseline 示例。

如果你要恢复到更接近原始 baseline 的启动档位，不走当前保守默认值，可以显式覆盖：

```bash
unset VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS
LOW_VRAM_MODE=0 \
MODEL_DIR=/mnt/d/models/qwen2.5-7b-awq \
SERVED_MODEL_NAME=qwen-7b-awq-local \
MAX_MODEL_LEN=2048 \
GPU_MEMORY_UTILIZATION=0.8 \
VLLM_CPU_OFFLOAD_GB=0 \
VLLM_ENFORCE_EAGER=0 \
./scripts/run_vllm_local.sh
```

这组恢复值更接近原始配置，但在当前 `RTX 4080 16GB` + WSL 环境下不保证可直接启动成功。

## 6. 产物怎么看

每次正式运行都有独立目录：

```text
results/batches/<batch_run_id>/
```

关键文件：

- `baseline_metrics.csv`
- `baseline_service_metrics.csv`
- `baseline_summary.md`
- `plots/*.png`（如果依赖已装）
- `results/raw/benchmark/<batch_run_id>/manifest.json`

`manifest.json` 现在会显式记录：

- `run_mode`：`single_case` 或 `matrix`
- `planned_cases`
- `completed_cases`
- `failed_cases`

`analysis/validate_batch.py` 会优先用这些字段做批次闭环校验，而不是事后再从矩阵反推。

## 7. 当前已知阻碍

这次在真实 WSL smoke 里已经拿到更具体的结论：

1. `torch / vllm / pandas / matplotlib / seaborn` 现在都已安装
2. `Qwen2.5-7B-Instruct-AWQ` 在当前 `RTX 4080 16GB` 上启动时，出现了真实 `CUDA out of memory`
3. `Qwen2.5-0.5B` 已在 `2026-06-27` 用 `MAX_MODEL_LEN=256`、`GPU_MEMORY_UTILIZATION=0.45`、`VLLM_CPU_OFFLOAD_GB=4` 的低显存参数跑通 `Phase 2 smoke`
4. 上述 7B-AWQ 报错发生在 `2026-06-27`，日志显示：
   - GPU 总显存：`15.99 GiB`
   - 启动当时可用显存：`12.97 GiB`
   - 失败时还差约 `1.02 GiB`
5. 后续再次验证时，启动前快照一度只剩 `587 MiB` 可用显存，因此脚本现在会在 `< 4096 MiB` 时直接 fail fast，而不是继续等待 `vLLM` 超时
6. `2026-06-27` 继续代跑正式 `baseline` 时，`vllm serve` 在某些宿主机会话形态下会卡在 WSL 的 `p9_client_rpc`，表现为：
   - 进程仍存活，但 `/health` 长时间不通
   - 端口 `19100` 没有开始监听
   - `ps` 可见进程状态为 `D`，等待点为 `p9_client_rpc`
   - 当前模型目录位于 `/mnt/d/models/...`，日志里也能看到 checkpoint 文件系统类型是 `9P`
7. 因此当前 runbook 和 `run_vllm_local.sh` 的默认 `0.5B` 模型路径已经切到 `/root/models/qwen2.5-0.5b`；如果你仍想复用 Windows 盘上的模型副本，需要显式传回 `MODEL_DIR=/mnt/d/models/qwen2.5-0.5b`

这说明当前真正阻塞 `Phase 2` 开始开发的不是 WSL 本身，而是：

- 如果要回到 `7B-AWQ`，还需要先释放一部分显存
- 如果只是继续推进链路开发，当前默认已经切到更小的 `0.5B` 本地模型
- 如果正式 `baseline` 启动卡在 `p9_client_rpc`，优先怀疑 `/mnt/d` 的 `9P` 读盘阻塞；更稳妥的方式是把模型放到 WSL Linux 文件系统后再复跑

另外，当前仓库仍位于 `/mnt/d/...`。这不是硬阻塞，但会让首启更慢。
