# Phase 2 Runbook

更新时间：2026-06-28

## 当前兼容性结论（2026-06-28）

当前有效环境已经切换到一台 `NVIDIA GeForce RTX 5090 (sm_120, 32 GiB)` 机器。以这台机器为准，新增事实如下：

- 当前可用运行栈：`torch 2.11.0+cu130`、`vllm 0.23.0`、`transformers 5.12.1`、`tokenizers 0.22.2`。
- 旧 `.venv` 中 `flashinfer` 二进制链损坏、旧版 `torch 2.6.0+cu124 / vllm 0.8.5` 不支持 `sm_120`、以及根分区空间不足，是本次环境重建的三层根因。
- 当前 `.venv` 已重建完成，并保留 `.venv.new -> .venv` 符号链接，以兼容 `uv` 生成的不可重定位虚拟环境。
- [bench/config.py](./bench/config.py) 与 [scripts/run_vllm_local.sh](./scripts/run_vllm_local.sh) 当前默认都会优先使用 `/root/autodl-tmp/qwen2.5-0.5b`。
- 已实际跑通一次默认模型 `Phase 2 smoke`：`batch_run_id=phase2smoke-20260628T111721Z`。这次产物位于：
  - `results/raw/benchmark/phase2smoke-20260628T111721Z/`
  - `results/raw/prometheus/phase2smoke-20260628T111721Z/`
  - `results/batches/phase2smoke-20260628T111721Z/`
- 为了让 smoke/小样本批次不会被误判失败：
  - [analysis/aggregate_results.py](./analysis/aggregate_results.py) 现在会把 `num_prompts` 写入 `baseline_metrics.csv`
  - [analysis/validate_batch.py](./analysis/validate_batch.py) 现在按每个 case 的真实 `num_prompts` 校验 `completed`，而不是把 `40` 写死
- [scripts/verify_phase2_smoke.sh](./scripts/verify_phase2_smoke.sh) 与 [scripts/run_phase2_suite.sh](./scripts/run_phase2_suite.sh) 已补回可执行位，runbook 中的 `./scripts/...` 调用现在可直接执行。
- [scripts/run_vllm_local.sh](./scripts/run_vllm_local.sh) 现在默认 `VLLM_DISABLE_LOG_STATS=0`，不再默认传 `--disable-log-stats`。
- [bench/collect_metrics.py](./bench/collect_metrics.py) 已补两层兼容：
  - `kv_cache_usage_perc` 同时支持 `vllm:kv_cache_usage_perc` 与旧名 `vllm:gpu_cache_usage_perc`
  - 当当前 `vllm 0.23.0` 的 `/metrics` 没有吐出任何 `vllm:` 业务指标时，`request_success` 会优先回退到 `http_requests_total`，`prompt/generation/success delta` 会继续回退到 benchmark 结果，避免服务侧 CSV 被整列写成 `0`
- AWQ 的硬件门槛已经不再是当前机器的阻塞项：本机 GPU 计算能力为 `12.0`，而当前安装的 `vllm 0.23.0` 中 `AWQConfig.get_min_capability()` 返回 `75`。
- 本机当前已经存在本地 AWQ 权重目录：`/root/autodl-tmp/qwen2.5-7b-awq`。
- `7B-AWQ baseline` 已实际跑完并校验通过：`batch_run_id=phase2-awq-baseline-20260628T124711Z`。
- 由于首轮 AWQ baseline 只采了 case 前后两个服务侧快照，它仍然适合看 `counter` 类指标，但不足以解释排队、KV cache 压力和服务负载过程。
- 为了解决这个问题，仓库现在已经把服务侧 `gauge` 指标改成“case 运行期间周期采样并聚合 `avg/max/p95`”，对应配置项是 `SERVICE_METRICS_POLL_INTERVAL_S`，默认 `0.5` 秒。
- 使用新采样方式重跑后的完整 AWQ baseline 已校验通过：`batch_run_id=phase2-awq-baseline-resampled-20260628T221900Z`。
- 使用同一新采样链路重跑的非量化 7B baseline 已校验通过：`batch_run_id=phase2-7b-baseline-resampled-20260628T162953Z`，详细分析见 [baseline_detailed_analysis.md](./results/batches/phase2-7b-baseline-resampled-20260628T162953Z/baseline_detailed_analysis.md)。
- 本次 resampled AWQ baseline 的关键产物位于：
  - `results/raw/benchmark/phase2-awq-baseline-resampled-20260628T221900Z/`
  - `results/raw/prometheus/phase2-awq-baseline-resampled-20260628T221900Z/`
  - `results/batches/phase2-awq-baseline-resampled-20260628T221900Z/`
- 使用 `./.venv/bin/python3 analysis/validate_batch.py --batch-run-id phase2-awq-baseline-resampled-20260628T221900Z --output-dir results/batches/phase2-awq-baseline-resampled-20260628T221900Z` 复核后，结果为 `expected_cases=48`、`completed_cases=48`、`failed_cases=0`、`errors=[]`。
- 这次 resampled 结果里，`service_metrics_during_run_sample_count` 在所有 `48` 个 case 上都非零，`num_requests_running_during_run_max` 与 `kv_cache_usage_perc_during_run_max` 也都成功落盘；但 `num_requests_waiting_during_run_max` 与 `server_load_during_run_max` 整批仍为 `0.0`，因此当前更适合用它说明“未观测到服务侧等待队列”和“KV cache 压力较低”，不适合把 `server_load` 当主要解释变量。更完整的数据解读见 [phase2_awq_resampled_baseline_analysis_20260628.md](./phase2_awq_resampled_baseline_analysis_20260628.md)，核心判断是当前 AWQ baseline 主要由输出长度/解码阶段和高并发竞争主导，不是由 KV cache 或 vLLM waiting queue 主导。

下文若提到旧的 `V100S` 或 `RTX 4080` 兼容性背景，均只保留作历史上下文，不再代表当前机器的主结论。


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
cd /GitHub/LLM_test
source .venv/bin/activate
./scripts/setup_phase2_deps.sh
```

## 4. 最小 smoke

如果你要先验证整条 `Phase 2` 链路：

```bash
cd /GitHub/LLM_test
source .venv/bin/activate
READINESS_TIMEOUT_SECONDS=600 \
BENCHMARK_NUM_PROMPTS=2 \
BENCHMARK_STREAM_NUM_PROMPTS=1 \
./scripts/verify_phase2_smoke.sh
```

这个 smoke 脚本现在默认会启用一组更保守的本地开发参数：

- `LOW_VRAM_MODE=1`
- `MODEL_DIR=/root/autodl-tmp/qwen2.5-0.5b`
- `SERVED_MODEL_NAME=qwen-05b-local`
- `MAX_MODEL_LEN=256`
- `GPU_MEMORY_UTILIZATION=0.45`
- `DTYPE=half`
- `VLLM_ENFORCE_EAGER=1`
- `VLLM_CPU_OFFLOAD_GB=4`
- `VLLM_MAX_NUM_SEQS=1`
- `VLLM_MAX_NUM_BATCHED_TOKENS=256`

这组默认值的目的不是拿来做正式 baseline，而是优先提高本地直接运行成功率。
它已经在当前 `RTX 5090` 环境完成一次实际 smoke；更早的 `RTX 4080` / WSL / `9P` 读盘背景仅保留作历史参考。
其中 `MODEL_DIR` 当前默认优先切到 `/root/autodl-tmp/qwen2.5-0.5b`。

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

当前机器已经存在本地 `7B-AWQ` 权重目录 `/root/autodl-tmp/qwen2.5-7b-awq`。如果要直接跑覆盖完整矩阵的正式 `7B-AWQ` baseline，可以先单独起 `vLLM`：

```bash
cd /GitHub/LLM_test
source .venv/bin/activate
LOW_VRAM_MODE=0 \
MODEL_DIR=/root/autodl-tmp/qwen2.5-7b-awq \
SERVED_MODEL_NAME=qwen-7b-awq-local \
MAX_MODEL_LEN=3072 \
GPU_MEMORY_UTILIZATION=0.85 \
VLLM_CPU_OFFLOAD_GB=0 \
VLLM_ENFORCE_EAGER=0 \
./scripts/run_vllm_local.sh
```

另一个 shell 跑 baseline：

```bash
cd /GitHub/LLM_test
source .venv/bin/activate
MODEL_DIR=/root/autodl-tmp/qwen2.5-7b-awq \
SERVED_MODEL_NAME=qwen-7b-awq-local \
VLLM_BASE_URL=http://127.0.0.1:19100 \
SERVICE_MAX_MODEL_LEN=3072 \
./scripts/run_phase2_suite.sh baseline
```

如果只想先跑流式子矩阵：

```bash
MODEL_DIR=/root/autodl-tmp/qwen2.5-7b-awq \
SERVED_MODEL_NAME=qwen-7b-awq-local \
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
