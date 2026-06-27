# Phase 1 Runbook

更新时间：2026-06-27

## 1. Phase 1 做了什么

本阶段已完成：

1. 在 `WSL2 + Ubuntu-22.04` 下跑通本地 `vLLM`
2. 直接使用本地模型目录 `/mnt/d/models/qwen2.5-7b-awq`
3. 在仓库内落地最小 `FastAPI` 网关实现
4. 固化启动脚本、冒烟脚本、端到端验收脚本
5. 修掉当前机器上的几类实际阻碍：
   - 默认端口冲突
   - `FlashInfer` sampler JIT 失败
   - `.venv` 内 CUDA 工具链未被自动发现
   - 代理环境影响本地 `127.0.0.1` 请求
   - 网关转发 `None` 字段导致的 `400`

## 2. 关键文件

- 详情计划：[phase1_detailed_implementation_plan.md](./phase1_detailed_implementation_plan.md)
- 原始总计划更新版：[phase0_phase1_local_wsl_plan.md](./phase0_phase1_local_wsl_plan.md)
- 网关入口：[gateway/main.py](./gateway/main.py)
- vLLM 启动脚本：[scripts/run_vllm_local.sh](./scripts/run_vllm_local.sh)
- 网关启动脚本：[scripts/run_gateway_local.sh](./scripts/run_gateway_local.sh)
- 冒烟脚本：[scripts/smoke_test_phase1.sh](./scripts/smoke_test_phase1.sh)
- 端到端验收脚本：[scripts/verify_phase1_local.sh](./scripts/verify_phase1_local.sh)

## 3. 当前默认运行参数

- 模型目录：`/mnt/d/models/qwen2.5-7b-awq`
- 服务模型名：`qwen-7b-awq-local`
- 默认 `vLLM` 端口：`19100`
- 默认网关端口：`18080`
- 默认 `gpu_memory_utilization`：`0.8`
- 默认关闭 `FlashInfer` sampler：`VLLM_USE_FLASHINFER_SAMPLER=0`

## 4. 你怎么验收

先进入仓库并激活环境：

```bash
cd /mnt/d/LLM_test/LLM_test
source .venv/bin/activate
```

### 4.1 一条命令验收整条链路

```bash
GPU_MEMORY_UTILIZATION=0.8 ./scripts/verify_phase1_local.sh
```

通过标准：

- 脚本最终输出 `phase1 verification complete`
- 中间能看到：
  - `vLLM /health`
  - `vLLM /v1/models`
  - `vLLM /v1/chat/completions`
  - `gateway /healthz`
  - `gateway authorized chat`
  - `gateway unauthorized chat should fail`

### 4.2 分步验收

先起 `vLLM`：

```bash
GPU_MEMORY_UTILIZATION=0.8 ./scripts/run_vllm_local.sh
```

另开一个 shell，验证：

```bash
curl --noproxy "*" http://127.0.0.1:19100/health
curl --noproxy "*" http://127.0.0.1:19100/v1/models
curl --noproxy "*" -X POST http://127.0.0.1:19100/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data '{
    "model": "qwen-7b-awq-local",
    "messages": [{"role": "user", "content": "请用一句话介绍你自己。"}],
    "max_tokens": 64
  }'
```

再起网关：

```bash
./scripts/run_gateway_local.sh
```

再验证网关：

```bash
curl --noproxy "*" http://127.0.0.1:18080/healthz
curl --noproxy "*" -X POST http://127.0.0.1:18080/v1/chat/completions \
  -H "Authorization: Bearer local-dev-token" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "qwen-7b-awq-local",
    "messages": [{"role": "user", "content": "请返回一个 JSON，字段只有 ok。"}],
    "max_tokens": 64
  }'
```

无 token 验证：

```bash
curl --noproxy "*" -X POST http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data '{
    "model": "qwen-7b-awq-local",
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 16
  }'
```

预期返回结构化 `401`。

## 5. 日志验收

成功跑完一次网关请求后，检查：

```bash
tail -n 20 logs/gateway.jsonl
```

至少应看到：

- 一条 `status_code: 200`
- 一条 `status_code: 401`

## 6. 后续入口

Phase 1 收尾后，下一步建议直接进入：

1. Phase 2 原生 benchmark 命令封装
2. workload 矩阵编排
3. 请求侧 + 服务侧两层指标采集与结果汇总
