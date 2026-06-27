# Phase 1 详情实施计划

更新时间：2026-06-27

## 1. 目标

本文件用于把 [phase0_phase1_local_wsl_plan.md](./phase0_phase1_local_wsl_plan.md) 中的 `Phase 1` 从“方向性方案”收敛成可直接执行的实施说明。

本阶段只完成两件事：

1. 跑通本地 `vLLM` 模型服务
2. 在其前面补一层最小可用的 `FastAPI` 网关

本阶段不做：

- 多模型路由
- 压测脚本
- 参数 sweep
- 数据库存储
- Docker 化

## 2. 当前已确认事实

### 2.1 运行环境

- 当前系统为 `WSL2 + Ubuntu-22.04`
- WSL 内 `/dev/dxg` 可见
- WSL 内 `nvidia-smi` 正常
- 本地 GPU 为 `NVIDIA GeForce RTX 4080 16GB`
- 当前仓库路径：`/mnt/d/LLM_test/LLM_test`
- 当前虚拟环境路径：`/mnt/d/LLM_test/LLM_test/.venv`

### 2.2 已安装依赖

当前 `.venv` 内已确认：

- `Python 3.12.13`
- `torch 2.11.0+cu130`
- `vllm 0.23.0`
- `fastapi 0.136.3`
- `uvicorn 0.49.0`
- `httpx 0.28.1`
- `pydantic-settings 2.14.2`
- `orjson 3.11.9`

### 2.3 当前端口现实约束

在这台机器当前状态下，以下端口已确认不可绑定：

- `8000`
- `8100`
- `8101`
- `18000`

已确认可绑定的高位端口包括：

- `18080`
- `19100`
- `19101`

补充发现：

- 通过当前工具做跨会话本地联调时，端口联通性会受会话隔离影响
- 仓库中已补充单-shell 验证脚本，用来规避这类工具侧网络假象

因此本轮实际验收优先采用：

- `vLLM`：`19100`
- `gateway`：`18080`

### 2.4 本地模型资产

本阶段固定使用本地模型目录，不再依赖联网拉取：

- 模型目录：`/mnt/d/models/qwen2.5-7b-awq`

已确认该目录中存在：

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`
- `model.safetensors.index.json`

从 `config.json` 可确认：

- 架构：`Qwen2ForCausalLM`
- 量化方式：`AWQ`
- 量化位宽：`4-bit`
- `torch_dtype`：`float16`

## 3. Phase 1 最终范围

### 3.1 vLLM 服务范围

本阶段的模型服务只要求：

- 单模型
- 单卡
- OpenAI-compatible API
- 本地监听 `19100`
- 支持：
  - `GET /health`
  - `GET /v1/models`
  - `POST /v1/chat/completions`

### 3.2 网关范围

网关只要求：

- `GET /healthz`
- `POST /v1/chat/completions`
- Bearer token 校验
- 请求体参数校验
- `request_id` 透传或生成
- 将请求转发到 `vLLM`
- 将 `vLLM` 错误统一映射为结构化 JSON
- 落一份 JSONL 结构化日志

## 4. 仓库内交付物

本阶段完成后，仓库里应至少新增以下内容：

```text
gateway/
├── __init__.py
├── main.py
├── schemas.py
├── config.py
└── logger.py

logs/
results/
scripts/
├── run_vllm_local.sh
├── run_gateway_local.sh
├── smoke_test_phase1.sh
└── verify_phase1_local.sh
```

说明：

- `logs/` 用于网关日志样本
- `results/` 先保留空目录，为后续压测阶段占位
- `run_vllm_local.sh` 固化本地模型启动命令
- `run_gateway_local.sh` 固化本地网关启动命令
- `smoke_test_phase1.sh` 固化最小验收命令
- `verify_phase1_local.sh` 在单个 shell 中完成 `vLLM -> gateway -> smoke` 端到端验证

## 5. 执行顺序

严格按下面顺序做，不跳步：

1. 固化 `vLLM` 启动脚本
2. 用本地模型目录启动 `vLLM`
3. 验证 `vLLM /health`、`/v1/models`、`/v1/chat/completions`
4. 再实现 `FastAPI` 网关
5. 验证网关接口与结构化日志

原因：

- 如果 `vLLM` 本体没跑通，网关层的任何报错都会混淆问题归因
- 只有先把模型服务基线固定住，后面的错误标准化和日志才有意义

## 6. vLLM 实施细节

### 6.1 固定启动参数

本地 `vLLM` 启动参数固定为：

```bash
vllm serve /mnt/d/models/qwen2.5-7b-awq \
  --served-model-name qwen-7b-awq-local \
  --host 0.0.0.0 \
  --port 19100 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8
```

这里明确改成“本地目录路径”而不是 HF 模型名，原因是：

- 已有完整本地模型
- 可以减少联网变量
- 能避免缓存目录与下载失败问题

当前机器上的额外收敛结论：

- 需要显式设置 `CUDA_HOME` 指向 `.venv` 内的 CUDA 工具链
- 需要显式设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，绕开本机 WSL 环境下 `flashinfer` sampler 的 JIT 编译问题
- 当前桌面常驻进程会占用一部分显存，因此首轮稳定值应下调到 `GPU_MEMORY_UTILIZATION=0.8`

### 6.2 启动脚本要求

`scripts/run_vllm_local.sh` 需要满足：

- 自动激活当前仓库 `.venv`
- 默认使用 `/mnt/d/models/qwen2.5-7b-awq`
- 默认监听 `19100`
- 支持通过环境变量覆盖模型路径与端口
- 启动前打印关键配置

建议环境变量：

```bash
export MODEL_DIR=/mnt/d/models/qwen2.5-7b-awq
export SERVED_MODEL_NAME=qwen-7b-awq-local
export VLLM_PORT=19100
export MAX_MODEL_LEN=2048
export GPU_MEMORY_UTILIZATION=0.8
export VLLM_USE_FLASHINFER_SAMPLER=0
```

### 6.3 vLLM 验收命令

```bash
curl http://127.0.0.1:19100/health
curl http://127.0.0.1:19100/v1/models
curl -X POST http://127.0.0.1:19100/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data '{
    "model": "qwen-7b-awq-local",
    "messages": [
      {"role": "user", "content": "请用一句话介绍你自己。"}
    ],
    "max_tokens": 64
  }'
```

## 7. FastAPI 网关设计

### 7.1 目录

```text
gateway/
├── __init__.py
├── main.py
├── schemas.py
├── config.py
└── logger.py
```

### 7.2 文件职责

`gateway/config.py`

- 读取环境变量
- 定义默认值
- 暴露全局配置对象

`gateway/schemas.py`

- 定义请求模型
- 定义错误响应模型
- 限制最基本的字段合法性

`gateway/logger.py`

- 封装 JSONL 日志写入
- 统一日志字段结构

`gateway/main.py`

- 定义 `FastAPI app`
- 定义 `/healthz`
- 定义 `/v1/chat/completions`
- 做 token 校验
- 调用 `vLLM`
- 组装统一响应/错误

### 7.3 网关环境变量

```bash
export VLLM_BASE_URL=http://127.0.0.1:19100
export GATEWAY_TOKEN=local-dev-token
export GATEWAY_LOG_PATH=./logs/gateway.jsonl
export GATEWAY_PORT=18080
```

### 7.4 统一错误格式

第一版统一成：

```json
{
  "error": {
    "message": "human readable message",
    "type": "invalid_request_error",
    "code": "bad_request",
    "request_id": "..."
  }
}
```

至少覆盖：

- 缺少 `Authorization`
- Bearer token 错误
- 请求体字段缺失
- `vLLM` 不可达
- `vLLM` 返回非 2xx

### 7.5 结构化日志字段

第一版固定记录：

- `timestamp`
- `request_id`
- `path`
- `model`
- `max_tokens`
- `status_code`
- `latency_ms`
- `error_type`

有能力稳定提取时再追加：

- `input_tokens`
- `output_tokens`

## 8. 验收脚本设计

`scripts/smoke_test_phase1.sh` 需要覆盖：

1. 直连 `vLLM /health`
2. 直连 `vLLM /v1/models`
3. 直连 `vLLM /v1/chat/completions`
4. 网关 `GET /healthz`
5. 网关 `POST /v1/chat/completions`
6. 一个无 token 的失败请求

脚本失败即返回非零退出码。

如果要在单个 shell 中完成端到端验收，直接运行：

```bash
GPU_MEMORY_UTILIZATION=0.8 ./scripts/verify_phase1_local.sh
```

## 9. 实施边界

### 9.1 本阶段不做

- 流式返回
- `/v1/completions`
- `/v1/embeddings`
- SSE 中继
- 中间件拆分
- 请求限流
- Prometheus 指标
- 数据库存储

### 9.2 允许的简化

- 同步返回，不做流式
- 只转发 `chat/completions`
- 日志直接写本地 JSONL 文件
- 只支持单一 Bearer token

## 10. 完成标准

只有以下条件全部满足，Phase 1 才算完成：

- [ ] `scripts/run_vllm_local.sh` 可启动本地模型目录
- [ ] `vLLM /health` 正常
- [ ] `vLLM /v1/models` 正常
- [ ] `vLLM /v1/chat/completions` 正常
- [ ] `gateway/` 代码完整落地
- [ ] `GET /healthz` 正常
- [ ] 通过网关调用 `POST /v1/chat/completions` 正常
- [ ] 缺 token 或非法请求返回结构化 `4xx`
- [ ] `logs/gateway.jsonl` 至少生成一条样本
- [ ] `scripts/smoke_test_phase1.sh` 可一次跑完基础验收
- [ ] `scripts/verify_phase1_local.sh` 可在单 shell 中跑完端到端验证

## 11. 本文件与原计划的关系

本文件是 `Phase 1` 的执行层文档。

- 总体边界与约束以 [phase0_phase1_local_wsl_plan.md](./phase0_phase1_local_wsl_plan.md) 为准
- 本文件负责把其中 `5.5` 与 `5.6` 转成可执行步骤
- 如两者冲突，以“更贴近当前机器真实状态和已验证路径”的写法为准
