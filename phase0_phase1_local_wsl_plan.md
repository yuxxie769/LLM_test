# Phase 0 + Phase 1 本地 WSL 跑通计划

更新时间：2026-06-27

## 1. 结论先行

本地 Phase 0 + Phase 1 的最终决策如下：

1. 运行环境固定为 `WSL2 + Ubuntu`，不走 Windows 原生 Python，也不把 Docker 作为第一阶段必需层。
2. 主实验定义继续沿用 [experience1_implementation_plan_v2.md](./experience1_implementation_plan_v2.md) 的 `Qwen/Qwen2.5-7B-Instruct` + `Qwen/Qwen2.5-7B-Instruct-AWQ`，但本地冒烟规格单独固定为 `Qwen/Qwen2.5-7B-Instruct-AWQ`。
3. 本地 WSL 中的代码和 Hugging Face 缓存放在 Linux 文件系统，例如 `/home/<user>/workspace/LLM_test`，不从 `/mnt/d/...` 直接跑。
4. Python 环境固定使用全新 `uv` 虚拟环境，Python 版本固定为 `3.12`。
5. Phase 1 只实现轻量 FastAPI 网关：参数校验、Bearer token、`request_id`、结构化日志、vLLM 错误标准化；不在第一版提前引入复杂中间件、Docker 编排或多模型切换。

这样定的核心原因：

- vLLM 官方安装文档当前仍要求 `Linux`，并明确说明 `Windows` 不原生支持，Windows 侧建议走 `WSL`。
- 微软官方建议：Linux 命令行场景下，项目文件放在 WSL 文件系统里性能更好；当前仓库位于 `D:\LLM_test\LLM_test`，如果直接从 `/mnt/d/...` 跑，会和推荐路径相反。
- 当前机器是 `RTX 4080 16GB`，而原始总计划把 `24GB GPU` 视为最低可行规格；因此本地不能把“完整 7B 非量化基线”当成 Phase 1 跑通前提。
- `Qwen/Qwen2.5-7B-Instruct-AWQ` 的官方 Hugging Face 页面直接给了 `pip install vllm` 和 `vllm serve "Qwen/Qwen2.5-7B-Instruct-AWQ"` 的用法，最适合作为本地 WSL 跑通模型。

## 2. 输入与约束

### 2.1 已读本仓库文档

- [experience1_implementation_plan_v2.md](./experience1_implementation_plan_v2.md)

关键信息：

- 主线顺序固定为 `Phase 0 -> Phase 1 -> Phase 2 -> Phase 4 -> Phase 3`
- 主实验模型固定为：
  - `Qwen/Qwen2.5-7B-Instruct`
  - `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Phase 1 的目标是先把 `vLLM + FastAPI` 跑通，再进入压测和对比实验
- 原始文档给出的最低可行 GPU 是 `24GB`

### 2.2 本机观察结果

以 2026-06-27 的本机检查结果为准：

- Windows 侧 `nvidia-smi` 可见 GPU：`NVIDIA GeForce RTX 4080`
- 显存总量：`16376 MiB`
- 驱动版本：`610.47`
- CUDA UMD 版本：`13.3`
- 当时已有约 `3998 MiB` 显存被其他进程占用
- `wsl --version` 可返回版本信息，说明 WSL 组件本身已安装，当前可见版本为：
  - `WSL 2.6.3.0`
  - `Kernel 6.6.87.2-1`
  - `WSLg 1.0.71`
- `wsl --status` 显示默认版本是 `WSL 2`
- 用户已确认主账号中已有 `Ubuntu` 发行版
- 但当前命令是在 `CodexSandboxOffline` 账号上下文执行的；该上下文下 `wsl -l -v` 没有列出已注册发行版，因此我无法在当前沙箱里直接读取你主账号的 WSL 发行版列表

这里有一个必须写清楚的推论：

- `RTX 4080 16GB` 满足 vLLM 对 NVIDIA GPU 计算能力 `7.5+` 的要求
- 但它不满足原始总计划里“24GB 起步”的本地完整主实验要求
- 所以本地 WSL 的目标应当定义为“Phase 1 跑通和网关验收”，而不是“本地完成完整 baseline 实验”

## 3. 决策过程

### 3.1 运行环境怎么选

| 方案 | 结论 | 原因 |
|---|---|---|
| Windows 原生 Python + vLLM | 否决 | vLLM 官方文档明确写了不原生支持 Windows |
| WSL2 + Ubuntu + Python 直装 vLLM | 采用 | 官方支持路径最短，定位问题简单，最适合第一阶段跑通 |
| WSL2 + Docker + vLLM | 暂不采用 | 需要再叠一层容器 GPU 工具链，第一阶段只会增加变量 |

最终决策：

- Phase 1 先走 `WSL2 + Ubuntu + uv + vLLM`
- Docker 只作为后续复现/封装选项，不进入 Phase 1 验收口径

### 3.2 代码放哪儿跑

| 方案 | 结论 | 原因 |
|---|---|---|
| 直接从 `/mnt/d/LLM_test/LLM_test` 跑 | 否决 | 微软官方明确建议 Linux CLI 工作负载把文件放在 WSL 文件系统中以获得更好的性能 |
| 拷贝/克隆到 `/home/<user>/workspace/LLM_test` | 采用 | 符合官方建议，减少 I/O 和路径兼容问题 |

最终决策：

- 当前 Windows 仓库继续保留
- 在 WSL 内再放一份工作副本，例如 `/home/<user>/workspace/LLM_test`

### 3.3 本地跑什么模型

| 方案 | 结论 | 原因 |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` 作为本地 Phase 1 主模型 | 否决 | 原始总计划已把 24GB 视为最低可行规格；当前 16GB 本地卡不应把非量化 7B 基线作为必跑项 |
| `Qwen/Qwen2.5-7B-Instruct-AWQ` 作为本地 Phase 1 主模型 | 采用 | 与主实验模型家族一致，且官方模型页直接给了 vLLM 启动示例 |
| `Qwen/Qwen2.5-3B-Instruct` | 仅保底回退 | 如果 7B-AWQ 仍因显存或环境问题不稳定，可用来只验证链路，不改变主实验定义 |

最终决策：

- 主实验规格不改，仍然保留 `7B Instruct + 7B AWQ`
- 本地 WSL 冒烟规格改为先跑 `7B AWQ`
- 只有在 `7B AWQ` 仍然无法稳定启动时，才降到 `3B` 做纯链路验证

### 3.4 Python 环境怎么建

| 方案 | 结论 | 原因 |
|---|---|---|
| 复用已有 Conda / PyTorch 环境 | 否决 | vLLM 官方明确建议新环境，并提醒 conda 下的 PyTorch/NCCL 组合可能引发兼容问题 |
| `uv venv --python 3.12` 新环境 | 采用 | 官方文档显式推荐，且 Python 3.12 在当前文档里是正向路径 |

最终决策：

- 使用全新 `uv` 虚拟环境
- 不复用已有 Conda 环境
- 不在 Phase 1 里引入“先装 PyTorch 再拼 vLLM”的路径

## 4. Phase 0 最终冻结结果

Phase 0 在这里需要拆成两层，避免“主实验规格”和“本地 WSL 冒烟规格”混淆。

### 4.1 主实验规格

这部分保持与原始总计划一致，不改：

- 主模型：`Qwen/Qwen2.5-7B-Instruct`
- 量化模型：`Qwen/Qwen2.5-7B-Instruct-AWQ`
- 接口：OpenAI-compatible `/v1/chat/completions`
- 主压测工具：`Locust`
- 关键指标：`QPS / P95 latency / TTFT / tokens/s / error rate / GPU memory used`
- 第一版 workload：
  - 并发：`1, 4, 8, 16`
  - 输入长度：`128, 512, 2048`
  - 输出长度：`128, 512`

这部分冻结后，不因为本地机器只有 16GB 显存就修改主实验对象。

### 4.2 本地 WSL 冒烟规格

本地只为跑通 Phase 1，因此单独冻结一套更窄的规格：

- 运行环境：`WSL2 + Ubuntu`
- 模型：`Qwen/Qwen2.5-7B-Instruct-AWQ`
- 单卡运行
- 上下文上限先压到 `2048`
- `gpu_memory_utilization` 先压到 `0.85`
- 只验证：
  - vLLM 服务可启动
  - `/health`
  - `/v1/models`
  - `/v1/chat/completions`
  - FastAPI 网关透传与错误标准化

### 4.3 Phase 0 最终决策理由

这样冻结的原因是：

- 主实验规格服务于后面的压测、量化对比和参数 sweep，不能因为本地显存不足被动改小
- 本地 WSL 的目标是把环境、模型服务、网关链路先跑通，优先减少变量
- `7B AWQ` 与后续正式实验的模型家族一致，比直接降到 `3B` 更接近真实主线

## 5. Phase 1 执行方案

### 5.1 Windows 侧准备

1. 先释放显存。
   当前检查时约有 `3998 MiB` 显存被其他程序占用。跑本地 WSL 前，优先关闭 Stable Diffusion、GPU 加速浏览器标签页、桌面叠加层等高占用程序。
2. 先在你的主账号里确认 Ubuntu 发行版版本与状态。

```powershell
wsl --status
wsl -l -v
```

当前前提按“你的主账号里已经有 Ubuntu”处理；如果 `wsl -l -v` 中该发行版版本是 `2`，直接跳到下一节。

只有当你的主账号里确实没有可用发行版时，才执行安装：

```powershell
wsl --install -d Ubuntu-22.04
```

如果系统不识别 `Ubuntu-22.04`，则按微软文档先看在线列表：

```powershell
wsl --list --online
```

再执行：

```powershell
wsl --install -d <DistroName>
```

3. 重启机器并完成首次 Ubuntu 用户初始化。
4. 再次验证：

```powershell
wsl --status
wsl -l -v
```

期望结果：

- 默认版本是 `2`
- 至少有一个 Ubuntu 发行版，且版本是 `2`

### 5.2 WSL 内基础环境

进入 Ubuntu 后先做 GPU 和目录准备：

```bash
export PATH=/usr/lib/wsl/lib:$PATH
nvidia-smi
mkdir -p ~/workspace
cd ~/workspace
```

说明：

- NVIDIA 官方文档明确提到，在 WSL 下找不到 `nvidia-smi` 时，优先使用 `/usr/lib/wsl/lib/nvidia-smi` 或把 `/usr/lib/wsl/lib` 加入 `PATH`
- 工作目录放在 `~/workspace` 下，不从 `/mnt/d/...` 直接跑

如果要把当前仓库带入 WSL，推荐：

```bash
cp -r /mnt/d/LLM_test/LLM_test ~/workspace/LLM_test
cd ~/workspace/LLM_test
```

### 5.3 Python 与依赖安装

先安装系统依赖：

```bash
sudo apt update
sudo apt install -y build-essential git curl
```

再安装 `uv` 并创建新环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
```

安装运行时依赖：

```bash
uv pip install vllm --torch-backend=auto
uv pip install fastapi "uvicorn[standard]" httpx pydantic-settings orjson
```

这里明确不做的事：

- 不预装独立 PyTorch 再手工拼 vLLM
- 不复用旧 Conda 环境
- 不在 Phase 1 里引入 Docker

### 5.4 启动本地 vLLM 服务

首选启动命令：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen-7b-awq-local \
  --host 0.0.0.0 \
  --port 8100 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85
```

这样定的原因：

- `7B-AWQ` 比原始非量化 `7B` 更适合本地 16GB 卡
- `2048` 已足够覆盖主实验第一版 workload 中最大的输入长度
- `0.85` 比原始计划中的 `0.9` 更保守，优先提高首轮启动成功率

直接验收命令：

```bash
curl http://127.0.0.1:8100/health
curl http://127.0.0.1:8100/v1/models
curl -X POST http://127.0.0.1:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data '{
    "model": "qwen-7b-awq-local",
    "messages": [
      {"role": "user", "content": "请用一句话介绍你自己。"}
    ],
    "max_tokens": 64
  }'
```

### 5.5 FastAPI 网关最小实现范围

建议目录：

```text
llm-bench/
├── gateway/
│   ├── main.py
│   ├── schemas.py
│   ├── logger.py
│   └── config.py
├── logs/
└── results/
```

网关职责只做这 5 件事：

1. Bearer token 校验
2. 请求体参数校验
3. 生成或透传 `request_id`
4. 将请求转发到 `http://127.0.0.1:8100/v1/chat/completions`
5. 将 vLLM 报错转换为统一 JSON 错误格式，并落结构化日志

明确不做：

- 不做多模型路由
- 不做鉴权系统集成
- 不做复杂 middleware 链
- 不做数据库落盘

建议环境变量：

```bash
export VLLM_BASE_URL=http://127.0.0.1:8100
export GATEWAY_TOKEN=local-dev-token
export GATEWAY_LOG_PATH=./logs/gateway.jsonl
```

建议启动方式：

```bash
uvicorn gateway.main:app --host 0.0.0.0 --port 8000
```

建议验收命令：

```bash
curl http://127.0.0.1:8000/healthz
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer local-dev-token" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "qwen-7b-awq-local",
    "messages": [
      {"role": "user", "content": "请返回一个 JSON，字段只有 ok。"}
    ],
    "max_tokens": 64
  }'
```

## 6. Phase 1 验收口径

只有以下项目全部满足，才算“本地 WSL Phase 1 跑通”：

- [ ] `wsl -l -v` 可见 Ubuntu，且版本是 `2`
- [ ] WSL 内 `nvidia-smi` 可用
- [ ] `vllm serve` 能稳定启动 `Qwen/Qwen2.5-7B-Instruct-AWQ`
- [ ] `http://127.0.0.1:8100/health` 返回成功
- [ ] `http://127.0.0.1:8100/v1/models` 返回模型列表
- [ ] 直连 `vLLM /v1/chat/completions` 有有效响应
- [ ] 通过 FastAPI 网关调用同一接口也有有效响应
- [ ] 无效请求能返回结构化 `4xx`
- [ ] 至少落一份 `gateway.jsonl` 结构化日志样本

## 7. 失败时的回退路径

### 7.1 `7B-AWQ` 启动失败

按顺序处理：

1. 先清理 Windows 侧 GPU 占用
2. 把 `--max-model-len` 从 `2048` 降到 `1024`
3. 把 `--gpu-memory-utilization` 从 `0.85` 降到 `0.8`
4. 仍失败时，回退到：

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --served-model-name qwen-3b-local \
  --host 0.0.0.0 \
  --port 8100 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8
```

但要在记录里明确写清楚：

- 这是“链路保底方案”
- 不代表主实验规格已变更
- 不代表后续 Phase 4 可直接在本地完成

### 7.2 WSL 内看不到 GPU

按顺序处理：

1. 确认 Windows 侧 `nvidia-smi` 正常
2. 在 WSL 内执行：

```bash
export PATH=/usr/lib/wsl/lib:$PATH
nvidia-smi
```

3. 如果仍失败，优先检查：
   - Windows NVIDIA 驱动是否过旧
   - WSL 是否需要 `wsl --update`
   - 是否误在 WSL 内安装了 Linux NVIDIA 显卡驱动

### 7.3 从 `/mnt/d/...` 跑出现卡顿或奇怪路径问题

直接切回 Linux 文件系统路径，不做额外优化尝试：

- 正确路径：`/home/<user>/workspace/LLM_test`
- 不推荐路径：`/mnt/d/LLM_test/LLM_test`

## 8. 最终决策摘要

最终方案不是“把整套主实验硬塞进本地 16GB 卡”，而是：

- 主实验规格继续保持 `7B Instruct + 7B AWQ`
- 本地 WSL 先以 `7B AWQ` 跑通 `vLLM + FastAPI`
- 代码与缓存都放在 WSL Linux 文件系统
- 环境固定为 `WSL2 + Ubuntu + uv + Python 3.12`
- Phase 1 只做轻量网关，不提前进入容器化和复杂工程化

这是当前约束下成功率最高、且不会污染后续主实验定义的做法。

## 9. 参考资料

- vLLM GPU 安装文档：<https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>
- vLLM Online Serving 文档：<https://docs.vllm.ai/en/latest/serving/online_serving/>
- Microsoft WSL 安装文档：<https://learn.microsoft.com/en-us/windows/wsl/install>
- Microsoft WSL 文件系统与性能说明：<https://learn.microsoft.com/en-us/windows/wsl/filesystems>
- NVIDIA CUDA on WSL User Guide：<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
- Qwen `Qwen2.5-7B-Instruct-AWQ` 模型页：<https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ>
