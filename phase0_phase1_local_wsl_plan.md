# Phase 0 + Phase 1 本地 WSL 跑通计划

更新时间：2026-06-27

## 兼容性补充（2026-06-28）

新增一组必须记录的本机兼容性结论：

- 当前实际部署机器不是 `RTX 4080 16GB`，而是 `Tesla V100S 32GB`，计算能力为 `sm_70`。
- 这张卡无法运行本仓库先前假设的 `torch 2.11.0+cu130 + vllm 0.23.0` 组合；最小 `torch.zeros(..., device="cuda")` 就会报 `CUDA error: no kernel image is available for execution on the device`。
- 为了在这台机器上跑通环境，运行栈已切换为：
  - `torch 2.6.0+cu124`
  - `vllm 0.8.5.post1`
  - `transformers 4.51.3`
  - `tokenizers 0.21.1`
- 因为 `vllm 0.8.5` 的 CLI 明显早于当前仓库脚本假设：
  - `vllm serve` 不接受 `--offload-backend auto`
  - `vllm bench serve` 不接受 `--backend / --input-len / --output-len / --num-warmups / --temperature`
  - 旧版 `bench serve` 只能走 `openai-comp + /v1/completions + --random-input-len + --random-output-len`
- 因此仓库内已补一层版本兼容适配，确保 `Phase 1` 与 `Phase 2` 都能在 `sm_70` + 旧版 `vLLM` 组合上继续工作。


## 1. 结论先行

本地 Phase 0 + Phase 1 的最终决策如下：

1. 运行环境固定为 `WSL2 + Ubuntu`，不走 Windows 原生 Python，也不把 Docker 作为第一阶段必需层。
2. 主实验定义继续沿用 [experience1_implementation_plan_v2.md](./experience1_implementation_plan_v2.md) 的 `Qwen/Qwen2.5-7B-Instruct` + `Qwen/Qwen2.5-7B-Instruct-AWQ`，但本地冒烟规格单独固定为 `Qwen/Qwen2.5-7B-Instruct-AWQ`。
3. 本地 WSL 中的代码和 Hugging Face 缓存放在 Linux 文件系统，例如 `/home/<user>/workspace/LLM_test`，不从 `/mnt/d/...` 直接跑。
4. Python 环境固定使用全新 `uv` 虚拟环境，Python 版本固定为 `3.12`。
5. Phase 1 只实现轻量 FastAPI 网关：参数校验、Bearer token、`request_id`、结构化日志、vLLM 错误标准化；不在第一版提前引入复杂中间件、Docker 编排或多模型切换。
6. 本地已有完整模型目录 `D:\models\qwen2.5-7b-awq`，Phase 1 优先直接使用本地目录，不再依赖联网拉取 Hugging Face 模型。

这样定的核心原因：

- vLLM 官方安装文档当前仍要求 `Linux`，并明确说明 `Windows` 不原生支持，Windows 侧建议走 `WSL`。
- 微软官方建议：Linux 命令行场景下，项目文件放在 WSL 文件系统里性能更好；当前仓库位于 `D:\LLM_test\LLM_test`，如果直接从 `/mnt/d/...` 跑，会和推荐路径相反。
- 当前机器是 `RTX 4080 16GB`，而原始总计划把 `24GB GPU` 视为最低可行规格；因此本地不能把“完整 7B 非量化基线”当成 Phase 1 跑通前提。
- `Qwen/Qwen2.5-7B-Instruct-AWQ` 的官方 Hugging Face 页面直接给了 `pip install vllm` 和 `vllm serve "Qwen/Qwen2.5-7B-Instruct-AWQ"` 的用法，最适合作为本地 WSL 跑通模型。
- 当前机器上还额外验证到：`8000/8100/8101/18000` 这几个端口不可稳定绑定，Phase 1 需要改用高位端口。

## 2. 输入与约束

### 2.1 已读本仓库文档

- [experience1_implementation_plan_v2.md](./experience1_implementation_plan_v2.md)

关键信息：

- 主线顺序固定为 `Phase 0 -> Phase 1 -> Phase 2 -> Phase 4 -> Phase 3`
- 主实验模型固定为：
  - `Qwen/Qwen2.5-7B-Instruct`
  - `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Phase 1 的目标是先把 `vLLM + FastAPI` 跑通，再进入压测和对比实验
- Phase 2 已调整为“原生 benchmark + 矩阵编排 + 两层指标采集”，而不是自写 benchmark 主框架
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
- 当前命令会话本身运行在 `WSL2` 内核上，而不是 Windows 原生命令行
- 当前工作目录实际位于 `/mnt/d/LLM_test/LLM_test`，仍然是 Windows 文件系统挂载路径
- 在不受沙箱限制的系统检查里，`/dev/dxg` 存在，WSL 内 `nvidia-smi` 可正常返回，说明 GPU 透传本身是正常的
- Windows 侧 `nvidia-smi` 也可正常返回，且 `wsl -l -v` 可见 `Ubuntu-22.04` 与 `docker-desktop`，版本都为 `2`
- 之前在受限会话里出现过 `GPU access blocked by the operating system`，该结果已确认是沙箱假阴性，不代表机器本身的 WSL GPU 配置异常
- 用户已确认主账号中已有 `Ubuntu` 发行版
- 本地模型目录 `/mnt/d/models/qwen2.5-7b-awq` 已确认包含 `config.json`、`tokenizer.json`、两片 `safetensors` 权重和 `model.safetensors.index.json`
- 当前桌面常驻进程会占用一部分显存；在实际跑通时，`gpu_memory_utilization=0.85` 不稳定，`0.8` 可以稳定通过
- `vLLM 0.23.0` 在当前 WSL 环境下默认会走 `FlashInfer` top-k/top-p sampler；该路径会触发 JIT 编译问题，需要显式关闭
- 在 2026-06-27 的一次真实 Phase 2 smoke 中，`Qwen2.5-7B-Instruct-AWQ` 启动失败的根因已明确为真实显存不足，而不是 WSL 假阴性：
  - GPU 总显存：`15.99 GiB`
  - 启动时可用显存：`12.97 GiB`
  - 失败分配：约 `1.02 GiB`
  - 结论：当前仍有约 `3 GiB` 显存被其他进程占用，导致 7B-AWQ 在本机 16GB 卡上无法稳定完成启动
- 为了避免 Phase 2 smoke 在显存明显不够时空等，脚本已加入 `nvidia-smi` 启动前快照和 fail-fast 预检；最近一次预检直接读到仅剩 `587 MiB` 空闲显存，因此脚本会在 `< 2048 MiB` 时立刻退出

这里有一个必须写清楚的推论：

- `RTX 4080 16GB` 满足 vLLM 对 NVIDIA GPU 计算能力 `7.5+` 的要求
- 但它不满足原始总计划里“24GB 起步”的本地完整主实验要求
- 所以本地 WSL 的目标应当定义为“Phase 1 跑通和网关验收”，而不是“本地完成完整 baseline 实验”
- 当前真正需要处理的不是 WSL 或 GPU 透传，而是把运行目录迁到 Linux 文件系统，并在非受限环境里完成依赖安装
- 当前如果要马上推进 `Phase 2`，还需要先释放额外显存；否则应临时切到更小模型只做链路开发
- 在当前仓库路径下也可以跑通，但要接受 `9P` 文件系统带来的较慢首启，并在运行参数上做额外收敛

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
- 如果本地已有完整权重目录，Phase 1 优先直接用本地目录路径而不是 HF 模型名

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
- 主 benchmark 工具：优先 `vLLM` 原生 benchmark，必要时用 `SGLang` / `GenAI-Perf` 做交叉验证
- `Locust` 仅作为网关层补充验证工具
- 关键指标分两层：
  - 请求侧：`QPS / P50 latency / P95 latency / TTFT / TPOT / ITL / error rate`
  - 服务侧：`GPU memory used / num_requests_running / num_requests_waiting / gpu_cache_usage_perc / prompt throughput / generation throughput`
- 第一版 workload：
  - 并发：`1, 4, 8, 16`
  - 输入长度：`128, 512, 2048`
  - 输出长度：`128, 512`

这部分冻结后，不因为本地机器只有 16GB 显存就修改主实验对象。

### 4.2 本地 WSL 冒烟规格

本地只为跑通 Phase 1，因此单独冻结一套更窄的规格：

- 运行环境：`WSL2 + Ubuntu`
- 模型：`/mnt/d/models/qwen2.5-7b-awq`
- 单卡运行
- 上下文上限先压到 `2048`
- `gpu_memory_utilization` 先压到 `0.8`
- `VLLM_USE_FLASHINFER_SAMPLER=0`
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

### 5.2 WSL 内预检

在 Ubuntu 里先不要急着安装 vLLM，先确认 GPU 设备层没有问题：

```bash
export PATH=/usr/lib/wsl/lib:$PATH
ls -l /dev/dxg
nvidia-smi
pwd
```

预期结果：

- `/dev/dxg` 存在
- `nvidia-smi` 正常返回
- 如果当前目录仍在 `/mnt/d/...`，说明你还没切到 WSL Linux 文件系统

补充说明：

- 某些受限沙箱或代理会话里，即便机器本身正常，也可能让 `nvidia-smi` 报 `GPU access blocked by the operating system`
- 如果你是在这类受限会话中执行命令，需要回到自己的正常 Ubuntu 终端复核，不要据此误判 WSL GPU 异常
- 仓库中已补充 `scripts/wsl_preflight.sh`，可直接复用这组预检

如果这里不过，先不要进入 Python 和依赖安装阶段，优先处理 GPU 透传问题。

### 5.3 WSL 内基础环境与目录

预检通过后，再做目录准备：

```bash
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

### 5.4 Python 与依赖安装

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

如果想一条命令完成“复制到 Linux 文件系统 + 预检 + 建环境 + 安装依赖”，可直接运行：

```bash
./scripts/setup_local_wsl.sh
```

这里明确不做的事：

- 不预装独立 PyTorch 再手工拼 vLLM
- 不复用旧 Conda 环境
- 不在 Phase 1 里引入 Docker

### 5.4.1 当前已确认可直接复用的环境

当前仓库目录 `/mnt/d/LLM_test/LLM_test` 内已经完成以下安装：

- `.venv`
- `Python 3.12.13`
- `torch 2.11.0+cu130`
- `vllm 0.23.0`
- `fastapi 0.136.3`
- `uvicorn 0.49.0`
- `httpx 0.28.1`
- `pydantic-settings 2.14.2`
- `orjson 3.11.9`

因此后续 Phase 1 默认直接复用当前仓库下的 `.venv`。

### 5.5 启动本地 vLLM 服务

当前机器上，最终跑通的不是文档最初写的 `8100 + 0.85`，而是下面这组收敛后的参数。

最终建议启动命令：

```bash
CUDA_HOME=/mnt/d/LLM_test/LLM_test/.venv/lib/python3.12/site-packages/nvidia/cu13 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve /mnt/d/models/qwen2.5-7b-awq \
  --served-model-name qwen-7b-awq-local \
  --host 0.0.0.0 \
  --port 19100 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8
```

这样定的原因与实际遇到的问题：

- 本地已有完整模型目录，直接用本地路径可以减少联网变量
- 当前机器上 `8000/8100/8101/18000` 不可稳定绑定，所以改用高位端口
- 在实际首启中，`0.85` 会因为桌面常驻进程占用显存而失败，`0.8` 可稳定通过
- `vLLM 0.23.0` 在当前 WSL 环境里默认会走 `FlashInfer` sampler，该路径会触发 JIT 编译问题；显式设置 `VLLM_USE_FLASHINFER_SAMPLER=0` 后可稳定绕过
- `flashinfer`/`nvcc` 路径默认不会自动指向 `.venv` 内 CUDA 工具链，因此需要显式设置 `CUDA_HOME`

已在仓库中固化为脚本：

```bash
./scripts/run_vllm_local.sh
```

直接验收命令：

```bash
curl --noproxy "*" http://127.0.0.1:19100/health
curl --noproxy "*" http://127.0.0.1:19100/v1/models
curl --noproxy "*" -X POST http://127.0.0.1:19100/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data '{
    "model": "qwen-7b-awq-local",
    "messages": [
      {"role": "user", "content": "请用一句话介绍你自己。"}
    ],
    "max_tokens": 64
  }'
```

### 5.6 FastAPI 网关最小实现范围

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
4. 将请求转发到本地 `vLLM /v1/chat/completions`
5. 将 vLLM 报错转换为统一 JSON 错误格式，并落结构化日志

明确不做：

- 不做多模型路由
- 不做鉴权系统集成
- 不做复杂 middleware 链
- 不做数据库落盘

建议环境变量：

```bash
export VLLM_BASE_URL=http://127.0.0.1:19100
export GATEWAY_TOKEN=local-dev-token
export GATEWAY_LOG_PATH=./logs/gateway.jsonl
export GATEWAY_PORT=18080
```

建议启动方式：

```bash
./scripts/run_gateway_local.sh
```

实现时额外需要注意：

- 网关内部对 `httpx` 必须显式 `trust_env=False`，否则本机代理环境会把 `127.0.0.1` 请求也带偏，导致错误的 `503`
- 转发请求体时要 `exclude_none=True`，避免把空字段一并转发给 `vLLM`，造成不必要的 `400`

建议验收命令：

```bash
curl --noproxy "*" http://127.0.0.1:18080/healthz
curl --noproxy "*" -X POST http://127.0.0.1:18080/v1/chat/completions \
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

如果要在单个 shell 中完成端到端验收，可直接运行：

```bash
GPU_MEMORY_UTILIZATION=0.8 ./scripts/verify_phase1_local.sh
```

## 6. Phase 1 验收口径

只有以下项目全部满足，才算“本地 WSL Phase 1 跑通”：

- [ ] `wsl -l -v` 可见 Ubuntu，且版本是 `2`
- [ ] WSL 内 `/dev/dxg` 可见
- [ ] WSL 内 `nvidia-smi` 可用
- [ ] `vllm serve` 能稳定启动 `/mnt/d/models/qwen2.5-7b-awq`
- [ ] `http://127.0.0.1:19100/health` 返回成功
- [ ] `http://127.0.0.1:19100/v1/models` 返回模型列表
- [ ] 直连 `vLLM /v1/chat/completions` 有有效响应
- [ ] 通过 FastAPI 网关调用同一接口也有有效响应
- [ ] 无效请求能返回结构化 `4xx`
- [ ] 至少落一份 `gateway.jsonl` 结构化日志样本
- [ ] `./scripts/verify_phase1_local.sh` 可在单 shell 中跑完整条链路

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
  --port 19100 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8
```

但要在记录里明确写清楚：

- 这是“链路保底方案”
- 不代表主实验规格已变更
- 不代表后续 Phase 4 可直接在本地完成
- 如果 `19100` 也被占用，则继续改用空闲高位端口，或直接复用 `./scripts/verify_phase1_local.sh` 的动态端口分配逻辑

### 7.1.1 本次实际遇到的具体问题

本次 Phase 1 真正遇到并修掉的问题如下：

1. `GPU access blocked by the operating system`
   - 结论：这是受限沙箱里的假阴性，不是机器本身的 WSL GPU 故障
   - 修法：回到不受限会话复核 `/dev/dxg` 与 `nvidia-smi`

2. `8100/8101/8000` 端口绑定失败
   - 结论：当前机器上这些端口不可稳定绑定
   - 修法：改用高位端口，并在脚本里支持动态分配端口

3. `flashinfer` sampler JIT 编译失败
   - 现象：`vLLM` 在 warmup 期间因 `FlashInfer` top-k/top-p sampler 构建失败而退出
   - 修法：设置 `VLLM_USE_FLASHINFER_SAMPLER=0`

4. `nvcc` / `CUDA_HOME` 找不到
   - 现象：`flashinfer` 或相关 CUDA 组件找不到默认 `/usr/local/cuda`
   - 修法：显式把 `CUDA_HOME` 指向 `.venv/lib/python3.12/site-packages/nvidia/cu13`

5. 网关 `503`
   - 现象：网关明明访问的是 `127.0.0.1`，但健康检查返回 `503`
   - 原因：`httpx` 默认继承了代理环境变量
   - 修法：在网关里对 `httpx.AsyncClient` 显式设置 `trust_env=False`

6. 网关转发 `400`
   - 现象：直连 `vLLM` 成功，但通过网关转发得到 `400`
   - 原因：转发体里包含了 `None` 字段
   - 修法：`payload.model_dump(..., exclude_none=True)`

### 7.2 WSL 内看不到 GPU

按顺序处理：

1. 确认 Windows 侧 `nvidia-smi` 正常
2. 在 WSL 内执行：

```bash
export PATH=/usr/lib/wsl/lib:$PATH
ls -l /dev/dxg
nvidia-smi
```

3. 如果仍失败，优先检查：
   - Windows NVIDIA 驱动是否过旧
   - WSL 是否需要 `wsl --update`
   - 是否误在 WSL 内安装了 Linux NVIDIA 显卡驱动
   - 当前命令是否运行在受限沙箱或代理会话里

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
