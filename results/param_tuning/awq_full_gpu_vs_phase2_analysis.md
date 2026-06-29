# AWQ Full-GPU Sweep Analysis

## Final Call Recommendation After 7B Normal Compare

补充时间：2026-06-29

本结论现在同时参考五组结果：

- normal 7B baseline：`phase2-7b-baseline-resampled-20260628T162953Z`
- AWQ baseline：`phase2-awq-baseline-resampled-20260628T221900Z`
- AWQ full-GPU warmup sweep：`phase3-formal-awq-gpu-warmup-20260628T1556Z`
- AWQ 长上下文补实验：`phase3-long-context-awq-20260629T065835Z`
- 显存分解 baseline：normal `phase2-7b-baseline-detailed-20260629T052330Z`，AWQ `phase2-awq-baseline-detailed-20260629T143227Z`

最终建议：**后续本机推理服务默认使用 AWQ 模型，并保留 `max_model_len=3072`、`max_num_batched_tokens=4096`、`max_num_seqs=32` 作为当前默认调用配置。**

理由：

- 与 normal 7B 相比，AWQ 在 `24/24` 个重叠 Phase 2 workload 上 QPS 更高，P95 latency 更低。平均 QPS 从 `2.90 req/s` 提升到 `6.61 req/s`，平均 P95 E2EL 从 `3344.13 ms` 降到 `1443.35 ms`。
- AWQ 的核心收益来自 decode 成本下降：平均 P95 TPOT 从 normal 7B 的 `10.35 ms` 降到 AWQ 的 `4.20 ms`，因此长输出场景收益稳定。
- normal 7B 在高并发长输入短输出 case 上观测到 `num_requests_waiting_during_run_max=12`，AWQ baseline 没有观测到 vLLM waiting queue；这说明 AWQ 降低了服务侧堆积风险。
- AWQ tuning 的九个单变量档位之间差异较小，`3072/4096/32` 不是单指标极值的过拟合点，而是吞吐、P95、TTFT 和配置安全边界之间最稳的折中。

需要保留的限制：

- 当前 `gpu_memory_used_mb_after` 口径下，AWQ baseline 进程级显存约 `27.8 GB`，normal 7B 约 `25.8 GB`，因此不能写成“AWQ 在本实验中显著降低 nvidia-smi 显存”。
- 这个显存字段包含 vLLM runtime、allocator、CUDA graph、KV cache 预留等，不等价于纯模型权重大小。AWQ 的推荐依据是吞吐和延迟优势，而不是当前这列显存指标。
- 如果未来目标转为“最小显存部署”，需要单独设计显存分解实验或 sweep `gpu_memory_utilization` / KV cache 预算，而不是直接引用当前 model compare 的 `gpu_memory_used_mb_after`。

## Closeout Additions

### Long-Context Supplement

补实验范围为：

- `concurrency=8/16`
- `input_tokens=2048`
- `output_tokens=128/512`
- `max_model_len=3072/4096`
- `max_num_batched_tokens=4096/8192`

主要结论：

- 在 `c8 / in2048 / out128` 下，`3072/4096` 仍是最稳妥的点：`10.98 QPS`、`759.05 ms` P95、`135.62 ms` TTFT。把 `max_num_batched_tokens` 提到 `8192` 只会让 QPS、P95 和 TTFT 同时变差。
- 在 `c8 / in2048 / out512` 下，`max_model_len=4096, max_num_batched_tokens=4096` 吞吐最高，为 `3.13 QPS`；`3072/4096` 的 P95 只低约 `3 ms`，两者差距很小，但 `8192` 依旧没有带来正收益。
- 在 `c16 / in2048 / out128` 下，`max_model_len=4096` 开始显著优于 `3072`。最高吞吐是 `4096/4096` 的 `16.78 QPS`，最低 P95 是 `4096/8192` 的 `888.16 ms`，但 `8192` 只换来很小的 P95 优势，同时继续压缩 KV cache 预算，因此默认值不改成 `8192`。
- 在 `c16 / in2048 / out512` 这个最接近 Phase 2 压力点的 workload 下，最优点稳定落在 `4096/4096`：`4.67 QPS`、`3037.73 ms` P95、`215.29 ms` TTFT。`3072` 会同时拖慢吞吐和 TTFT，`8192` 也没有把吞吐再推高。

更新后的解释是：

- 默认配置仍保留 `3072/4096/32`，因为它在常规 `512/128` workload 和 `c8` 长输入下都足够稳，而且不会白白挤掉 KV cache。
- 但如果目标 workload 长期贴近 `c16 / in2048 / out512`，应把 `max_model_len` 提到 `4096`，而不是优先把 `max_num_batched_tokens` 拉到 `8192`。

### Memory Decomposition

详细 baseline 现在已经把“AWQ 是否省显存”这个问题拆开了：

- normal 7B 启动后健康态显存约 `28475 MiB`，AWQ 约 `29003 MiB`
- normal 7B 的 model weights memory 为 `14.19 GiB`，AWQ 只有 `5.2 GiB`
- normal 7B 的 available KV cache memory 为 `12.7 GiB`，AWQ 扩到 `22.26 GiB`
- normal 7B 的 GPU KV cache size 为 `237,728` tokens，AWQ 扩到 `416,864` tokens
- CUDA graph pool 两边都很小，normal 约 `0.09 GiB`，AWQ 约 `0.06 GiB`

因此可以明确写成：

- AWQ 确实大幅降低了权重显存
- 但 vLLM 会把这部分释放出的空间继续分配给 KV cache 和 runtime 预留
- 所以最终 `nvidia-smi used memory` 不会按权重压缩比例同步下降

这个结论已经足够支撑文档中关于 AWQ 显存口径的收尾说明。

调用建议：

```bash
MODEL_DIR=/root/autodl-tmp/qwen2.5-7b-awq \
SERVED_MODEL_NAME=qwen-7b-awq-local \
MAX_MODEL_LEN=3072 \
GPU_MEMORY_UTILIZATION=0.9 \
VLLM_MAX_NUM_BATCHED_TOKENS=4096 \
VLLM_MAX_NUM_SEQS=32 \
VLLM_DISABLE_LOG_STATS=0 \
./scripts/run_vllm_local.sh
```

如果要和 Phase 2 baseline 保持完全一致的资源预算，则把 `GPU_MEMORY_UTILIZATION=0.9` 改回 `0.8`；如果要复现 Phase 3 full-GPU tuning 结论，则使用 `0.9`。两者不要混在同一张对比表里直接解释显存。

## Scope

- Sweep run: `phase3-formal-awq-gpu-warmup-20260628T1556Z`
- Model: `qwen2.5-7b-awq` on full GPU, `gpu_memory_utilization=0.9`
- Fixed workload: `concurrency=8`, `input_tokens=512`, `output_tokens=128`, `warmup_repeats=1`, `measured_repeat=2`, `num_prompts=40`
- Phase2 reference: `results/batches/phase2-awq-baseline-20260628T124711Z`, matching case `c8 / in512 / out128`

## Method Change

- This run uses the updated phase3 methodology: each parameter setting first executes one explicit warmup pass, then only the two measured repeats are aggregated into `param_tuning.csv`.
- Compared with the earlier no-warmup run, the main purpose of this adjustment is to remove first-live-batch JIT / graph-capture noise from the tuning conclusion itself rather than compensating for it later in analysis.
- Measured-repeat stability is now tight: average absolute spread across all nine settings is `0.094` QPS, `5.35` ms P95, `4.81` ms TTFT.

## Phase2 Reference

- Phase2 AWQ reference mean: `13.62` QPS, `593.29` ms P95, `68.99` ms TTFT, `27363` MB GPU memory.

## Main Findings

- Best throughput point is `max_model_len=3072` at `13.74` QPS, `602.31` ms P95, `63.98` ms TTFT, `29487` MB. Relative to phase2 AWQ, that is `+0.84%` QPS, `+1.52%` P95, `-7.26%` TTFT, `+7.76%` memory.
- Lowest P95 point is `max_num_batched_tokens=2048` at `596.98` ms, but its gain over the throughput winner is small. This means the ranking difference is now a real trade-off at the margin, not a warmup artifact.
- Lowest TTFT point is `max_num_batched_tokens=4096` at `59.94` ms, which reinforces that `max_num_batched_tokens=4096` is the most balanced default for this workload.

## Parameter Analysis

### `max_model_len`

| Value | QPS | P95 (ms) | TTFT (ms) | TPOT (ms) | GPU MB | dQPS vs Phase2 | dP95 vs Phase2 | dTTFT vs Phase2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2048` | 13.53 | 602.56 | 70.35 | 4.053 | 29487 | -0.68% | +1.56% | +1.97% |
| `3072` | 13.74 | 602.31 | 63.98 | 4.043 | 29487 | +0.84% | +1.52% | -7.26% |
| `4096` | 13.58 | 597.05 | 67.91 | 4.050 | 29487 | -0.28% | +0.63% | -1.57% |

- `max_model_len` remains a weak first-order knob under this workload because live context is only about `512 + 128 = 640` tokens, well below every tested ceiling.
- The three settings are now tightly clustered. `3072` wins on QPS, `4096` wins on P95, and the difference is below 1% on both metrics. This is exactly the kind of result that should be interpreted as “choose for safety margin”, not “chase a false optimum”.
- Since memory is flat at about `29.5 GB`, there is no meaningful capacity penalty among these three values under the current workload.

### `max_num_batched_tokens`

| Value | QPS | P95 (ms) | TTFT (ms) | TPOT (ms) | GPU MB | dQPS vs Phase2 | dP95 vs Phase2 | dTTFT vs Phase2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2048` | 13.56 | 596.98 | 69.21 | 4.051 | 29487 | -0.48% | +0.62% | +0.31% |
| `4096` | 13.65 | 600.66 | 59.94 | 4.068 | 29487 | +0.19% | +1.24% | -13.12% |
| `8192` | 13.32 | 610.91 | 64.93 | 4.140 | 28986 | -2.18% | +2.97% | -5.89% |

- `4096` gives the best throughput and the best TTFT, while `2048` only edges it on P95 by less than 1%. That is a mild latency/throughput trade-off, not a decisive reversal.
- `8192` is the clearest non-winner in this group: throughput drops, P95 worsens, and only about `1.7%` memory is saved. This suggests the larger token budget is not translating into better packing efficiency for the current queue shape.
- There is no waiting observed in this warmed run, so the knob is now acting more through packing efficiency and prompt admission smoothness than through obvious queue buildup.

### `max_num_seqs`

| Value | QPS | P95 (ms) | TTFT (ms) | TPOT (ms) | GPU MB | dQPS vs Phase2 | dP95 vs Phase2 | dTTFT vs Phase2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `16` | 13.17 | 613.55 | 73.73 | 4.146 | 29488 | -3.34% | +3.41% | +6.87% |
| `32` | 13.54 | 598.78 | 68.58 | 4.063 | 29487 | -0.60% | +0.92% | -0.61% |
| `64` | 13.39 | 607.82 | 60.48 | 4.149 | 29527 | -1.67% | +2.45% | -12.34% |

- `32` is still the best overall point. `16` is clearly too restrictive for throughput, while `64` fails to convert extra scheduler headroom into better latency or throughput.
- The fact that `64` slightly lowers TTFT but still loses on QPS and P95 suggests that wider seq headroom can help admission on individual requests without improving end-to-end packing quality for the batch as a whole.
- This is a useful sign that `max_num_seqs` should be matched to realistic in-flight concurrency rather than pushed upward by default.

## Attribution

- The dominant separation still comes from prefill-side behavior rather than decode-side behavior. Across all nine settings, `mean_tpot_ms` stays in a narrow band around `4.04` to `4.15` ms, while TTFT and P95 move more noticeably.
- That means these parameters are mainly influencing request admission, prefill batching, and scheduler shape. They are not materially changing decode token cost once generation is already underway.
- The warmup change worked as intended: measured-repeat variance is now small enough that parameter deltas can be interpreted directly, instead of needing to separate cold and warm repeats post hoc.
- Relative to phase2 AWQ, the new sweep shows that most settings are within roughly `±1%` QPS and `+0.6%` to `+3.4%` P95. The remaining systematic gap is much smaller than before and is consistent with run-shape differences rather than a parameter regression.

## Recommendation

- Keep `max_model_len=3072`, `max_num_batched_tokens=4096`, `max_num_seqs=32` as the default phase3 recommendation. It is either the best or effectively tied for best on every primary metric, and it avoids overfitting to a tiny single-metric edge.
- If a future workload shifts toward longer prompts, revisit `max_model_len` first. The current conclusion is intentionally conditional on `512/128`, where context ceiling is mostly inactive.
- If memory becomes the top constraint, `max_num_batched_tokens=8192` is not the answer here; it saves too little memory for the throughput and latency penalty. The more realistic memory trade-off should be explored elsewhere, for example with model-side or GPU-utilization-side constraints.
- The methodology change should remain in place for later sweeps. Warmup-before-measurement is the correct default for this repo because it makes the tuning result itself match the steady-state interpretation the report is trying to defend.

