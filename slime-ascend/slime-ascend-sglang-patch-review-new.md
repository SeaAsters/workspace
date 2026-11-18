# slime-ascend SGLang Patch 下沉排查

> **排查日期**：2026-11-18
> **对比对象**：slime-ascend main / SGLang main
> **文档性质**：本地源码静态盘点，设备验证及现场兼容性另行核查。

背景：slime-ascend 升级时，推理补丁与 SGLang 主线、现场 RL 调用协议之间存在兼容问题。本次从现行安装补丁栈出发，梳理可下沉到推理框架的改动及应留在 RL 侧的适配。

排查基线：

- slime-ascend main：`e58ab12fd6d55952d03f3ef97367242f1435f321`。
- SGLang main：`8f2e5fe54efabf30a87138c6e6477676412686d4`。
- slime-ascend 安装基线：SGLang `v0.5.21`，提交 `e00930c5489053f26d86b179cee0d087f846acbb`，再应用 6 个 SGLang patch。
- 另有 3 个 sgl-kernel-npu patch，需要在依赖仓推进。

以下状态沿用本地 main 的静态排查结果。“未包含”表示尚有代码差异，不表示已经在设备上复现故障；“已有等价能力”也不表示旧 RL 调用协议可以直接兼容。未核实远端 PR 和 release 分支状态，不登记未经确认的 PR 编号。

## 目录

- [1. 待解决](#1-待解决)
- [2. Patch 总表](#2-patch-总表)
- [3. 主集成补丁拆项](#3-主集成补丁拆项)
- [4. 权重更新接口](#4-权重更新接口)
- [5. 采样支持集合与行为概率](#5-采样支持集合与行为概率)
- [6. HiCache 主机内存释放与恢复](#6-hicache-主机内存释放与恢复)
- [7. 完整及增量权重拉取](#7-完整及增量权重拉取)
- [8. 确定性推理补丁拆项](#8-确定性推理补丁拆项)
- [9. sgl-kernel-npu 的 TMS 补丁](#9-sgl-kernel-npu-的-tms-补丁)
- [10. 不下沉到 SGLang 的范围](#10-不下沉到-sglang-的范围)
- [11. 专项推进建议](#11-专项推进建议)
- [12. 排查范围说明](#12-排查范围说明)

---

## 1. 待解决

### 推理侧：建议下沉到 SGLang

- `0001-v0.5.21-tms-forward-graph-args.patch`：TMS adapter 位置参数透传，main 尚未包含，建议独立下沉。
- `sglang-release_hicache.patch` 中 release/resume tag 幂等：main 尚未包含，建议与 HiCache 能力拆开提交。
- `sglang-release_hicache.patch` 中 host KV 内存释放/恢复：main 尚未包含，建议下沉；需要明确 storage backend 和共享 backing 的支持边界。
- `sglang.patch` 中 DSA 压缩 tail 的 memory-saver region：main 对应分配点仍缺失，建议独立下沉。
- `sglang.patch` 中 PD bootstrap/transfer 超时及取消：main 存在其他超时/回收机制，但未覆盖此补丁的全部目的，应基于当前状态机重写。
- `sglang.patch` 中 graph capture token 数、draft offloader、DSA head shape、GLM4V 和量化恢复 hook：仍有差异，建议拆成独立问题推进。

### 已有基础能力：优先迁移接口

- `sglang.patch` 中权重更新兼容：main 已有 begin/end 会话、runner selector 和量化后处理。RL 应迁移现行协议；不建议直接下沉“删除会话检查”的写法。
- `sglang-top_p.patch`：main 已有 sampling-mask、支持集合及行为 logprob 能力。普通采样优先复用；EAGLE/MTP/NGRAM 支持边界需另行核对和补齐。
- `sglang.patch` 中 retracted 请求存在时仍轮询 transfer：main 已有等价处理，不再重复下沉。

### 依赖侧：下沉到 sgl-kernel-npu

- `0001-fix-torch-memory-saver-allocation-lifecycle.patch`：分配 pause/resume/free 生命周期。
- `0002-fix-tms-sglang-cuda-graph-api-and-preload-mode.patch`：NPU graph 参数和子进程 preload 配置。
- `0003-fix-tms-unaligned-allocations-and-nested-regions.patch`：未对齐分配回退与嵌套 region 恢复。

这三项归属明确，但本轮未核查 sgl-kernel-npu 最新 main，暂不能标记为“上游未修复”。

### 其他：训练侧及待核对

- Ray NPU 资源、Slime accelerator、训练 CP/packing、PPO/GRPO loss 和训练侧 replay：不下沉到 SGLang。
- Megatron/Adaptor/Bridge/mbridge 的转换与训练适配：由相应仓库维护。
- `sglang-pull_weights.patch`：可下沉通用 checkpoint 同步组件，发布目录和增量编码协议需与 RL/publisher 拆分。
- `sglang-deterministic.patch`：包含 CUDA kernel、DeepGEMM/DeepEP 依赖和通用诊断，不能整体认定为 Ascend 推理修复。
- 删除 RadixCache 引用断言、放宽 Mooncake failed-session 标记、EAGLE 输入 clamp：需根因分析，不直接下沉现有写法。

## 2. Patch 总表

六个 SGLang patch 合计涉及 82 个不同文件路径、265 个 diff hunk；补丁文件之间有重叠，不能按文件名作为单一需求整体提交。

| patch                                         | 作用                                                          | SGLang main                                             | 下沉结论 / 备注                                      |
| --------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------- |
| `0001-v0.5.21-tms-forward-graph-args.patch` | TMS adapter 透传图捕获位置参数                                | 未包含，仍仅接收 `**kwargs`                           | 建议下沉，独立小修复                                 |
| `sglang-release_hicache.patch`              | tag 幂等、HiCache host KV 内存释放与恢复                      | 未包含对应完整实现                                      | 建议下沉，拆分为“幂等修复”和“host KV 生命周期”   |
| `sglang-top_p.patch`                        | 返回采样支持集合与截断后 logprob，贯通投机/PD/流式输出        | 已有 sampling-mask 基础能力，旧字段和投机支持范围不等价 | 优先迁移现有接口，剩余能力单独补齐                   |
| `sglang-pull_weights.patch`                 | 从发布目录拉取完整/增量权重，构建本地 checkpoint              | 未见 `/pull_weights` 和对应 local_checkpoint 实现     | 通用组件可下沉；发布协议需拆分；该接口不加载在线模型 |
| `sglang-deterministic.patch`                | router、FP8、DSA、MoE/DeepEP 数值对齐，附带 JIT/dump/IPC 改动 | 已有确定性基础设施，仍有多项差异                        | 按能力拆分，CUDA 与 NPU 分别验证                     |
| `sglang.patch`                              | 权重接口、PD、DSA、模型、缓存及观测混合包                     | 部分已有等价能力，部分缺失，部分代码已重构              | 必须拆项，详见下表                                   |

## 3. 主集成补丁拆项

**来源补丁**：`sglang.patch`。按职责分组列出，便于分别推进。

### 权重更新与兼容接口

| 改动                                         | 作用                                | main 现状                               | 结论                                             |
| -------------------------------------------- | ----------------------------------- | --------------------------------------- | ------------------------------------------------ |
| 权重更新 session 检查放宽                    | 兼容未调用 begin/end 的旧 RL 客户端 | tensor/distributed 更新仍要求会话       | 不直接下沉；迁移 RL 或提供显式 legacy adapter    |
| `/post_process_weights`                    | 独立触发量化恢复和加载后处理        | 已通过 begin/end 和加载流程提供相关能力 | 收敛现有生命周期，避免两套主协议                 |
| `disable_draft_model`                      | 权重更新仅作用 target               | 已有 `selector=target/draft/all`      | 旧字段在兼容层映射，不作为第二套控制参数长期维护 |
| `/get_weight_version`、`/weight_version` | 便捷查询权重版本                    | `/model_info` 已有版本来源            | 可作为兼容别名，非核心能力缺口                   |

### PD 调度与 KV 内存管理

| 改动                                   | 作用                                | main 现状                                                    | 结论                                   |
| -------------------------------------- | ----------------------------------- | ------------------------------------------------------------ | -------------------------------------- |
| PD bootstrap/transfer timeout          | 防止建链和传输阶段无限等待          | 相应分支未包含补丁的完整期限处理；另有 deferred-release 超时 | 建议下沉，区分请求期限与安全回收期限   |
| PD release/resume queue 与 buffer 处理 | 回收前取消传输，注销/恢复传输缓冲区 | 已有注册/注销及更复杂的 retirement/hold/drain 流程           | 核对剩余缺口，不照搬 `queue.clear()` |
| retracted 时继续轮询 transfer          | 避免已占用页面的传输无法完成        | 已有等价逻辑                                                 | 不重复下沉                             |
| decode 模式允许回退最后一个请求        | 缓解低内存下无法前进的问题          | `retract_decode` 仍至少保留一个请求                        | 候选修复，需重建触发用例               |
| DSA 压缩 tail memory region            | 使 tail KV 分配可随 KV tag 回收     | 对应 tail 分配点仍只有 custom mem pool 上下文                | 建议下沉                               |

### 图捕获与 target/draft 状态

| 改动                        | 作用                                                   | main 现状                  | 结论                           |
| --------------------------- | ------------------------------------------------------ | -------------------------- | ------------------------------ |
| graph capture 的 token 容量 | capturer 容量从 `bs` 改为 `bs * num_tokens_per_bs` | 仍传 `bs`                | 建议核对 token 布局后下沉      |
| draft 不重置全局 offloader  | 避免 draft 初始化覆盖 target 的 offloader              | 仍无条件设置全局 offloader | 建议下沉多 runner 状态隔离修复 |

### DSA、量化与多模态模型

| 改动                                | 作用                                          | main 现状                   | 结论                                                |
| ----------------------------------- | --------------------------------------------- | --------------------------- | --------------------------------------------------- |
| DSA query heads 重复及 gate 对齐    | 适配小 head 数，统一 q_scale/gate/graph shape | 未见对应 helper             | 可下沉，先明确 32-head 约束来源和后端适用范围       |
| `INDEXER_ROPE_NEOX_STYLE`         | 允许覆写 indexer RoPE 风格                    | 未见该配置                  | 优先用模型配置/正式参数表达，不直接推广临时环境变量 |
| compressed-tensors restore hook     | Linear 转发恢复 hook，MoE 对可选 hook 判存在  | 仍有对应差异                | 建议下沉到现行权重更新流程                          |
| GLM4V encoder/language 模式         | 分段构建与权重加载                            | 未见对应完整处理            | 可下沉，需补足 language-only 资源隔离验证           |
| GLM4V `rope_parameters` 回退      | 兼容不同 Transformers 配置格式                | 仍直接读取 `rope_scaling` | 建议作为模型配置兼容修复下沉                        |
| GLM4V `get_mm_data`               | 接收预计算视觉 embedding 并生成 mRoPE         | 未见对应入口                | 可下沉为模型通用能力                                |
| QwenVL 改用 `legacy_load_mm_data` | 回退旧多模态加载路径                          | 仍使用新的 `load_mm_data` | 不直接回退；先用失败样例确认兼容问题                |

### 观测与通用工具

| 改动                        | 作用                                    | main 现状                          | 结论                                     |
| --------------------------- | --------------------------------------- | ---------------------------------- | ---------------------------------------- |
| `/server_info` timeout    | 限制内部状态查询等待时间                | 仍直接等待内部状态                 | 可下沉，但应显式标识超时/部分结果        |
| `/v1/loads` inflight 明细 | 返回每请求阶段与等待时长                | 主要读取共享负载快照，无该明细实现 | 可做独立诊断能力，不改变默认轻量查询路径 |
| PD 时间指标与 TTFT/ITL      | 补充重试、阶段耗时，修正 token 计数条件 | 未包含补丁对应完整改动             | 可独立下沉，核对跨线程/进程时钟语义      |
| profiler 完成条件           | 从步数字典相等改为达到目标即完成        | 未包含该改动                       | 候选通用修复，补触发用例                 |
| FastAPI route path 回退     | 支持 path_format/child route 路径解析   | 未包含该改动                       | 可独立下沉                               |

### 需核对或不建议直接下沉的改动

| 改动                                 | 作用                                | main 现状                               | 结论                                         |
| ------------------------------------ | ----------------------------------- | --------------------------------------- | -------------------------------------------- |
| Mooncake auxiliary `dst_addr`      | 使用循环变量替代同索引表达式        | 表达方式不同                            | 等价简化，不作为重点故障修复                 |
| Mooncake failed-session 策略         | 单次失败后不立即将 session 标为失败 | 未采用该策略                            | 待核对错误分类与重试规则，不直接删除失败标记 |
| RadixCache 未完成请求切片            | 按已提交 KV 长度限制缓存插入        | 已重构为 `checkpoint(req, up_to=...)` | 原补丁不直接适用，需验证重构后是否仍有缺口   |
| RadixCache `dec_lock_ref` 断言删除 | 遇到非当前树引用时提前退出          | 仍保留跨 tree 引用断言                  | 不建议下沉；应修复引用生命周期               |
| EAGLE graph 输入 clamp               | 限制概率与 token 索引范围           | 未包含该写法                            | 待核对，不能用 clamp 掩盖非法输入            |

## 4. 权重更新接口

### main 的现状

SGLang 已有：

- `begin_weight_update()` / `end_weight_update()`。
- `selector=target|draft|all`。
- 量化权重恢复、加载后处理和平台权重更新 handler。
- 无会话时拒绝 tensor/distributed 权重更新。

slime patch 则允许旧客户端在没有会话时直接传 bucket，再自行调用 `/post_process_weights`。两者是调用协议差异，不是 main 尚未提供在线权重更新。

| 维度              | slime patch                    | SGLang main            | 处理建议                              |
| ----------------- | ------------------------------ | ---------------------- | ------------------------------------- |
| 更新边界          | 允许不显式开启 session         | 要求 begin/end         | RL 迁移会话协议                       |
| 后处理            | 独立 `/post_process_weights` | 会话与加载生命周期管理 | 复用 main，保留必要的旧版适配         |
| target/draft 选择 | 增加 `disable_draft_model`   | 使用 selector          | 在兼容层做参数映射                    |
| 量化 scheme hook  | 补充/保护 restore hook         | 部分 scheme 仍有差异   | 将 hook 修复下沉，不撤掉 session 防护 |

结论：**权重生命周期能力应由 SGLang 提供，现场旧协议兼容应明确边界；不把“取消会话检查”作为默认下沉方案。**

依据：[main 权重会话](D:/WorkSpace/sglang/python/sglang/srt/managers/scheduler_components/weight_updater.py)（第 358 行）、[源 patch](D:/WorkSpace/slime-ascend/patches/third_party/sglang/sglang.patch)（第 894 行）。

## 5. 采样支持集合与行为概率

**来源补丁**：`sglang-top_p.patch`。

### main 的现状与补丁差异

| 维度                           | slime patch                                                                | SGLang main                                          | 结论                                    |
| ------------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------- | --------------------------------------- |
| 请求开关                       | custom_params 中的 `return_top_p_token_ids` / `return_top_p_log_probs` | `return_sampling_mask`、`sampling_logprobs_mode` | 优先迁移正式接口                        |
| 支持集合                       | 用 PyTorch 重建 top-k/top-p/min-p mask                                     | 可捕获采样后的过滤权重                               | 复用 main，减少重复实现与边界差异       |
| 行为 logprob                   | 部分路径改写 next_token_logprobs                                           | 单独提供 selected/support sampling logprobs          | RL 读取正确字段，区分模型概率与行为概率 |
| Ascend 普通采样                | 收集候选集合；请求完整 top-p score 的组合会抛错                            | 有 Ascend filtered-probs capture 链路                | 静态能力已有，仍需设备回归              |
| EAGLE/MTP/NGRAM                | 在 speculative logprob 链路增加支持集合/归一化处理                         | sampling-mask 校验目前仅允许 DFlash family           | 尚不等价，需独立补齐算法支持            |
| 接受 token 不在重建 nucleus 内 | force-keep，再用并集重新归一化                                             | 对支持的 speculative 路径约束真实采样分布            | 不直接沿用 force-keep 作为正确性证明    |
| PD/流式输出                    | 增加 top-p 专用元数据链路                                                  | 已有 sampling-mask 对应链路                          | 迁移现行协议并核对输出对齐              |

补丁中“为了避免 -inf，将 accepted token 强行加入支持集合”的处理，只能保证数值有限，不能证明返回概率等于实际行为分布。训练侧 score centering、loss 和 replay 策略继续留在 RL；推理框架负责输出真实采样支持集合与相应概率。

结论：**普通采样优先复用 main；EAGLE/MTP 等投机场景作为剩余能力立项，不再整体维护第二套 top-p 输出链路。**

依据：[main Ascend sampler](D:/WorkSpace/sglang/python/sglang/srt/layers/sampler.py)（第 698 行）、[sampling-mask 归一化](D:/WorkSpace/sglang/python/sglang/srt/layers/sampler.py)（第 545 行）、[投机支持限制](D:/WorkSpace/sglang/python/sglang/srt/speculative/spec_sampling_mask.py)（第 22 行）、[源 patch](D:/WorkSpace/slime-ascend/patches/third_party/sglang/sglang-top_p.patch)。

## 6. HiCache 主机内存释放与恢复

**来源补丁**：`sglang-release_hicache.patch`。

该补丁包含两类问题，应分开推进。

| 改动                            | main 的现状                                                    | 结论                                          |
| ------------------------------- | -------------------------------------------------------------- | --------------------------------------------- |
| tag 去重及 release/resume 幂等  | resume 仍直接 `offload_tags.remove(tag)`，未包含对应幂等过滤 | 建议先下沉小修复                              |
| host pool 解除 tensor/view 引用 | 未见完整释放/重建实现                                          | 建议下沉                                      |
| unified/shared backing 生命周期 | 未见对应 host release/resume 实现                              | 与 host pool 一起下沉，验证最后一个使用方释放 |
| pin/register/unregister 恢复    | 需要与主机内存重建同步管理                                     | 必须作为完整生命周期验证                      |
| storage backend                 | 源 patch 在 storage 启用时跳过 host release                    | 支持边界尚不完整，不能宣称全场景支持          |

此处释放的是 host KV 缓存占用，不能解释为“保留缓存内容、恢复后继续直接使用”。验收需要同时检查主机内存实际回收、共享引用消失、注册状态和恢复后的真实生成。

依据：[main resume](D:/WorkSpace/sglang/python/sglang/srt/managers/scheduler_components/weight_updater.py)（第 537 行）、[源 patch](D:/WorkSpace/slime-ascend/patches/third_party/sglang/sglang-release_hicache.patch)。

## 7. 完整及增量权重拉取

**来源补丁**：`sglang-pull_weights.patch`。

| 改动                                        | 作用                                | main                                 | 结论                                   |
| ------------------------------------------- | ----------------------------------- | ------------------------------------ | -------------------------------------- |
| `/pull_weights`                           | 触发各 host 拉取权重                | 未见该接口                           | 可设计通用入口，先与现有更新接口收敛   |
| local checkpoint materialization            | 用完整权重或基线构建本地 checkpoint | 未见对应组件                         | 通用同步能力可下沉                     |
| delta chain / checksum / host lock          | 应用增量、校验并避免同机重复拉取    | 未见对应实现                         | 可复用，需补中断恢复与原子性设计       |
| `weight_vNNNNNN`、增量编码、pre-read hook | 对接发布者目录与协议                | 非现有统一契约                       | 可留在 RL/publisher 插件               |
| serving 模型加载                            | 将 checkpoint 切换到在线模型        | 该 patch 的 pull endpoint 本身不执行 | 仍需衔接模型加载、cache 清理及版本确认 |

结论：**“checkpoint 拉取成功”与“在线模型权重切换成功”必须分开。** 不能把该接口直接描述为完整在线权重更新方案。

依据：[源 patch](D:/WorkSpace/slime-ascend/patches/third_party/sglang/sglang-pull_weights.patch)（第 122 行）。

## 8. 确定性推理补丁拆项

**来源补丁**：`sglang-deterministic.patch`。

| 改动                                        | 目的                                          | main / 依赖情况                                                      | 结论                                               |
| ------------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------- |
| FP32 MoE router / batch-invariant GEMM      | 稳定 router logits 与归约行为                 | 已有 batch-invariant 基础设施，仍有补丁差异                          | 可下沉通用数值契约，按后端实现                     |
| GLM5 router GEMM                            | 特定形状的快速确定性计算                      | patch 引用的 `glm5_router_gemm.cuh` 未随此 patch 提供，main 也未见 | 先补齐 kernel 来源与依赖，不能仅搬 wrapper         |
| per-token FP8 group quant                   | 对齐归约、舍入和 scale 行为                   | 仍有差异，补丁为 CUDA 实现                                           | 可下沉 kernel 改进；不作为 NPU 已支持证据          |
| DSA head-gate / top-k / FP8 KV round-trip   | 对齐 prefill/decode 的数值与索引顺序          | 仍有差异                                                             | 可下沉，需 reference 和图路径验证                  |
| DeepGEMM batch-invariant / expert M padding | 固定分组 GEMM 的执行条件                      | 依赖 DeepGEMM API 和实现                                             | SGLang 配置与依赖仓能力协同推进                    |
| DeepEP LL prefill staging                   | 限制单阶段 token 容量，保持各 rank 相同阶段数 | 未见对应 helper                                                      | 可独立评审，不与所有确定性改动捆绑                 |
| DeepEP 量化 alignment / fabric API 探测     | 适配不同依赖 API                              | 可能依赖特定 fork 参数                                               | 通用探测可下沉，依赖新增参数在依赖仓落地           |
| JIT extra kernel roots                      | 加载外部 kernel 源码                          | 未见对应扩展                                                         | 可设计通用扩展机制；核心 kernel 应随框架交付       |
| tensor dump 筛选与分块                      | 降低对齐诊断成本                              | 未包含对应完整改动                                                   | 可下沉，作为低优先级调试功能                       |
| flattened bucket 返回前同步                 | 防止 IPC 生产者提前复用缓冲区                 | 对应加载方法仍直接返回                                               | 候选生命周期修复，核对 stream/持有方后确定同步范围 |

结论：**拆成 kernel 数值、通信依赖、诊断工具和缓冲区生命周期四类推进。** CUDA fast path 与 NPU 的支持结论分别给出；本轮没有设备验证结果。

依据：[源 patch](D:/WorkSpace/slime-ascend/patches/third_party/sglang/sglang-deterministic.patch)。

## 9. sgl-kernel-npu 的 TMS 补丁

| patch                                                           | 作用                                                                     | 最新 main 状态 | 下沉归属                               |
| --------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------- | -------------------------------------- |
| `0001-fix-torch-memory-saver-allocation-lifecycle.patch`      | 记录 allocation paused 状态，避免重复 unmap/free physical，检查 map 结果 | 本轮未核查     | sgl-kernel-npu / TMS allocator         |
| `0002-fix-tms-sglang-cuda-graph-api-and-preload-mode.patch`   | 透传 `auto_dispatch_capture`，子进程继承 preload 配置                  | 本轮未核查     | sgl-kernel-npu / TMS runtime           |
| `0003-fix-tms-unaligned-allocations-and-nested-regions.patch` | 未对齐 ACL 分配回退普通 allocator，嵌套退出恢复 region 状态              | 本轮未核查     | sgl-kernel-npu / TMS allocator/runtime |

这三项与 SGLang adapter、DSA memory region 配套验证。未对齐 fallback 不由 TMS VMM 管理，还需检查其对实际可回收内存容量的影响。

## 10. 不下沉到 SGLang 的范围

| 改动                                             | 归属                                | 原因                           |
| ------------------------------------------------ | ----------------------------------- | ------------------------------ |
| Ray NPU 资源、actor placement、Slime accelerator | Slime / RL runtime                  | RL 进程和训练资源编排          |
| 训练 CP、packing、训练 logprob/entropy           | Slime / Megatron                    | 训练计算与数据组织             |
| PPO/GRPO loss、score centering、训练 replay      | RL                                  | 训练算法与策略                 |
| checkpoint 转换、权重命名桥接、Megatron root     | Bridge / mbridge / RL 集成层        | 跨框架转换与环境适配           |
| 训练侧 empty-cache teardown 错误窄化吞掉         | Slime accelerator；底层根因另行定位 | 不能将异常屏蔽搬入推理常规路径 |
| 未登记的历史 adapter/CP patch                    | 待历史或现场版本差分                | 不属于本轮现行默认安装栈       |

## 11. 专项推进建议

1. **先提交小修复**：TMS adapter 参数透传、tag 幂等、DSA 压缩 tail 内存管理；TMS 底层补丁同步在依赖仓核对。
2. **同步处理升级接口**：明确权重 begin/end、runner selector、sampling-mask 字段及投机支持范围，形成旧 RL 到新 SGLang 的迁移说明。
3. **再推进稳定性和模型支持**：HiCache host 内存、PD 超时与取消、graph capture、offloader、量化 hook、DSA/GLM4V。
4. **独立设计大项**：deterministic 的各后端数值契约、EAGLE/MTP 采样语义、增量 checkpoint 发布和恢复。

每项以“上游能力落地 + 对应场景验证 + RL 调用迁移 + 删除旧 patch”为完成条件。现场版本尚未提供，因此本轮不对具体现场兼容性作结论。

## 12. 排查范围说明

本版按参考稿的结构重写，沿用 2026-11-18 首轮扫描证据。Slime submodule 未初始化，未执行 patch 应用、编译、NPU/GPU 或端到端 RL 测试；sgl-kernel-npu 最新主线和远端 PR 状态待后续核查。
