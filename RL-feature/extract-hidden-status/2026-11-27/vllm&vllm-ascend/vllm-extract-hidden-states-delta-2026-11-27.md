# vLLM / vllm-ascend hidden state extract / return / aux relay 增量跟踪：2026-11-27

UTC 窗口：`[2026-11-20T06:54:42Z, 2026-11-27T01:49:24Z)`。采集截止：`2026-11-27T01:49:24Z`。覆盖 **vLLM 主仓**与 **vllm-ascend**；SGLang 侧见 [../sglang/sglang-extract-hidden-states-delta-2026-11-27.md](<../sglang/sglang-extract-hidden-states-delta-2026-11-27.md>)。

代码基准：vLLM `29fa74e654c6`（`2026-11-26T19:20-06:00`）；vllm-ascend `d1108b2f8288`（`2026-11-27T01:34Z`）。

本期按用户要求：**只深读 1 条重点 PR，其余一句话带过。**

## 深读：#69875 —— PP 下游阶段的 aux 接收缓冲被按「模型 dtype」分配

[[Bugfix][Spec Decode][PP] Allocate aux hidden-state receive buffers in residual dtype](https://github.com/vllm-project/vllm/pull/69875)（`open`，实现改动 1 处 + 新增 61 行单测）

### 问题

PP 的每一段都要预先分配 downstream 要接收的张量，入口是 `reserve_aux_intermediate_tensor_slots()`（`vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py`）。它先调模型自己的 `make_empty_intermediate_tensors(batch_size, dtype, device)`（`dtype`＝模型 dtype，如 bf16），再给每个 aux 槽位手工 `torch.zeros((batch, hidden_size), dtype=dtype, device=device)`。

问题是 aux hidden states 取自**残差流**，而残差流的 dtype 不保证等于模型 dtype：PR 注释点名 Muse Spark 在 bf16 模型里保留 fp32 残差流。于是发送端产出 fp32 aux、接收端缓冲是 bf16 —— **PP 跨段时被静默降精度**，没有任何报错。

### 修法

```python
aux_dtype = (
    tensors["hidden_states"].dtype
    if "hidden_states" in tensors.tensors
    else dtype
)
for key in upstream_keys:
    tensors[key] = torch.zeros((batch_size, hidden_size), dtype=aux_dtype, device=device)
```

即以同一次 `make_empty` 产出的 `hidden_states` 的 dtype 为准，没有该 buffer 时才回落到模型 dtype。测试 `test_eagle3_pp_aux_dtype.py` 用三个参数分别钉住：`fp32 → fp32`（残差流比模型宽）、`bf16 → bf16`、`无 hidden_states → bf16`。

### 证据与验收含义

```python
@pytest.mark.parametrize("hidden_states_dtype,expected", [
    (torch.float32, torch.float32),
    (torch.bfloat16, torch.bfloat16),
    (None, torch.bfloat16),
])
```

**NPU 判断：与 Ascend 直接相关。** vllm-ascend 的 DeepSeek-V4 系列在 PP intermediate tensors 里就带着 `FP32 mixing state`（fp32 残差/mixing），而 Ascend 的 aux 传输是模型自己搬（`pp_transport_aux_hidden_states_`）。如果 Ascend 侧也按模型 dtype 预分配 aux 槽位，就是同一类静默降精度。验收断言应是「**aux 槽位 dtype == hidden_states dtype**」，而不是「PP 能跑通」。

顺带说明这条与上期的 #63436（DFlash 的 aux FC 用 draft dtype）是同一根因的两次发作：**「aux 的 dtype 跟残差流走，不跟模型 dtype 走」这个假设在两处都错过一次**。

## 其余条目（一句话）

- [#69064](https://github.com/vllm-project/vllm/issues/69064)（新）：VLM 走 `extract_hidden_states` 提不出正确结果——多模态 target 的 aux 路径仍未打通。
- [#53243](https://github.com/vllm-project/vllm/issues/53243)：`extract_hidden_states` 高估 DeepSeek-V4 KV cache 内存（与 #56822 同族的计费问题），仍 open。
- [#68824](https://github.com/vllm-project/vllm/pull/68824)（上期新建）：DeepSeek-V3.2「少最后一层 aux」的修复**本期仍未合入**，#68813 仍 open → 少一层的问题在 main 上还没修。
- [#52073](https://github.com/vllm-project/vllm/pull/52073)：NemotronH 混合模型的提取测试，覆盖补齐。
- [#65991](https://github.com/vllm-project/vllm/pull/65991)：关掉 extract_hidden_states decode draft 的 K=0 开关，11-24 更新，仍 open。
- vllm-ascend [#21831](https://github.com/vllm-project/vllm-ascend/pull/21831)（**已合入** 11-23）：GLM-5.3-Flash DFlash2 **并打通 PP**（aux 接口 + mHC 延后残差补全 + SP 安全捕获），是本期 Ascend 唯一「aux 捕获 + PP」同时合入的能力项。
- vllm-ascend [#22485](https://github.com/vllm-project/vllm-ascend/pull/22485)（已合入 11-20）：DSpark 跳过不用的 DeepSeek V4.1 HC buffer 操作，属 aux/HC 链路开销清理。
- vllm-ascend [#22198](https://github.com/vllm-project/vllm-ascend/pull/22198)（已合入 11-24）：Main2Main v0.34.0；升级后必须重跑 aux/PP 用例（上游本周期做了 relay 集中化、CPU-pages、draft 独立 PP 段）。
- vllm-ascend #22307（已合入 11-25，`9b7a557f`）：Ascend 代码路径收敛为 MRV2 only，删掉 V1 分支——与 aux 双轨清理同向。

## 无实质进展与边界

- vLLM [#57736](https://github.com/vllm-project/vllm/pull/57736)、[#64320](https://github.com/vllm-project/vllm/pull/64320)、[#56442](https://github.com/vllm-project/vllm/pull/56442)、[#50187](https://github.com/vllm-project/vllm/pull/50187)（本期关闭未合入）：无实质进展。
- vllm-ascend #21792（CPU pages）本期无更新；#21023/#17557 上期已判为 superseded。

## 检索与复核

- [本期结构化数据](vllm-extract-hidden-states-delta-2026-11-27.json) · [本期 CSV](vllm-extract-hidden-states-delta-2026-11-27.csv)。
- 深读依据：`gh pr diff 69875` 全文（含新单测）；其余用状态/提交/时间线。未运行 GPU/NPU。

## 扫描覆盖

- 关键词：vLLM 9 短语、vllm-ascend 10 短语 × `updated:>=2026-11-20` + RFC + `created:` 切分；去重 1534 条（1257 PR + 277 Issue）/ 1002 条（849 + 153）；候选 47 / 30 → 纳入 9 条（6 vLLM + 3 ascend）。
- 合入侧：窗口内 454（vLLM）/ 285（vllm-ascend）个提交。
