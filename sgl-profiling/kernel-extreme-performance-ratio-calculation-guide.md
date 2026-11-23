# NPU 算子极致性能比计算指引

更新日期：2026-11-23。

本指引固化本次 Ascend `kernel_details.csv` 分析采用的方法：保留原始数据，在明确的采样范围内剔除通信与重复边界，逐调用计算加权占比，再统计平均耗时。采集流程另见 [SGLang NPU Profiling 采集与解析指南](sglang-npu-profiling-guide.md)。

本文将两组加权占比之和记为“极致性能比” `R`，定义方向为**按指定规则加权的时间量 / 当前计算算子累计耗时**。它是本次分析约定的指标，不自动取倒数，也不等同于端到端吞吐提升、设备利用率或严格理论性能极限。

## 1. 先确定统计范围

每次统计必须记录：来源文件、模型/专家数、TP/EP、rank/device、prefill/decode、batch、步数或层范围，以及是否对首尾做过裁剪。缺少的配置标为未确认，不从文件名补全事实。

- 一份 rank/device 的数据单独计算分母和占比。多个 rank 不能未经说明合并后当作单卡或引擎墙钟耗时。
- 完整采样和四层局部截取是两种范围，结果不得混用。
- CSV 用 CSV 解析器读取，不能按逗号直接切分：形状字段自身包含逗号和引号。
- 标准 CSV 第一行为列名。手工整理的 XLSX 可能没有表头，第一行就是算子数据；应先检查，不能直接丢弃。
- 数值以未舍入的 `Duration(us)` 和原始 ratio 计算，最后才设置展示精度。

## 2. 如何剔除记录

### 2.1 原始数据不修改

完整复制输入，保留一个原始数据 sheet。后续筛选、删除重复边界、计算和汇总只在新 sheet 或新的分析文件中进行，并保留原始行号或其他可追溯标识。

### 2.2 排除通信及通信 launcher

本次数据使用以下识别规则：对 `Type` 做去空白、忽略大小写的前缀判断，排除 `hcom*` 和 `hccl*`。典型记录包括：

- `hcom_allReduce_AicpuKernel`；
- `hcom_allGather_AicpuKernel`；
- `HcclLaunchAicpuKernel`。

排除这些记录的**全部调用**，它们不参与分母、大小算子分组、加权占比和计算算子平均耗时。原始数据 sheet 仍保留它们。

对于新版本或其他通信后端，应核对 `Name`、`Type`、`Accelerator Core` 和时间线，扩充明确的通信清单；上述前缀不是所有平台的通用通信识别器。不能仅因记录运行在 `AI_CPU` 上就将其全部剔除。

`Cast`、`Transpose`、`Index`、`IndexPut`、`Slice`、`Concat`、`Fill` 等非通信设备任务仍计入本指标的“计算算子”。这里的计算口径包含数据搬运和索引开销，不只包含矩阵乘。

通信耗时如需分析，另表统计。通信 launcher 与 collective/计算可能重叠，不能把它们的 Duration 再加进本指标或直接累加成端到端延迟。

### 2.3 只在确认重复边界时删除末尾 attention

此前四层截取以一次 `SparseFlashAttention` 开始，以下一次 `SparseFlashAttention` 结束，形成“一个完整四层周期，加上下一个周期的 attention”的范围。对该截取：

1. 保留开头的 `SparseFlashAttention`。
2. 从统计范围排除末尾属于下一个周期的那一次 `SparseFlashAttention`。
3. 同时从总计算耗时分母、分组明细、调用次数和平均值中排除它。

**不是所有文件都删除最后一个 `SparseFlashAttention`。** 必须通过执行顺序、模型层数/周期或时间线确认它确实超出目标范围。

本次完整 8 步 decode 数据每步 11 次稀疏注意力，共 `8 × 11 = 88` 次，符合完整模型计数，不能套用四层截取的末尾剔除规则。完整文件最后一条记录也不一定是 attention。

### 2.4 不按算子名称去重，也不任意删除慢调用

- 同一算子在不同层、不同 step 的重复调用都是有效样本。图重放还可能复用 Task ID，不能只按 Name 或 Task ID 去重。
- 如果确认有导出副本重复，需要结合 rank/device、时间戳、任务标识等确定同一物理事件，并记录删除依据。
- 完整采样默认保留全部范围内调用。首步、长耗时和边界异常不能只因“看起来慢”就剔除。
- 若另外做稳态分析，应明确指定共同的稳定窗口，重新筛选、计算分母和所有统计，并将结果标为稳态口径。

## 3. 分母、阶段 ratio 与阈值

### 3.1 两组共用的分母

令 `C` 为完成上述范围选择与剔除后，保留下来的所有非通信调用集合。每条调用耗时为 `d_i`，单位 μs：

```text
T_compute = Σ(d_i), i ∈ C
```

分母包括大于、小于以及恰好等于 10 μs 的有效计算调用。它不是“大算子总耗时”或“小算子总耗时”，也不包含通信、host 调度、Free、Preparing、stream 等待或采集解析时间。

`T_compute` 是 Duration 累计值，**不是时间线区间并集或 step 墙钟耗时**。如果计算任务之间有重叠，应保留这一方法定义，并另行报告墙钟/重叠分析；不能在公式中悄悄换成墙钟分母。

每个数据集使用自己的 `T_compute`，比较两个模型时不得共用其中一个分母。

### 3.2 最大阶段 ratio 取哪些列

对每条调用，分别从以下十个执行阶段列取最大值，记为 `m_i`：

```text
aic_mac_ratio
aic_scalar_ratio
aic_mte1_ratio
aic_mte2_ratio
aic_mte3_ratio
aic_fixpipe_ratio
aiv_vec_ratio
aiv_scalar_ratio
aiv_mte2_ratio
aiv_mte3_ratio
```

```text
m_i = MAX(该调用的十个执行阶段 ratio)
```

不使用以下字段：`aic_icache_miss_rate`、`aiv_icache_miss_rate`、`cube_utilization(%)`、各阶段绝对时间或 cycles。

- ratio 按原值使用，例如 `0.8`，不要再除以 100；显示成百分比不改变其数值。
- 不将十个 ratio 求和，不求平均，也不把组内不同调用的最大 ratio 当成全部调用的 ratio。
- 不将 `m_i` 截断到 0.8；大算子按公式直接计算 `m_i / 0.8`。
- `Duration` 或 ratio 缺失、非数值、非有限值时必须报告并复核。不能把 `N/A`、空值或解析失败直接补成 0，形成虚低占比。确定为“不适用”的阶段可以明确排除该字段，但应披露处理方式；所有阶段都无有效值的调用不能产生有效加权占比。
- 本次实际计算使用的非通信样本均有有效的十阶段数值。下面参考代码对缺失值采用报错策略。

### 3.3 严格采用大于和小于

```text
大算子集合 B：d_i > 10 μs
小算子集合 S：d_i < 10 μs
边界集合 E：d_i = 10 μs
```

本次数据没有恰好等于 10 μs 的调用。以后若出现，按当前严格规则：`E` 仍计入 `T_compute`，但不计入 `B`、`S` 的加权小计；单独披露其数量和累计耗时，不能声称两组覆盖了全部调用。若要改为 `<=10 μs`，需显式变更口径，不能沿用同一结果标签。

## 4. 最终采取的计算公式

### 4.1 大于 10 μs

```text
p_i = (d_i / T_compute) × (m_i / 0.8), i ∈ B
R_big = Σ(p_i), i ∈ B
```

### 4.2 小于 10 μs

```text
p_i = (d_i / T_compute) × m_i, i ∈ S
R_small = Σ(p_i), i ∈ S
```

0.8 只用于大算子组，小算子组不除以 0.8。两组必须逐调用计算后再求和。

### 4.3 总体指标与其他百分比

```text
极致性能比 R = R_big + R_small
百分比展示 = R × 100%

原始时间占比 q_i = d_i / T_compute
```

`R_big`、`R_small` 和 `R` 直接展示计算结果，不重新归一化到 100%。由于大算子存在 `/0.8`，加权结果也不保证小于原始耗时占比，不能将其强制截断为 100%。

原始时间占比与 ratio 加权占比是两个指标，必须分列命名。不要用普通 Duration 占比替代用户指定的加权公式。

如果进一步给出 `1/R`，必须单独命名并说明额外假设；本指引的最终指标默认就是 `R`，不直接将倒数当作实测吞吐加速比。

### 4.4 算子平均耗时

对某一分组 `g`：

```text
n_g      = 本组调用次数
D_g      = Σ(d_i), i ∈ g
mean_g   = D_g / n_g
min_g    = MIN(d_i), i ∈ g
max_g    = MAX(d_i), i ∈ g
```

建议在同一个平均耗时 sheet 中提供两种粒度：

1. **按 Type 汇总**：回答某类算子总体的次数、累计耗时、平均耗时。此粒度可能混合不同形状，应明确标注。
2. **按具体工作量细分**：至少按 `(Name, Type, Accelerator Core, Input Shapes, Output Shapes, Input Data Types, Output Data Types)` 分组，避免把不同形状或 dtype 的调用当作同一工作量比较。

平均耗时页使用 `C` 中所有有效计算调用，不受 `>10`、`<10` 分组限制，也不按两组分别截断后再求一个未加权平均。

分组平均值合并时，必须按次数加权：

```text
总体平均 = Σ(D_g) / Σ(n_g)
```

不能直接将各组 `mean_g` 做普通平均，除非各组调用次数恰好相同。

## 5. 分组占比和对齐比较

对某类算子或某个形状组，分别保留：

```text
累计时间占比         = D_g / T_compute
单次平均时间占比     = mean_g / T_compute
ratio 加权占比       = Σ(p_i), i ∈ g
```

第一项反映该组所有调用的总成本，第二项反映一次平均调用相对当前整个统计窗口的比例。两者相差调用次数，不可混淆。

累计时间占比、逐调用加权占比可以在不重复的分组间求和；单次平均时间占比不能跨不同组简单相加后称为总体占比。

比较两个数据集时：

- 按执行顺序和 Name 对齐，并用输入输出形状辅助处理重复的 Cast 等算子。
- 不按阈值组分别强行对齐。同一算子可能在一边大于 10 μs，在另一边小于 10 μs。
- 单侧额外算子保留并标明，对端留空，不按 0 μs 计算“提速”。其耗时仍计入本侧完整统计范围的分母。
- 本次差值方向统一为 `Top16 − Full`；平均耗时变化率为 `mean_Top16 / mean_Full − 1`。
- 占比差以百分点表示：`(share_Top16 − share_Full) × 100`，不要与相对百分比变化混用。
- 只统计“形状不同”子集时，分子筛选该子集，分母仍用本侧全部有效计算调用，不改成“形状不同算子的耗时总和”。

## 6. 输出表格约定

每份输入生成新的工作簿，保留原文件不变，采用四个 sheet：

1. **原始数据**：完整复制 CSV，包含通信，保留原列顺序和数据值。
2. **大于10us**：仅复制 `B` 的原始数据行，顺序不变；右侧新增逐调用加权占比，末尾用 `SUM` 求和。
3. **小于10us**：仅复制 `S` 的原始数据行，顺序不变；右侧新增逐调用加权占比，末尾用 `SUM` 求和。
4. **平均耗时**：按 Type 汇总和按名称/形状/dtype 细分，至少包含次数、累计、平均、最短、最长和累计时间占比。

两个阈值 sheet 都显示同一个 `T_compute` 和各自公式。原始数据页保留被剔除记录，统计页说明排除条件及数量，确保可以复算。公式保留为 Excel 公式，结果无需归一化。

### 6.1 Excel 公式示例

以下列号仅适用于已核对的标准 48 列 `kernel_details` 布局；新文件必须优先按列名映射，不能盲目套列号。

- `J`：Duration。
- `F`：Type。
- 十阶段 ratio：`Y, AA, AC, AE, AG, AI, AN, AP, AR, AT`。
- `AW`：新增加权占比。
- 假设 `$AZ$2` 存放本侧 `T_compute`。

```excel
大于10us页，第2行：
=J2/$AZ$2*(MAX(Y2,AA2,AC2,AE2,AG2,AI2,AN2,AP2,AR2,AT2)/0.8)

小于10us页，第2行：
=J2/$AZ$2*MAX(Y2,AA2,AC2,AE2,AG2,AI2,AN2,AP2,AR2,AT2)

本页占比合计，末行号以实际筛选数量为准：
=SUM(AW2:AW3635)

平均耗时：
=本组Duration累计/本组调用次数
```

对于没有额外边界排除的完整 CSV，分母可以引用原始数据页：

```excel
=SUMIFS('原始数据'!J2:J22097,'原始数据'!F2:F22097,"<>hcom*",'原始数据'!F2:F22097,"<>hccl*")
```

对于四层截取，必须额外排除已经确认重复的边界行；数据和分母引用范围必须保持一致。

如果汇总表用文本 `>10us` 或 `<10us` 标记分组，`SUMIFS` 要做**字面文本匹配**，不能误触发比较运算符：

```excel
匹配文本 >10us 的条件："=>10us"
匹配文本 <10us 的条件："=<10us"
```

## 7. 可手工复核的示例

以下为构造数据，通信和重复边界已经剔除，计算总耗时为 100 μs：

| 调用 | Duration（μs） | 最大阶段 ratio | 分组 | 原始时间占比 | 公式加权占比 |
| --- | ---: | ---: | --- | ---: | ---: |
| A | 40 | 0.80 | >10 μs | 40% | 40% |
| B | 30 | 0.64 | >10 μs | 30% | 24% |
| C | 20 | 0.40 | >10 μs | 20% | 10% |
| D | 6 | 0.60 | <10 μs | 6% | 3.6% |
| E | 4 | 0.50 | <10 μs | 4% | 2% |

```text
R_big   = 40% + 24% + 10% = 74%
R_small = 3.6% + 2%       = 5.6%
R       = 79.6%
全部调用平均耗时 = 100 / 5 = 20 μs
```

本次实际完整 CSV 的复核值如下，均保留完整 8 步、排除通信，没有执行四层末尾剔除：

| 数据集 | 有效计算调用数 | T_compute（μs） | >10 μs 调用数 | R_big | <10 μs 调用数 | R_small |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full TP16 EP1 | 20624 | 162108.365 | 3634 | 55.3122% | 16990 | 18.0943% |
| Top16 TP16 | 21416 | 165539.524 | 3412 | 53.5177% | 18004 | 18.9934% |

这些数字是各自 profiling 样本内的加权统计，不是端到端 benchmark。

## 8. 最小参考实现

示例直接按列名读取；输入范围和 `exclude_source_rows` 必须事先确认。行号按文件的物理行位置计数：表头为第 1 行、第一条数据为第 2 行。本次标准导出的每条记录各占一行；若存在字段内换行，应改用明确的记录标识，不能套用这个物理行号约定。

```python
import csv
import math
from collections import defaultdict

RATIO_FIELDS = (
    "aic_mac_ratio", "aic_scalar_ratio", "aic_mte1_ratio",
    "aic_mte2_ratio", "aic_mte3_ratio", "aic_fixpipe_ratio",
    "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio",
    "aiv_mte3_ratio",
)
GROUP_FIELDS = (
    "Name", "Type", "Accelerator Core", "Input Shapes", "Output Shapes",
    "Input Data Types", "Output Data Types",
)

def number(value, field):
    result = float(value)  # N/A、空值等直接报错，不能替换成 0
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid {field}: {value!r}")
    return result

def analyze(path, exclude_source_rows=()):
    excluded = set(exclude_source_rows)
    compute = []
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for source_row, row in enumerate(csv.DictReader(stream), start=2):
            if source_row in excluded:
                continue
            kind = row["Type"].strip().lower()
            if kind.startswith(("hcom", "hccl")):
                continue
            duration = number(row["Duration(us)"], "Duration(us)")
            max_ratio = max(number(row[key], key) for key in RATIO_FIELDS)
            compute.append((source_row, row, duration, max_ratio))

    total = math.fsum(item[2] for item in compute)
    if total <= 0:
        raise ValueError("No positive compute duration")

    big, small, equal = [], [], []
    groups = defaultdict(list)
    for source_row, row, duration, max_ratio in compute:
        groups[tuple(row[key] for key in GROUP_FIELDS)].append(duration)
        if duration == 10:
            equal.append((source_row, duration))
            continue
        divisor = 0.8 if duration > 10 else 1.0
        share = duration / total * max_ratio / divisor
        (big if duration > 10 else small).append((source_row, share))

    r_big = math.fsum(share for _, share in big)
    r_small = math.fsum(share for _, share in small)
    averages = {
        key: {
            "count": len(values),
            "sum_us": math.fsum(values),
            "mean_us": math.fsum(values) / len(values),
            "min_us": min(values),
            "max_us": max(values),
        }
        for key, values in groups.items()
    }
    return {
        "compute_count": len(compute), "compute_total_us": total,
        "big": big, "small": small, "equal_10us": equal,
        "r_big": r_big, "r_small": r_small, "r_total": r_big + r_small,
        "averages": averages,
    }
```

返回的占比为小数，显示百分比时再乘 100 或使用 Excel 百分比格式。没有重复边界的完整文件调用 `analyze(path)`，不得无条件设置“删除最后一个 attention”。

## 9. 最后采用的规则与验收

最终口径固定为：

1. 保留原始数据，先确定单 rank、单阶段和相同采样范围。
2. 排除通信及通信 launcher；仅在已确认超出目标范围时排除重复边界。
3. 分母为上述范围内全部非通信调用的 Duration 累计，两组共用。
4. `>10 μs` 用 `duration / T_compute × max_ratio / 0.8`；`<10 μs` 用 `duration / T_compute × max_ratio`。
5. ratio 逐调用取十阶段最大值，不取 miss rate 或 cube utilization，不截断、不补缺失为零。
6. 先逐调用计算，再分组求和；保留原始时间占比与加权占比的区别。
7. 平均耗时采用 `累计Duration / 调用次数`；同时提供类型汇总与名称/形状/dtype 细分。
8. 恰好 10 μs 单独披露；不强行归入任一严格阈值组。

交付前核对：原文件未改动；分母和筛选范围一致；调用数与模型层数/step 数相符；各组累计及次数能回归总体；平均值按次数加权；公式可重算且无错误；缺失字段和额外剔除均有记录；百分比只在最后格式化，保留未舍入计算值。
