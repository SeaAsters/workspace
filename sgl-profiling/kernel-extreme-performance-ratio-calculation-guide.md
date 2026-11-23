# NPU 算子极致性能比计算指引

更新日期：2026-11-23。

本指引固化本次 Ascend `kernel_details.csv` 分析采用的方法：保留原始数据，在明确的采样范围内剔除通信与重复边界，先对同一逻辑位置的重复采样求平均耗时，再计算百分比和 ratio 加权占比。采集流程另见 [SGLang NPU Profiling 采集与解析指南](sglang-npu-profiling-guide.md)。

本文将两组加权占比之和记为“极致性能比” `R`，定义方向为**按指定规则加权的时间量 / 全部逻辑行平均耗时之和**。它是本次分析约定的指标，不自动取倒数，也不等同于端到端吞吐提升、设备利用率或严格理论性能极限。

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
3. 在求平均前排除该边界记录，使它不进入有效采样、平均值及后续分母。

**不是所有文件都删除最后一个 `SparseFlashAttention`。** 必须通过执行顺序、模型层数/周期或时间线确认它确实超出目标范围。

本次完整 8 步 decode 数据每步 11 次稀疏注意力，共 `8 × 11 = 88` 次，符合完整模型计数，不能套用四层截取的末尾剔除规则。完整文件最后一条记录也不一定是 attention。

### 2.4 不按算子名称去重，也不任意删除慢调用

- 同一算子在不同层、不同 step 的重复调用都是有效样本。图重放还可能复用 Task ID，不能只按 Name 或 Task ID 去重。
- 如果确认有导出副本重复，需要结合 rank/device、时间戳、任务标识等确定同一物理事件，并记录删除依据。
- 完整采样默认保留全部范围内调用。首步、长耗时和边界异常不能只因“看起来慢”就剔除。
- 若另外做稳态分析，应明确指定共同的稳定窗口，重新筛选、计算分母和所有统计，并将结果标为稳态口径。

## 3. 修订后的统计单位：先求平均，保留四层表的行结构

**最终采用“逻辑位置平均耗时口径”，不再以逐调用累计耗时作为主表百分比的分母。** 大算子阈值改为 `>=10 μs`，等于 10 μs 计入大算子组。

### 3.1 一行对应哪个算子

主表沿用此前四层统计表的执行顺序：一行对应四层范围内的一个固定逻辑位置。对多个完整采样周期或 step 中该位置的重复调用求平均。

“沿用四层表形式”指按逻辑位置逐行呈现，并不要求把完整模型数据强制裁成四层。分析完整模型时，以一次完整 forward 的全部逻辑位置为模板；只选四层时，以一个完整四层周期为模板。后文的均值、分母和分组公式均覆盖实际选定的完整模板。

例如，同一个四层周期有三次 KDA recurrent：即使 Name 和形状相同，它们属于不同层，也保留为三行。每行分别对相同层、相同位置的多次采样求平均，不能把三行压成一行后令该算子在分母中只出现一次。

确定逻辑行时，核对以下信息：

- 逻辑层号、周期内执行序号或经验证的序列对齐位置；
- `Name`、`Type`、`Accelerator Core`；
- `Input Shapes`、`Output Shapes`；
- `Input Data Types`、`Output Data Types`。

`kernel_details.csv` 不一定直接提供逻辑层号，必要时结合时间线或重复执行序列建立“原始行号 → 逻辑位置”的映射。不能只按 Name 或 Task ID 分组。

同一位置若出现不同形状、batch 或执行路径，应先拆成可比较的采样窗口，不把互斥路径的多个形状均值同时加进一个四层分母。采样周期不完整、各位置样本数不一致时，先复核并选取完整范围，不静默补零或删掉慢样本。

### 3.2 平均耗时

令 `g` 为主表中的逻辑行，`n_g` 为该行有效采样次数，`d_g,j` 为每次采样 Duration，单位 μs：

```text
mean_g = Σ(d_g,j) / n_g
min_g  = MIN(d_g,j)
max_g  = MAX(d_g,j)
```

应保留 `n_g`、最短和最长耗时供复核，但百分比使用 `mean_g`。不先按单次 Duration 分成大小算子，否则平均值接近阈值的同一逻辑行会被拆到两个组。

### 3.3 分母是全部逻辑行平均耗时之和

```text
T_mean = Σ(mean_g)，求和覆盖统计范围内全部非通信逻辑行
```

`T_mean` 是**平均四层算子累计时间**，不是一次随机调用的平均时间，也不是 `Σ(原始全部调用 Duration)`，更不是时间线并集或墙钟时延。

- 四层中的不同位置均计入，即使同名、同形状也不能漏计其出现次数。
- `>=10 μs` 和 `<10 μs` 两个 sheet 使用同一个 `T_mean`，不能分别拿本组均值之和作分母。
- Full 和 Top16 分别计算自己的 `T_mean`。
- 仅查看“形状不同”子集时，只筛选分子，分母仍是本侧完整四层全部逻辑行平均耗时之和。
- 若只截取了一个完整四层周期，每个位置只有一次采样，`mean_g` 等于该行 Duration；若有多步采样，先跨采样求平均再计算。

如另行给出原始累计耗时，标为 `T_raw = Σ(d_g,j)`，只用于追溯。不得把 `T_raw` 和 `T_mean` 混用。

## 4. 阶段 ratio 与最终计算规则

### 4.1 最大阶段 ratio

使用以下十个执行阶段列：

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

为使表中一行表示同一位置的平均采样，本指引统一采用：**先对每个阶段分别取采样均值，再在十个阶段均值之间取最大值**。

```text
stage_mean_g,s = Σ(ratio_g,j,s) / n_g
m_g = MAX(stage_mean_g,s)，s 遍历十个执行阶段
```

单次采样时即为该调用的十阶段最大值。不要取所有采样中出现过的最大峰值，也不要把 `MAX(阶段均值)` 与 `平均(逐次最大值)` 混用；它们一般不相等。

不使用 `aic_icache_miss_rate`、`aiv_icache_miss_rate`、`cube_utilization(%)`、各阶段绝对时间或 cycles。

- ratio 按原小数使用，例如 `0.8`，不再除以 100。
- 不将十阶段求和，不将最大阶段 ratio 截断到 0.8。
- Duration 或 ratio 缺失、非数值或非有限值时应报告并复核，不能直接补 0。确认“不适用”的阶段可以明确排除，但需披露；参考代码默认缺失即报错。

### 4.2 阈值按平均耗时判断

```text
大算子集合 B：mean_g >= 10 μs
小算子集合 S：mean_g < 10 μs
```

**平均耗时恰好为 10 μs，计入大算子组，使用 /0.8。** 两组互斥并覆盖全部有效逻辑行，不再单列“等于10 μs但不参与加权”的集合。

同一位置两次耗时分别为 8 和 12 μs，平均为 10 μs，则整行进入 `>=10 μs` 组；不能仍把这两次调用拆进不同阈值组。

### 4.3 先平均，再求百分比

```text
原始平均耗时占比：
q_g = mean_g / T_mean

mean_g >= 10 μs：
p_g = (mean_g / T_mean) × (m_g / 0.8)

mean_g < 10 μs：
p_g = (mean_g / T_mean) × m_g

R_big   = Σ(p_g)，g ∈ B
R_small = Σ(p_g)，g ∈ S
R       = R_big + R_small
```

展示百分比时才乘 100 或设置 Excel 百分比格式。`Σ(q_g) = 100%`；ratio 加权后的 `R` 不重新归一化，也不强行截断为 100%。0.8 只用于大算子组。

**最终顺序固定为：范围/剔除 → 逻辑位置对齐 → 求 Duration 和各阶段 ratio 的均值 → 以平均 Duration 分组 → 平均耗时之和作分母 → 算百分比和加权占比 → 求和。**

原版按原始逐调用累计 Duration 计算的百分比与本版口径不同，历史结果必须标明旧口径，不应直接作为本版复核值。

## 5. 表格保持四层统计的形式

### 5.1 单份数据

原数据保持不变，新建以下 sheet：

1. **原始数据**：完整复制原 CSV，包括通信，保留列顺序和原始行号。
2. **平均耗时**：保留四层中各逻辑位置的顺序，一行一个位置；Duration 列填该位置平均耗时，各阶段 ratio 列填对应阶段采样均值。
3. **大于等于10us**：从平均耗时表筛选 `mean_g >= 10` 的行；顺序与原四层表一致，末尾 `SUM` 加权占比。
4. **小于10us**：筛选 `mean_g < 10` 的行；顺序与原四层表一致，末尾 `SUM` 加权占比。

阈值 sheet 不再复制原始逐调用明细来直接计算主指标；复制的是**已经求平均后的四层逻辑行**。表格主体仍是熟悉的原字段布局，只在右侧增加采样次数、原始行号集合、平均耗时占比和加权占比等必要列。

平均耗时表应包含：逻辑位置、Name/Type、输入输出形状、Core/dtype、采样次数、平均/最短/最长 Duration、十阶段 ratio 均值、最大阶段均值 `m_g`、阈值组、`q_g` 和 `p_g`。

若另外按 Type 或 Name/形状汇总，只作为辅助视图，并保留原四层位置的出现次数；不能用无位置、去重后的均值表替代主表分母。

### 5.2 Full 与 Top16 并排比较

沿用四层对齐表形式，同一逻辑位置放在同一行，左右分别列出：

- 采样次数、平均耗时（μs）、最大阶段 ratio 均值；
- 平均耗时占本侧 `T_mean` 的百分比、ratio 加权占比；
- 输入输出形状、平均耗时差、平均耗时变化率、占比差（百分点）。

按逻辑位置及执行顺序对齐，并用 Name 和形状辅助核对。不要先按阈值拆开再对齐，因为同一位置可能在两边属于不同的阈值组。

单侧独有算子保留，对端留空，不按 0 μs 计算提速；该行平均耗时仍计入本侧分母。差值统一为 `Top16 − Full`，耗时变化率为 `mean_Top16 / mean_Full − 1`。

## 6. Excel 公式示例

以下假定已生成一行一个逻辑位置的平均耗时表，且沿用核对过的标准 48 列布局：

- `J`：该位置的平均 Duration，而不是某次调用或累计 Duration。
- `Y, AA, AC, AE, AG, AI, AN, AP, AR, AT`：十阶段 ratio 的采样均值。
- `AW`：按最终规则计算的加权占比。
- `AZ2`：`T_mean`，引用完整平均耗时表所有逻辑行的 J 列之和。

```excel
平均耗时表的分母，末行按实际四层逻辑行数填写：
=SUM('平均耗时'!J2:J237)

原始平均耗时占比：
=J2/$AZ$2

统一的加权占比公式，等于10也除以0.8：
=J2/$AZ$2*MAX(Y2,AA2,AC2,AE2,AG2,AI2,AN2,AP2,AR2,AT2)/IF(J2>=10,0.8,1)

阈值标记：
=IF(J2>=10,">=10us","<10us")

筛选后的占比合计，范围按该页实际行数填写：
=SUM(AW2:AW41)
```

不能再用原始数据页的 `SUMIFS(Duration)` 作为新口径的分母。若阈值标签是文本，`SUMIFS` 字面匹配条件分别为 `"=>=10us"` 和 `"=<10us"`，避免将标签当成比较运算。

## 7. 可手算的修订示例

假设同一个四层范围暂以三个逻辑位置演示，每个位置有两次有效采样；阶段 ratio 不随两次采样变化：

| 逻辑位置 | 两次 Duration（μs） | 平均 Duration（μs） | 最大阶段均值 | 分组 | 平均耗时占比 | 加权占比 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| A | 8、12 | 10 | 0.8 | >=10 μs | 28.5714% | 28.5714% |
| B | 18、22 | 20 | 0.4 | >=10 μs | 57.1429% | 28.5714% |
| C | 4、6 | 5 | 0.5 | <10 μs | 14.2857% | 7.1429% |

```text
T_mean  = 10 + 20 + 5 = 35 μs
T_raw   = 8 + 12 + 18 + 22 + 4 + 6 = 70 μs（仅用于追溯）
R_big   = 57.1429%
R_small = 7.1429%
R       = 64.2857%
```

A 的平均值等于 10，完整归入大算子组。分母使用 35，而不是原始调用累计的 70。

## 8. 最小参考实现

代码按列名读取。`position_by_source_row` 是人工确认或序列对齐后得到的“原始CSV数据行 → 四层逻辑位置”映射，不能用 Name 直接代替位置。表头计第1行，第一条数据计第2行；有字段内换行时应改用明确的记录标识。

输入应已限定为相同 batch/形状的完整重复周期。代码要求各逻辑行样本数一致；不一致时先修正采样范围，不能静默补齐。

```python
import csv
import math
from collections import OrderedDict

RATIO_FIELDS = (
    "aic_mac_ratio", "aic_scalar_ratio", "aic_mte1_ratio",
    "aic_mte2_ratio", "aic_mte3_ratio", "aic_fixpipe_ratio",
    "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio",
    "aiv_mte3_ratio",
)
IDENTITY_FIELDS = (
    "Name", "Type", "Accelerator Core", "Input Shapes", "Output Shapes",
    "Input Data Types", "Output Data Types",
)

def number(value, field):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid {field}: {value!r}")
    return result

def analyze(path, position_by_source_row, exclude_source_rows=()):
    excluded = set(exclude_source_rows)
    groups = OrderedDict()
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for source_row, row in enumerate(csv.DictReader(stream), start=2):
            if source_row in excluded:
                continue
            if row["Type"].strip().lower().startswith(("hcom", "hccl")):
                continue
            position = position_by_source_row[source_row]
            identity = tuple(row[key] for key in IDENTITY_FIELDS)
            group = groups.setdefault(position, {"identity": identity, "samples": []})
            if group["identity"] != identity:
                raise ValueError(f"Shape/path changed at position {position!r}")
            duration = number(row["Duration(us)"], "Duration(us)")
            ratios = [number(row[key], key) for key in RATIO_FIELDS]
            group["samples"].append((source_row, duration, ratios))

    if not groups:
        raise ValueError("No compute samples")
    if len({len(group["samples"]) for group in groups.values()}) != 1:
        raise ValueError("Incomplete cycles: sample counts differ")

    rows = []
    for position, group in groups.items():
        samples = group["samples"]
        count = len(samples)
        durations = [sample[1] for sample in samples]
        stage_means = [
            math.fsum(sample[2][s] for sample in samples) / count
            for s in range(len(RATIO_FIELDS))
        ]
        rows.append({
            "position": position, "identity": group["identity"],
            "source_rows": [sample[0] for sample in samples],
            "count": count, "mean_us": math.fsum(durations) / count,
            "min_us": min(durations), "max_us": max(durations),
            "stage_means": stage_means, "max_ratio": max(stage_means),
        })

    total_mean = math.fsum(row["mean_us"] for row in rows)
    if total_mean <= 0:
        raise ValueError("No positive mean duration")
    for row in rows:
        large = row["mean_us"] >= 10
        row["bucket"] = ">=10us" if large else "<10us"
        row["time_share"] = row["mean_us"] / total_mean
        row["weighted_share"] = row["time_share"] * row["max_ratio"] / (0.8 if large else 1)
    r_big = math.fsum(row["weighted_share"] for row in rows if row["bucket"] == ">=10us")
    r_small = math.fsum(row["weighted_share"] for row in rows if row["bucket"] == "<10us")
    return {"rows": rows, "total_mean_us": total_mean,
            "r_big": r_big, "r_small": r_small, "r_total": r_big + r_small}
```

返回百分比是小数，最后再设置百分比格式。逻辑位置的顺序沿用确认后的四层顺序，不能按算子类型排序后打乱主表。

## 9. 最后采用的规则与验收

1. 原数据不修改；先限定单rank、单阶段、完整且同工作量的统计范围。
2. 排除通信和确认重复的边界，保留范围内全部有效采样。
3. 四层的每个逻辑位置保留一行，对同一位置的重复采样先求平均；不同层的同名算子不合并。
4. 各阶段 ratio 先分别求采样均值，再取十阶段均值的最大值。
5. 两组共用 `T_mean = 全部逻辑行平均Duration之和`；不使用原始调用累计Duration作分母。
6. 平均Duration `>=10 μs`（包含等于10）用 `/0.8`；平均Duration `<10 μs` 不除以0.8。
7. 按平均耗时计算原始百分比与ratio加权占比，分别列示，再对加权占比求和。
8. 输出沿用四层表：按逻辑位置保留顺序，两侧并排对齐；采样次数、绝对平均耗时、形状、百分比和差值可追溯。

验收时核对样本数与周期数、四层逻辑行数、通信/边界排除清单、`Σ(mean_g)=T_mean`、`Σ(q_g)=100%`、两阈值组完整覆盖、ratio均值及最大值计算顺序、原始位置没有被同名去重、公式可重算，以及所有展示值都在最后舍入。
