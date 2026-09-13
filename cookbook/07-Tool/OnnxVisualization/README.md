# OnnxVisualization

+ A tool to fold the repeated sub-graphs of a flat ONNX into shared local functions, making the model more readable in Netron.

## Usage

| 目录                                                    | 内容                                                                            |
| ------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [`00-ModelZoo/`](00-ModelZoo/README.md)                 | **测试样例生成**：28 个 ONNX 样例 + 每个的期望结果，全项目模型的唯一真相源      |
| [`01-BasicUsage/`](01-BasicUsage/README.md)             | 基本用法 demo（严格度 / 嵌套层次 / 搜索路径 / 1.5GB 大模型）+ 268 项回归        |
| [`02-ParallelTopology/`](02-ParallelTopology/README.md) | 用在并联 + 跨连拓扑上（5 个 transformer 的分叉/汇合/跨连 DAG）                  |
| [`03-ParallelRepeat/`](03-ParallelRepeat/README.md)     | 随机压力测试：召回率的量化评估                                                  |
| [`04-LoopBackend/`](04-LoopBackend/README.md)           | `--backend loop` 的代价与收益基准（flat / function / loop 三者的 TRT 全面对比） |
| [`90-Research/`](90-Research/)                          | 做这个工具之前的四项调研，见第二部分。结论已并入本文，脚本保留可重跑            |
| [`standalone/`](standalone/README.md)                   | **免安装版**：拷走这一个目录就能处理自己的 ONNX，不需要 `tensorrt_cookbook`     |

## Switches

| Name            | Default value | Instruction                                                                                                                                                                                                  |
| --------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--max-level N` | 1             | 嵌套层数。1 = 不嵌套；2 会把「重复出现的块的组合」再包一层                                                                                                                                                   |
| `--method`      | `auto`        | 搜索路径。`serial` = 只用 1-D 降维；`auto` = 1-D 与图空间生长在同一贪心循环里竞争                                                                                                                            |
| `--backend`     | `function`    | 折叠后端。`function` = local function；`loop` = ONNX Loop                                                                                                                                                    |
| `--no-subgraph` | （默认开）    | 关闭 `Loop`/`If` body 内部的挖掘                                                                                                                                                                             |
| `--beam N`      | 1             | 模式选择的束宽。1 = greedy search；实测只值 +0.5pp 却要 1.7 倍耗                                                                                                                                             |
| `--strictness`  | L1            | 节点标签的严格程度：`L0` 只看 op_type，`L1` +属性，`L2` +常量输入，`L3` +激活 dtype/rank。**别用 `L0`**：属性不进标签，`Transpose.perm` / `Concat.axis` 不同的块会被静默合并，checker 和 TensorRT 都发现不了 |
| `--tolerance`   | 按后端        | `function` 用 0（要求逐比特），`loop` 用相对 `1e-5`                                                                                                                                                          |

## 两种 backend 对下游 TRT 的影响

+ `--backend loop`
  + （使用 Onnx 的 `GraphProto` 属性）会将重复的子图映射成`ILoopLayer`
  + 循环内各自的权重在运行期用 Gather 来抓，前提是各次重复拓扑同构和非权重属性完全一致
  + 减少常量折叠 / 无专用 kernel / 无跨迭代融合
  + 缩小 TensorRT 网络规模与 build 时间，不省显存，增加推理延迟
  + 12 层 transformer：TRT 层数 858 → 115，build 时间 16.6s → 8.9s；engine 体积 9.6MB → 9.3MB，延迟 0.412ms → 0.582ms
  + 1.5 GB 真实模型（GPT-2 类 24 层解码器，batch 1 x seq 64）：TRT 层数 5899 → 503，build 时间 22.2s → 13.5s，engine 体积 1551MB → 1550MB，延迟 延迟 2.32ms → 5.27ms
+ `--backend function`
  + （使用 Onnx 的 `ModelProto.functions`）会把 local function 内联回去
  + 既无收益也无无代价，它只对“让人看懂 ONNX”有意义
  + 层数与 flat 逐层相同，延迟、engine 全都一样
+ 坑：
  + trip count 为编译期常量的 `for` 循环会被 torch 导出器直接展开，`torch.jit.script` 也救不了。必须让 trip count 来自运行期张量
  + ONNX `Loop` 的 trip count 输入在 TRT 里是 shape input tensor，要用 `profile.set_shape_input()` + `context.set_tensor_address()`（`TRTWrapperShapeInput`）

## 现成轮子？

编译器里这件事叫 **outlining**（inlining 的逆操作）。

1. **没有现成包能做「自动发现重复子图 → 提取为子模块」**。现有工具要么只做**反方向**
   （`onnx.inliner`、`onnx_ir.InlinePass`），要么只提供**后一半**（人写模式 → 替换）。
2. 最接近的是 **`onnxscript.rewriter.RewriteRule(..., as_function=True)`**，实测能把 25 节点收成
   5 节点 + 4 个 function，数值逐比特一致、TRT 也能 parse。但有 3 个硬限制：
   (a) `target_pattern` 必须手写；(b) replacement 只能是**单个节点**，否则 ValueError；
   (c) **每个 match 生成一个独立 `FunctionProto`**（用 `overload` 区分），不合并成共享 body。
   —— 第 (c) 条正是后来自己写改写层（P5）的主因。
3. `onnx_ir` 的 `CommonSubexpressionEliminationPass` 只合并「输入完全相同」的重复计算，
   与「输入不同、结构相同」不是一回事。
4. ONNX 官方 RFC [onnx#7301](https://github.com/onnx/onnx/issues/7301) 只讨论**表示**，
   明确不涉及自动发现，且尚未实现。
5. 可借用的积木：`networkx` 的 `DiGraphMatcher`/`vf2pp`（同构验证）、
   `onnx_ir` 的 IR/pass 框架。`gspan-mining` 不能用 —— 它不支持 ONNX 的**有序输入**。
6. 相邻领域的现成思路：LLVM `MachineOutliner`、LLVM `-loop-reroll`、
   SUBDUE（MDL 驱动的子结构发现 + 替换）。

## 子任务 3：通用 FSM 不可行，降维路线很好走

被测对象：`nn.TransformerEncoder(6 层, d_model=64, nhead=4)` 导出的 ONNX。

**三条建模前提**（写错任何一条算法直接失效）：

- 权重 / initializer **不能**当图节点（同构块之间权重必然不同）；
- ONNX 算子输入**有序**，边必须带 `slot`，同构判定必须 edge-labeled；
- 节点标签 = `(op_type, 全部 attribute)`，属性不同即不同模式。

**为什么通用路线不可行**：

- 搜索空间：slim 后 246 节点，连通子图数按 ~1.85x/节点 增长（ONNX 图极稀疏，out-degree 几乎全是 1）。
  一个 encoder layer 是 **41 节点**，`246 × 1.85^40 ≈ 2.6e12` —— 暴力枚举永远到不了。
- 目标函数应该是 **MDL 压缩收益 + 实例非重叠**，不是「出现次数 ≥ 2」
  （否则一个 41 节点块的**所有**连通子模式都「出现 6 次」）。
- **反直觉：加大 WL 哈希半径反而会破坏重复性**。半径大到碰图边界后，
  第 1 层（挨着输入）和第 6 层不再相似；24 轮后 246 个节点全成单例（图直径仅 26）。
  WL 只能做**小半径播种**。

**降维路线**：串行重复块在拓扑序里就是**周期性子串**。实测周期 **41**、重复 **6** 次、
位置 `[0,41,82,123,164,205]`、覆盖率 **246/246 = 100%**，VF2（带 label+slot）验证 6 个实例
**全部同构**。O(n) 后缀自动机即可，不是 NP 难问题。

工作量估计：降维路线 ~1 人周（风险低）；通用路线 ~1~2 人月（风险高）。**先做降维路线。**

### 拓扑序交错问题

拓扑序不唯一，选错了会出大事。`topological_order_experiment.py` 的实测：

| 结构                    | 任意拓扑序                             | recency 拓扑序                     |
| ----------------------- | -------------------------------------- | ---------------------------------- |
| 串行块 + 独立并行链     | `ADBDCDADBDCD…` → **完全漏检**         | `ABCABCABCABC DDDDDD` → 正确       |
| 块内部有 Q/K/V 并行分支 | 正确                                   | 正确（块内并行无害）               |
| 双塔各自有重复块        | `AEBFCG…` → **报出语义无意义的假模式** | `ABCABCABCABC EFGEFGEFGEFG` → 正确 |

- 「块内部的并行」无害；「块之间的并行」致命，且不只漏检，**还会产假模式**。
- 采用 **recency 拓扑序**：优先发射「前驱完成时间最晚」的就绪节点，即先走完当前分支。
- 理论依据：一组不相交节点集能在某个拓扑序里同时连续 ⟺ 各自收缩成点后仍是 DAG（凸性）。
  recency 只是启发式而非保证，所以**图空间验证（凸性 + VF2 + 接口一致性）不可省**。

### 两个绕不开的事实

- **`FunctionProto` 没有 `initializer` 字段**（实测 DESCRIPTOR），function 也不是闭包
  → **权重必须作为 function 输入**，调用点传入各实例自己的 initializer 名。
  transformer 的 `Block0` 因此是 35 入（1 激活 + 34 权重）/ 1 出。
- **Netron 支持 ONNX local function**（`onnx.js` 里有 `context.functions` / `callers` / `uses`），
  已在浏览器里确认渲染良好。已知限制：function body 内的 value_info（shape）不显示
  （lutzroeder/netron#1447）。

## API 选型：onnx_graphsurgeon

主力 **`onnx_graphsurgeon`**，逃生舱 `onnx.helper` + 直接操作 proto。依据都是实测：

1. 同一个 outlining 任务，gs **51 行 vs 原生 73 行**。多出的 22 行全是生产者索引、
   重命名簿记、手写拓扑排序、清死节点 —— 恰好是最容易出 bug 的部分。
2. **`gs.Function` 是一等公民**（继承自 `Graph`），import→export 往返无损，checker 通过。
3. **大模型无额外开销**：1.52 GB / 6119 节点，`gs.import_onnx` 0.33 s、峰值内存与裸
   `onnx.load` 相同（lazy values，不 materialize 权重）—— 这原本是最大顾虑，已排除。
4. gs 是 TensorRT 官方工具，cookbook 已有 `07-Tool/OnnxGraphSurgeon`。

两种实现输出**逐字节等价**，ORT max diff = 0.0。
不选 `onnx_ir` 作主力的理由是非技术性的（0.2.1 API 还在动、不是 TRT 生态），
技术上它其实最现代。

**降险设计**：挖掘算法（P2/P3/P4）只跑在自己的只读 `GraphIR` 上，
API 只出现在 P0/P1 导入与 P5 改写两个薄层，换 API 时挖掘算法一行不改。

## 实现：M1 与关键取舍

流水线：P0 常量折叠 → P1 只读 `GraphIR` → P2a recency 规范拓扑序 → P2b 最长重复子串 + MDL 打分
→ P3 凸性/对齐/接口一致性验证 → P4 选择 → P5 `gs.Function` + 调用节点 → P6 校验与报告。

1. **一个模式只发一份 `FunctionProto`** —— onnxscript.rewriter 做不到，这是自己写 P5 的主因。
2. **对齐检查用位置映射而非 VF2 搜索** —— 构造 body 用的就是位置映射，
   对不上的候选即使同构也没用。大模型上挡掉 5 个 `alignment_mismatch`。
3. **实例内部张量被外部消费 → 提升为额外 function 输出**，不是拒绝。
   大模型的 `Block0` 就有 4 个输出。接口不一致才作废（大模型上 14 次 `interface_mismatch`）。
4. **性能**：第一版对每个哈希桶都物化真实标签元组，大模型 2m45s；
   改成「只对最优桶物化并核对」后 **43 s**，结果完全一致。
5. **已知取舍（非静默截断）**：每个长度只取收益最高的一个桶；外层屏蔽后重跑会捡回其他模式。
6. `report.json` 的 `patterns[*].instances` 存**原始节点名**（如 `/layers.0/self_attn/Transpose`），
   可从 Netron 里的 `Block0` 查回原图。

## 逐项增强

### 嵌套层次 `max_level`

整个 P1~P5 按层重跑：折出来的调用节点在下一层就是普通节点。
报告新增 `patterns[*].level` 与 `patterns[*].n_original_node`（穿透嵌套解析后的原始节点数）；
`coverage` 按**原始节点**去重计算，嵌套不会把它算超过 100%。

| 模型                               | max_level=1         | max_level=2                                               | max_level=3  |
| ---------------------------------- | ------------------- | --------------------------------------------------------- | ------------ |
| 两级 transformer（`(3层+tanh)×2`） | 主图 8 节点，1 fn   | **主图 2 节点**，2 fn（Block1 每实例覆盖 124 个原始节点） | 同 2（收敛） |
| 6 层纯链 transformer               | 主图 6 节点，1 fn   | 同 1（**正确地什么都不多找**）                            | 同 1         |
| `model-large.onnx`                 | 主图 110 节点，3 fn | **主图 88 节点**，4 fn                                    | —            |

**一条刻意加的限制：同质串不做嵌套。** P4 是收益驱动的，总先抓最大的块，
所以第二层只在「内层块比外层组更频繁」时才有东西可抓。反过来，6 个相同块的纯链
在 level 1 会被切成「2 组 3 个」—— 这是**任意切分**，会让读者误以为存在并不存在的结构。
因此 level ≥ 1 的候选若 body 是同一算子的均匀串则拒绝，计入 `rejected.homogeneous_run`
（非静默丢弃），且**不消耗其位置**，搜索会继续找别的模式。level 0 不受此限制。

### 图空间搜索 `method`

**先被实测推翻了一个假设。** `DESIGN.md` 原判断「1-D 降维看不见并行重复」**不成立**：
手写的 fan-out / 共享 hub / 共享累加链 / torch 导出的 6-expert MoE，1-D **全都能找到**。
理论上也说得通 —— 块能被 outline 的前提是**凸**，而一组不相交的凸集收缩后仍是 DAG，
所以**存在**某个拓扑序让它们全部连续。1-D 唯一的弱点是**挑错了拓扑序**。

于是改成量化测量（随机 DAG + 植入 K 份随机块，200 例）。**真正的失败模式**是：
一个外来节点落进块中间 → 块在拓扑序里不再连续 → **被拆成两个模式**或**丢掉边界节点**。
「块整个没找到」只占 6/200。

实现要点（与原设计的差异）：

- **WL 播种基本没用** —— WL 颜色会把**块外上下文**编码进去，同一块的两个实例挂在不同张量上
  颜色就不同、分不到一个桶。（`DESIGN.md` 4.2.3 自己写过要小心，实现时还是踩了。）
- 真正有效的是**齐步生长**，种子换成 1-D 已对齐好的实例组（哪怕是截断/拆开的）。
- **边的匹配键必须用节点原始标签，绝不能用 WL 颜色。**
- **两条路径必须在同一个贪心循环里按 MDL 收益竞争。** 分先后跑效果为零
  （1-D 会先把截断的块提交掉，图空间没得改）—— 这是第一版的写法。

|                                           |       `serial` |     `auto`（默认） |
| ----------------------------------------- | -------------: | -----------------: |
| 每个实例都精确召回（200 随机模型）        |          56.0% |          **86.5%** |
| `model-large.onnx` 主图节点 / function 数 |        110 / 3 |         **45 / 1** |
| `model-large.onnx` 覆盖率 / 耗时          | 97.2% / 46.1 s | **99.2% / 41.8 s** |

transformer 与并联拓扑模型两种 method 结果完全相同（1-D 本来就对）。

顺带修掉一个缺陷：压力测试首轮报 14 个「数值不一致」、`max_abs_diff = nan`。
不是改图错了 —— 模型本身会产生 NaN（`Sqrt` 负数、`Exp` 溢出），而 `nan != nan`。
校验改用 `np.array_equal(..., equal_nan=True)`，修好后 200 例 **0 数值错误**。

### `Loop` 折叠后端

12 层 TransformerEncoder（d_model=128，**B200**，2026-09-06 重测）：

```
            ONNXnode  function  TRTlayer  build(s)  engine(MB)  latency(ms)
flat             492         0       858     16.61         9.6        0.412
function          12         1       858     12.81         9.6        0.471
loop               1         0       115      8.86         9.3        0.582
```

`function` 版与 `flat` **逐层相同**（858 vs 858），印证 parser 会内联。

`Loop` 的表达能力约束（`analyse()` 给出精确拒绝理由，**从不静默**）：
实例必须构成**链**；中间输出只能被下一迭代消费；其余输入必须逐迭代相同
（引用外层作用域，子图可以、function 不行）或能堆叠成 `[K,...]` 的 initializer。

**链拆分**：一个模式可含多条链（`(3层+tanh)×2` → 2 条各 3 迭代），每条各折一个 Loop。
**全有或全无**：只要有实例落单成长度 1 的链，整个模式退回 function。

适用面：✅ 6/12 层 transformer 链、两级 `(3层+tanh)×2`、双塔、4 个 MLP 块；
❌ 5 分支 fan-out（5 条长度 1 的链）、`02-ParallelTopology` 分叉模型 —— 全部**退回 function 并给出理由**。

**数值：`function` 逐比特，`loop` 不是（也不该要求）。** `loop` 约 3.6e-07（~3 个 fp32 ULP），
跨输入稳定。权重去常量化 → runtime 挑不同 kernel → 累加顺序变，是舍入不是逻辑错误
（逻辑错误给 O(1) 偏差，见下面抓到的 6.0 和 1.298）。

> **基准脚本里的一个坑**：`nn.TransformerEncoder(layer, N)` 用 `deepcopy` 克隆同一层，
> **N 层权重完全相同**，TRT 会在 engine 里去重（1.1MB 而非 9.5MB），engine 体积对比失去意义。
> 第一版基准因此得出「Loop 让 engine 大 8 倍」的错误结论，已改为重新随机化每层权重。

### `Loop` 后端多输出支持

初版只支持一个循环变量、一个输出，`model-large.onnx` 因此被拒。现在四种 `Loop` 机制全用上：

| ONNX `Loop` 机制              | 用来表达                                                                  |
| ----------------------------- | ------------------------------------------------------------------------- |
| **loop-carried 变量**         | 交给下一迭代的每个值。**可以有多个**                                      |
| **scan output**               | 交给外部的每个值，堆成 `[K,...]`，再 `Gather(scan, k)` 把每片发回原消费者 |
| **读外层作用域**              | 逐迭代相同的输入                                                          |
| **`Gather(W_stacked, iter)`** | 逐迭代不同的 initializer                                                  |

一个值可**同时**是 loop-carried 和 scan output，此时 body 里用 `Identity` 复制成两个输出
（ONNX 按**位置**而非名字识别 body 输出）。

```
function : main graph 45 nodes, TRT layers=5899, ORT rel=0.0
loop     : main graph 70 nodes, TRT layers= 503, ORT rel=4.5e-06
           24 iterations, 2 loop variable(s), 2 scan output(s), 48 slice nodes, 12 stacked initializers
```

**TRT layer 5899 → 503（11.7x）**。注意主图节点数**反而从 45 涨到 70** ——
scan output 要给每个迭代的每个消费者补一个 `Gather(scan, k)`，24×2 = 48 个切片节点。
主图节点变多、TRT 网络大幅变小，这两件事不矛盾。

两个关键放宽（都是实测逼出来的）：**多个 loop-carried 槽承载同一个值是允许的**
（transformer 残差把块输入同时喂给 attention 和后面的 `Add`，是两个槽、一个循环变量）；
**多个 loop-carried 变量是允许的**（大模型需要 2 个）。

校验改进：相对误差改成**逐输出**计算，否则一个既有大输出又有小输出的模型
会把小输出上的任意大误差藏起来。误差随迭代深度累积：12 层 ~3.6e-07，24 迭代的大模型 ~4.5e-06。

### 召回率 86.5% → 95.5%

原计划是「把 MDL 贪心换成带回溯的选择」。**先把失败按类型拆开，发现瓶颈根本不在选择顺序**：

```
failure breakdown: exact 173, truncated_only 11, missing_instances_only 15, both 1
```

主导的是**「少了一个实例」**，典型样本 `planted 6n x5 → got (6,4)` —— 块形状完全正确，就是漏了一个。

| 改进                                                             |  精确召回 | 平均节点召回 |
| ---------------------------------------------------------------- | --------: | -----------: |
| 起点                                                             |     86.5% |        94.6% |
| + 实例补全 `find_more_instances`（模式定下来后回图里做结构匹配） |     88.5% |        95.1% |
| + **边消歧**（生长时 + 实例匹配时）                              |     93.0% |        96.1% |
| + **WL 半径 3 → 1**                                              | **95.5%** |    **98.6%** |
| + beam=4（未设为默认）                                           |     96.0% |        98.7% |

**贡献最大的是边消歧（+4.5pp）。** 原来的边匹配键 `(方向, out_slot, in_slot, 邻居标签)`
要求唯一匹配，但一个块经常把同一个值喂给两个**同类型**的节点：

```
edges of B0_0: [(out,0,0,'B0_1'), (out,0,0,'B0_2'), (out,0,1,'B0_4')]
                └──── 同一个 key，两个不同邻居 ────┘
```

遇到歧义就跳过 → **三分之一的随机块把侧分支丢在外面**。
改用邻居**自身的边集签名**配对同 key 的多条边；配错也不会出错，`verify` 会挡掉。

**WL 半径 3 → 1（+2.5pp）** 完全印证了 `DESIGN.md` 自己写过的警告，而实现时默认设成了 3。
半径 0/1/2/3 分别是 95.5% / 95.5% / 93.5% / 93.0%。

**beam search 踩的坑**：第一版 beam **越宽结果越差**（95.5% → 92.5% → 83.5%）。
逻辑上不可能 —— 贪心的路径本来就在束里。原因是**已无法继续扩展的状态被丢出了束**，
于是更差但还能扩展的状态赢了。回归用例 `beam_*` 断言「束越宽总收益不会变低」。

剩下的 4.5% 部分是**收益相同的另一种切法**（植入 `3n x4` 收益 6，找到 `4n x3` 收益也是 6），
不算真错 —— 植入答案并非唯一最优。

### 控制流子图挖掘

`Loop` / `If` 的 body 和主图一样会有重复块，而且**主图可能整个只有一个 `Loop` 节点** ——
只看顶层什么都找不到：

```
subgraph=False: main graph 2 node(s), 0 function(s) [nothing],        sub-graphs seen 0
subgraph=True : main graph 2 node(s), 1 function(s) [Sub0Block0=3x4], sub-graphs seen 1 (13 nodes)
body of /Loop now reads: [Identity, Sub0Block0, Sub0Block0, Sub0Block0, Sub0Block0]
```

- **`FunctionProto` 属于 model 而不是某个 graph**，所以 body 可以像主图一样调用它 ——
  函数注册在**根图**上，调用节点插在 body 里。
- 子图列表在**主图折叠之后**再取一次，并与折叠前的集合求交：折叠既会**新建**子图（Loop 后端），
  也可能让原有子图被吸收进 function 而失效。
- 顺带修：`coverage` 的分母原来只算主图节点，主图只有一个 `Loop` 时会报出 **1200%**。

> **绕过了 onnx_graphsurgeon 0.6.1 的一个 bug**：图里**同时**有 local function 和子图时，
> `gs.Graph.toposort()` 会**无限递归**（内层 `get_used_funcs` 迭代的是 `self.subgraphs()`
> 而不是传进来的 node 列表），而「在子图里 outline」恰好造出这个组合。
> `outliner.sort_nodes_in_place()` 自己做拓扑排序，并且把**子图从外层作用域读取的张量**
> 也算作依赖 —— 那不在节点自己的输入表里，漏掉会把节点排到它的生产者前面。

## 抓到的正确性 bug

都是**`onnx.checker` 通过但数值错误**的静默错误，只有 onnxruntime 逐元素比对能发现。
它们只在关掉预处理时暴露（onnxslim 平时会把 `Constant`/`Identity` 折掉）。

| bug                                             | 根因                                                                                                                                               |        无修复时 ORT diff | 回归用例                              |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | -----------------------: | ------------------------------------- |
| `Constant` 属性只按 dtype/shape 比较            | 属性值是**烤进 function body** 的，不像 initializer 输入按调用点传；只比 dtype/shape 会把不同常量判为同一标签                                      |                  **6.0** | `constant_attribute`                  |
| function body 输入按张量身份 `id()` 重映射      | 参考实例中同一张量喂多个槽时后写覆盖先写，其他调用点静默丢参数。必须按 `(offset, slot)` 键                                                         |                **1.298** | `shared_input`                        |
| 输入模型**本身已含 local function** 时断言崩溃  | `assert n_function == len(model.functions)` 把原有 function 也算进分母。`torch.onnx.export(..., export_modules_as_functions=...)` 就会产生这种输入 |    直接 `AssertionError` | `preexisting_function`                |
| `--strictness L0` 合并属性不同的 shape 敏感算子 | L0 的标签里**不含属性**，于是 `Transpose.perm` / `Concat.axis` 不同的块被判为同一模式。不是回归，是 L0 的定义使然 —— 但没人写明它因此不安全        | **0.114** / 直接跑不起来 | `transpose_perm_L0`、`concat_axis_L0` |

前两个是做并联/跨连拓扑测试时撞出来的。**教训**：`onnx.checker` 通过完全不代表图是对的。
（此前 `verify.py` 注释里断言「输入槽不去重也没关系，因为调用点传真实张量」——
调用点确实没问题，但 **body 侧的映射会塌缩**，那个推理是错的。）

第四个是补 shape 敏感算子样例时发现的，也是**同一个教训的第三次复现**：
L0 下 `onnx.checker` 通过、**TensorRT 也能 parse（12 层）**，
唯独 onnxruntime 报 `max_abs_diff = 0.114`（`Concat` 那个更干脆，形状对不上直接跑不起来）。
结论写进了开关表：**`--strictness L0` 对含属性的算子不安全**。

第三个是补测试样例断言时抓到的：`flat_mlp_as_function` 这个模型在 zoo 里躺了很久，
只是从没人对它做过断言。修法是记下 `n_function_preexisting`，原有 function 原样带过
（其 body 不参与挖掘，是已知限制）。**教训**：一个样例躺在语料库里不等于它被测过。

### 测试样例整理

> 唯一**没有**收编进 zoo 的是 `08-LoopBackend` 自己导出的 12 层 transformer。
> 它不是任何 zoo 模型的副本，而且它**刻意重新随机化每层权重**（见上面 deepcopy 那个坑）。

## 已知缺口

- shape 敏感算子只覆盖了「属性 vs 输入」这一条轴（`Transpose.perm` / `Concat.axis` /
  `Reshape` 的目标形状，见 [`00-ModelZoo`](00-ModelZoo/README.md) 第三节）。
  `Split`、`Gather.axis`、`Squeeze` 的 `axes` 输入、动态 shape 下的 `Reshape(-1)`
  都还没有用例。
- 没有量化模型（QDQ）、动态 shape、fp16/bf16 的样例。
- **主图与子图之间不共享 `FunctionProto`**：各自生成，body 相同也不合并。
- 剩余 4.5% 召回率，多为「收益相同的另一种切法」，不算真错。
- 通用图挖掘路线（WL 播种 + 最小 DFS 编码 + 反单调剪枝 + MDL + 非重叠选择）没做，
  ~1~2 人月，等真遇到 1-D 抓不到的并行重复结构再说。
