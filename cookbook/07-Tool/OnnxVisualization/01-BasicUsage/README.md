# ONNX Outliner —— M1（可视化导向）

把展开的 ONNX 里重复出现的子图**自动**找出来，折叠成共享的 local function，
让模型在 Netron 里能一眼看懂。

> **M1 的定位**：只服务可视化，不承诺保语义 —— 喂给 TensorRT 的可以仍是原始 ONNX。
> 不过实测下来**所有测试用例都做到了 onnxruntime 逐比特相同、且 TensorRT 也能正常 parse**，
> 保语义这件事比 M3 计划的来得早。见下面的实测结果。

## 用法

```bash
# 命令行
python3 -m tensorrt_cookbook.onnx_outliner model.onnx -o model-outlined.onnx --report report.json

# 常用开关
python3 -m tensorrt_cookbook.onnx_outliner model.onnx --min-repeat 3 --min-size 5 --strictness L2 --strict
python3 -m tensorrt_cookbook.onnx_outliner model.onnx --max-level 2      # 允许嵌套一层
python3 -m tensorrt_cookbook.onnx_outliner model.onnx --method serial    # 只用 1-D 降维（默认 auto = 两条路径竞争）
python3 -m tensorrt_cookbook.onnx_outliner model.onnx --backend loop     # 折成 ONNX Loop（默认 function）
python3 -m tensorrt_cookbook.onnx_outliner model.onnx --no-subgraph      # 不挖 Loop/If body（默认挖）
```

```python
from tensorrt_cookbook import OutlineConfig, outline

report = outline("model.onnx", "model-outlined.onnx", OutlineConfig(strictness="L1", max_level=2))
print(report["patterns"])
```

跑 demo 和回归测试：

```bash
python3 main.py           # transformer / 严格度 / 嵌套层次 / 搜索路径 / 1.5GB 大模型
python3 test_outliner.py  # 268 项回归检查，失败时返回非 0
```

两个脚本用到的**每一个输入模型都来自 [`../00-ModelZoo`](../00-ModelZoo/README.md)**，
本目录不再自己造模型。期望结果也在那边（`model_zoo.EXPECT`），
所以「模型」和「它应该被折成什么样」不可能各改各的。

## 不装 cookbook，直接处理自己的模型

上面的 `main.py` 是 cookbook demo，`import tensorrt_cookbook`。
如果你只想把**自己的 ONNX** 折一下，用 [`../standalone/`](../standalone/README.md)：

```bash
python3 ../standalone/main.py my_model.onnx                       # 输出 my_model-outlined.onnx
python3 ../standalone/main.py my_model.onnx -o out.onnx --report report.json
python3 ../standalone/main.py my_model.onnx --max-level 2 --strictness L2 --strict
python3 ../standalone/main.py --help                              # 开关与 `python3 -m tensorrt_cookbook.onnx_outliner` 一致
```

+ **不需要** `pip install tensorrt_cookbook`，**不需要** `TRT_COOKBOOK_PATH`，也不需要 `tensorrt`、`torch`。
  把 `../standalone/` 整个目录拷到任何地方就能跑。
+ 必需依赖只有 `onnx`、`onnx_graphsurgeon`、`numpy`；
  `onnxslim`（常量折叠）、`onnxruntime`（数值交叉校验）、`tensorrt`（parse 校验）缺席时自动跳过并在报告里注明。
+ `../standalone/onnx_outliner/` 是 `tensorrt_cookbook/onnx_outliner` 的独立副本，
  唯一改动是 `_tensorrt_check` 不再借用 `TRTWrapperV1`。
  `python3 test_standalone.py` 会验证两点：副本没有相对上游漂移，
  以及在一个 `import tensorrt_cookbook` 被禁掉的解释器里 `../standalone/main.py` 端到端跑通。

## 实测结果

### 6 层 TransformerEncoder

```
   522 nodes in the input file
   246 nodes after onnxslim (P0)
     6 nodes in the main graph + 1 local function(s), coverage 100.0%
       Block0:  41 nodes x  6 instances, gain  200, interface 35 in / 1 out
verification: {'onnx_checker': 'pass',
               'onnxruntime': {'status': 'pass', 'max_abs_diff': 0.0},
               'tensorrt': {'status': 'pass', 'n_layer': 389}}
main graph now reads: [Block0, Block0, Block0, Block0, Block0, Block0]
```

主图正好是 6 个 `Block0`，覆盖率 100%，**数值逐比特相同**。

### 真实大模型（`00-Data/model/model-large.onnx`，1.5 GB / 6119 节点）

```
  6119 nodes in the input file
  2661 nodes after onnxslim (P0)
   110 nodes in the main graph + 3 local function(s), coverage 97.2%
       Block0: 110 nodes x 23 instances, gain 2398, interface 87 in / 4 out
       Block1:   5 nodes x  8 instances, gain   28, interface 5 in / 2 out
       Block2:   4 nodes x  4 instances, gain    9, interface 7 in / 1 out
rejected candidates: {'alignment_mismatch': 5, 'interface_mismatch': 14}
verification: {'onnx_checker': 'pass',
               'onnxruntime': {'status': 'pass', 'max_abs_diff': 0.0},
               'tensorrt': {'status': 'pass', 'n_layer': 5899}}
```

**2661 → 110 个主图节点**，一个 110 节点的块重复 23 次被抓了出来。全程 ~43 s。
`rejected` 那一行说明 P3 的图空间验证确实在干活：19 个 1-D 搜索给出的候选被挡掉了。
加上 `--max-level 2` 还能再降到 **88 个节点**（多一个 `Block3`，12 节点 × 2 实例）。

### 回归测试

`test_outliner.py`，**268/268 通过**。分两部分：

**一、Model Zoo 全扫**（每个样例一个用例，共 28 个）。
逐条比对 `model_zoo.EXPECT` 里的期望模式，外加 `onnx.checker`、
onnxruntime 逐比特比对、「一个模式一个 `FunctionProto`」、「每个调用都能解析到函数」四道闸门。
样例清单与各自考察什么见 [`../00-ModelZoo/README.md`](../00-ModelZoo/README.md)。

**二、深度用例**（同样的模型，但要跑不止一次 outline 或要断言报告内部字段）：

| 用例 | 结构 | 期望 | 结果 |
| --- | --- | --- | --- |
| `serial_plus_parallel` | 串行 `A→B→C` ×4 + 独立并行 `D` 链 | `(3,4)` 和 `(3,2)` | ✅ 18 → 6 节点 |
| `internal_branch` | 块内部有 3 条并行分支 | `(5,4)` | ✅ 20 → 4 节点 |
| `two_tower` | 两个独立塔，各有重复块 | `(3,4)` ×2 | ✅ 24 → 8 节点 |
| `transformer_6layer` | 6 层 TransformerEncoder | `(41,6)` | ✅ 246 → 6 节点 |
| `flat_mlp` | 4 个 MLP block | `(4,4)` | ✅ 16 → 4 节点 |
| `flat_mlp_as_function` | PyTorch **已经折好**的模型 | 不折，原有 function 原样带过 | ✅ 新增 |
| `shared_input` | 参考实例共享张量、其他实例不共享 | `(3,4)` 且数值正确 | ✅ 回归 |
| `constant_attribute` | 4 个块持有**不同**的 `Constant` | **不应折叠** | ✅ 回归 |
| `ambiguous_sibling` | 同一个值喂给两个同类型节点 | `(4,4)`（不是 `(3,4)`） | ✅ 边消歧 |
| `transpose_perm_L1` / `concat_axis_L1` | 两种 `Transpose.perm` / `Concat.axis` | 切成**两个**模式，逐比特相同 | ✅ shape 敏感 |
| `transpose_perm_L0` / `concat_axis_L0` | 同上但 `--strictness L0` | **必须错**，且必须是**数值闸门**发现的（checker 与 TRT 都放行） | ✅ 反向断言 |
| `reshape_shape_input` | 四个不同的 `Reshape` 目标形状 | 目标是**输入**不是属性 → **必须合成一个**模式 | ✅ 防过紧 |
| `nested_level1/2/5` | 2 组 × 3 个内层块 | L1 只出 `(3,6)`；L2/L5 出 `(3,6)+(4,2)` | ✅ 嵌套 |
| `transformer_level3` | 6 层纯链 + `max_level=3` | **仍只出 `(41,6)`**（同质串不嵌套） | ✅ |
| `split_block_serial/auto` | 植入的 5 节点块被拓扑序打断 | serial 只出 `(4,2)`；auto 出 `(5,2)` | ✅ path B |
| `loop_transformer` | 6 层链 + `--backend loop` | 1 个 Loop，主图 1 节点，TRT 389→96 层 | ✅ |
| `loop_two_chain` / `loop_two_tower` | 两条链 | 各折一个 Loop | ✅ 链拆分 |
| `loop_fan_out` | 5 分支 fan-out | **退回 function** 并给出理由 | ✅ |
| `loop_scan_output` | 块有 2 个输出（1 carried + 1 外部） | 1 个循环变量 + 1 个 scan output + 4 个切片节点 | ✅ 多输出 |
| `loop_two_carried` | 块把 2 个值交给下一迭代 | **2 个循环变量** | ✅ 多循环变量 |
| `beam_*`（3 个） | beam=1/2/4 的总收益 | **束越宽总收益不会变低** | ✅ 防回归 |
| `subgraph_loop` / `subgraph_if` | 重复块只在 `Loop`/`If` body 里 | body 折成 4 次调用，主图不动 | ✅ 子图挖掘 |
| `subgraph_off` | 同上 + `--no-subgraph` | 什么都不折 | ✅ |

前三个正是 `../90-Research/03-PatternMiningFeasibility/topological_order_experiment.py` 里
「任意拓扑序会漏检 / 会报假模式」的那三个结构，它们是规范拓扑序存在的理由。

**一个顺带修掉的 bug**：输入模型本身就含 local function 时
（`torch.onnx.export(..., export_modules_as_functions=...)` 就会产生），
`assert n_function == len(model.functions)` 会直接崩。
原有函数被原样带过、不参与统计，报告新增 `n_function_preexisting` 记录它们。
由 `flat_mlp_as_function` 这个样例暴露 —— 它进 zoo 已久，但一直没有断言在跑。

## 嵌套层次 `max_level`

`max_level=1`（默认）只折一层：节点 → 块。`max_level=2` 会在折完块之后，
把**重复出现的「块的组合」**再包一层，以此类推。

真实的两级模型（6 层 encoder 排成 `(3 层 + tanh) × 2`，见 `main.py::TwoStageNet`）：

```
max_level=1: main graph   8 nodes, 1 function(s) [L0 Block0=41x6]
max_level=2: main graph   2 nodes, 2 function(s) [L0 Block0=41x6, L1 Block1=4x2]
      Block1 wraps 4 nodes = 124 original nodes per instance
max_level=3: main graph   2 nodes, 2 function(s) [L0 Block0=41x6, L1 Block1=4x2]
```

主图从 8 个节点降到 **2 个**，每个 `Block1` 代表 124 个原始节点。第 3 层没有新东西，自动收敛。

`00-Data/model-large.onnx` 上 `max_level=2` 也确实多抓到一层：
主图 **110 → 88 节点**，多出一个 `Block3`（12 节点 × 2 实例，每实例覆盖 19 个原始节点），
数值仍然逐比特相同。

### 一条刻意加的限制：同质串不做嵌套

P4 的选择是**收益驱动**的，总是先抓最大的那块。所以第二层**只在「内层块比外层组更频繁」时才有东西可抓** ——
比如 6 层排成 2 组 3 层。反过来，一条 6 个相同块的**纯链**，
level 1 会想把它切成「2 组 3 个」，但这是**任意切分**：六个相同的层并没有天然的「三个一组」，
报出来只会让读者误以为存在并不存在的结构。

所以 level ≥ 1 的候选如果 body 是**同一个算子的均匀串**，会被拒绝，
并计入 `rejected.homogeneous_run`（不是静默丢弃）。level 0 不受此限制 ——
6 个连续 `Softplus` 在原图里确实就是一个重复块。

实测：6 层 TransformerEncoder 和并联拓扑那个模型，`--max-level 3` 都**正确地什么都不多找**。

## 控制流子图挖掘 `subgraph`

`Loop` / `If` 的 body 和主图一样会有重复块，而且**主图可能整个只有一个 `Loop` 节点** ——
只看顶层的工具在这种模型上什么都找不到。默认开启（`--no-subgraph` 关闭）。

torch 导出的、body 里含 4 个重复块的 `Loop` 模型：

```
subgraph=False: main graph 2 node(s), 0 function(s) [nothing],           sub-graphs seen 0
subgraph=True : main graph 2 node(s), 1 function(s) [Sub0Block0=3x4],    sub-graphs seen 1 (13 nodes)
body of /Loop now reads: [Identity, Sub0Block0, Sub0Block0, Sub0Block0, Sub0Block0]
```

**`FunctionProto` 属于 model 而不是某个 graph**，所以 body 可以像主图一样调用它 ——
函数注册在根图上，调用节点插在 body 里，`onnx.checker` / ORT / TRT 全部通过。
`--backend loop` 也能用在子图里，结果是 `Loop` 套 `Loop`（实测数值逐比特相同、TRT 可 parse）。

> **绕过了 onnx_graphsurgeon 0.6.1 的一个 bug**：图里同时有 local function 和子图时，
> `gs.Graph.toposort()` 会无限递归（内层 `get_used_funcs` 迭代的是 `self.subgraphs()`
> 而不是传进来的 node 列表）。而「在子图里 outline」恰好造出这个组合。
> `outliner.sort_nodes_in_place()` 自己做拓扑排序，并且把**子图从外层作用域读取的张量**
> 也算作依赖 —— 那不在节点自己的输入表里，漏掉会把节点排到它的生产者前面。

## 模式选择的束宽 `beam`

`--beam N`：在「下一个提交哪个模式」上做束搜索，`beam=1`（默认）就是纯贪心。
实测**只值 +0.5pp，却要 1.7 倍耗时**（大模型 41.7s → 71.9s），真实模型上结果完全相同，
所以默认保持 1。瓶颈不在选择顺序，在候选生成 —— 详见 `../03-ParallelRepeat/README.md`。

## 折叠后端 `backend`

| 值 | 产出 | TRT 行为 |
| --- | --- | --- |
| `function`（默认） | local function | parser **内联回去**，layer 数、engine、延迟全部不变 |
| `loop` | ONNX `Loop` | 映射成真正的 `ILoopLayer`，**body 只实例化一次** |

12 层 TransformerEncoder（d_model=128，**B200**，2026-09-06 重测）：

```
            ONNXnode  function  TRTlayer  build(s)  engine(MB)  latency(ms)
flat             492         0       858     16.61         9.6        0.412
function          12         1       858     12.81         9.6        0.471
loop               1         0       115      8.86         9.3        0.582
```

**TRT layer 858 → 115（7.5x 更少）、build 16.6s → 8.9s**，但
**engine 基本不变（不省显存）、延迟 1.41x**。这是一个明确的取舍，所以默认关闭。

`Loop` 支持**多个循环变量**和**多个 scan output**：交给下一迭代的值成为循环变量，
交给外部的值堆成 `[K,...]` 再用 `Gather(scan, k)` 把每片发回原消费者。
逐迭代相同的输入直接读外层作用域，逐迭代不同的 initializer 堆成 `[K,...]` 后 `Gather(W, iter)`。

仍然苛刻：实例必须构成**链**、输出不能被非相邻的同链迭代消费。
不满足时**退回 function 并在报告里给出精确理由**（`patterns[*].loop_rejected_because`）。
一个模式若含多条链（`(3层+tanh)×2`），会**每条链折一个 Loop**。

`00-Data/model-large.onnx`（1.5GB / 6119 节点）：24 迭代、**2 个循环变量、2 个 scan output**、
48 个切片节点，TRT **5899 → 503 层**。注意主图节点数反而从 45 涨到 70（切片节点），
但 TRT 网络小了 11.7 倍 —— 两件事不矛盾。

数值上 `function` 逐比特相同，`loop` 有约 **3.6e-07 的相对误差**（权重去常量化 → kernel 变了 →
累加顺序变了）。所以校验容差按后端取默认：`function` 要求逐比特，`loop` 用相对 `1e-5`，
`--tolerance` 可覆盖。

详见 [`../04-LoopBackend/README.md`](../04-LoopBackend/README.md)。

## 搜索路径 `method`

| 值 | 含义 |
| --- | --- |
| `serial` | 只用 1-D 降维（规范拓扑序的重复子串） |
| `parallel` | 只用图空间（WL 锚点 + 齐步生长） |
| `auto`（默认） | 两者的候选在**同一个贪心循环里按 MDL 收益竞争** |

1-D 路径真正的弱点**不是**看不见并行重复（实测 fan-out / hub / 累加链它全都能找到），
而是**一个外来节点落进块中间**，块不再是拓扑序里连续的一段，于是被**拆成两段**或**丢掉边界节点**。
图空间路径以 1-D 给出的（哪怕截断的）实例组为**种子**做齐步生长，把剩下的补回来。

随机植入块的压力测试（`../03-ParallelRepeat/stress_test.py`，200 个模型）：

| | `serial` | `auto`（第一轮） | `auto`（现在） |
| --- | ---: | ---: | ---: |
| 每个实例都精确召回 | 56.0% | 86.5% | **95.5%** |
| 平均召回的植入节点比例 | 89.1% | 94.6% | **98.6%** |
| 数值错误 | 0 | 0 | 0 |

从 86.5% 到 95.5% 的四步改进（每步单独量化）见
[`../03-ParallelRepeat/README.md`](../03-ParallelRepeat/README.md)：
**实例补全**（+2.0）、**边消歧**（+4.5，贡献最大）、**WL 半径 3→1**（+2.5）、**beam**（+0.5）。

`model-large.onnx` 上：主图 **110 → 45 节点**、function **3 → 1 个**、覆盖率 **97.2% → 99.2%**，
最大模式从 `110×23` 变成 `110×24`（多找到一个实例），而且**还快了一点**（46.1s → 41.8s）。

详见 [`../03-ParallelRepeat/README.md`](../03-ParallelRepeat/README.md)。

## 严格度 `strictness`

| 档 | 两个节点要一致到什么程度 |
| --- | --- |
| `L0` | `op_type` + `domain` |
| `L1`（默认） | `+ 全部 attribute` |
| `L2` | `+ 常量输入的 dtype/shape` |
| `L3` | `+ 激活张量的 dtype/rank` |

在 TransformerEncoder 上四档结果完全一样 —— 这是诚实的结论：那 6 层结构上确实一模一样。
在异构的 `model-large.onnx` 上就有区别了：**L0/L1/L2 得到 110 节点 + 3 个 function，
L3 得到 122 节点 + 2 个 function**（激活的 dtype/rank 把一部分块区分开了）。

## 流水线与代码结构

```
onnx_outliner/
├── config.py      OutlineConfig
├── preprocess.py  P0  常量折叠（onnxslim），522 → 246 节点
├── graph_ir.py    P1  只读 GraphIR：节点=算子，边=(out_slot,in_slot)，initializer 作叶子
├── ordering.py    P2a recency 规范拓扑序
├── discover.py    P2b 最长重复子串（滚动哈希）+ MDL 收益打分
├── verify.py      P3  凸性 / 对齐 / 接口一致性
├── rewrite.py     P5  gs.Function + 调用节点（一份 body，k 处调用）
├── outliner.py    P4+P6 贪心选择、驱动、校验、报告
└── __main__.py    CLI
```

**P4 的选择策略**：找到收益最高的候选 → 验证 → 提交 → 把已占用位置屏蔽 → 再找，
直到没有正收益。双塔场景就靠这一步先拿一个塔再拿另一个。

**层次化**：整个 P1~P5 会按 `max_level` 重跑。每跑完一层，
折出来的调用节点在下一层就是普通节点，于是「重复的块的组合」被包成外层 function。
报告里 `patterns[*].level` 标明层号，`patterns[*].n_original_node` 是**穿透嵌套解析后**
一个实例覆盖的原始节点数，`instances` 里存的始终是**原始节点名**。

## 关键实现要点

### 1. `FunctionProto` 没有 `initializer` 字段，权重必须当输入

实测 `onnx.FunctionProto.DESCRIPTOR.fields` 里没有 `initializer`，ONNX function 也不是闭包。
所以 `Block0` 的签名是 **35 个输入**（1 个激活 + 34 个权重），调用点传各实例自己的 initializer。
这和 `torch.onnx.export(..., export_modules_as_functions=...)` 的形态一致。

### 2. 一份 body、k 处调用

`onnxscript.rewriter(as_function=True)` 每个 match 发一份独立 `FunctionProto`（靠 `overload` 区分）。
本工具**一个模式只发一份**，这是自己写 P5 的主要原因。

### 3. 对齐检查用「位置映射」而不是 VF2 搜索

候选的各实例都来自规范拓扑序的连续窗口，所以实例 j 的第 t 个节点天然对应实例 0 的第 t 个节点。
我们直接用这个映射逐条比对内部边（含 `out_slot`/`in_slot`），而不是用 VF2 去*搜索*一个映射 ——
因为构造 function body 时用的就是这个位置映射，它对不上的候选就算「同构」也没法用。
对不上的记进 `rejected.alignment_mismatch`。

### 4. 实例内部张量被外部消费 → 提升为额外 function 输出

不是拒绝候选。大模型上 `Block0` 就有 **4 个输出**（不止 1 个），说明这条路径真实存在。
但只有当**所有实例**都在同样的 `(offset, out_slot)` 引出时才成立，否则接口签名不同、整组作废
（记进 `rejected.interface_mismatch`，大模型上发生了 14 次；开嵌套后 18 次）。

### 5. 两个靠数值比对才抓到的正确性 bug（详见 `../02-ParallelTopology/README.md`）

* **`Constant` 属性只按 dtype/shape 比较** —— 属性值是**烤进 function body** 的，
  不像 initializer 输入按调用点传，所以必须按**值**比较。修复前 ORT diff = 6.0。
* **function body 输入按张量身份重映射** —— 参考实例里同一张量喂多个槽时后写覆盖先写，
  其他调用点静默丢参数。必须按 `(offset, slot)` 键。修复前 ORT diff = 1.298。

两个都**默认路径碰不到**（onnxslim 会折掉 `Constant`/`Identity`），且**`onnx.checker` 都通过**。
这就是为什么 onnxruntime 逐元素比对不是可选项。

### 6. 性能：滚动哈希 + 只对最优桶做实证

第一版对每个长度的**每个**哈希桶都去物化真实标签元组做去混淆，代价 O(n·L)/长度，
大模型上跑了 **2m45s**。改成「按哈希计数选出最优桶，只对它物化并核对真实标签」后降到 **43s**，
结果完全一样。61 位哈希碰撞概率可忽略，而且核对这一步保证碰撞只会损失搜索质量、不会产生错误结果。

**已知取舍（不是静默截断）**：每个长度只取收益最高的一个桶。同一长度藏了两个不同重复块时，
本轮只报较优的那个；外层「屏蔽已提交实例后重跑」会在下一轮找到另一个。

## 不做的事（留给 M2 / M3）

* **控制流子图内部**（`Loop`/`If` 的 body）不挖掘，报告里以 `skipped_subgraph_node` 显式列出数量。
  实测 `00-Data/model/model-for.onnx`：0 个模式，不崩，正常输出。
* **`Loop` 的多输出**：目前只支持链上有一个输出的模式。多输出需要 scan output
  再把每次迭代的切片发回各自的消费者，未实现（`model-large.onnx` 就因此退回 function）。

## 报告字段

`report.json` 里 `patterns[*].instances` 存的是**原始节点名**，
这样在 Netron 里看到 `Block0` 可以查回它对应原图的哪些节点 —— 这是可视化能用起来的前提。
