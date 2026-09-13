# 子任务 3：自研「DAG 模式挖掘 + 替换」的可行性分析

## 待评估的思路（用户原话）

> 找到一种「有向无环图中进行模式匹配」之类的算法，我们从单个算子开始，维护一个子模式集合，
> 遍历寻找图中出现多次的模式，向维护的子模式不断添加新算子，然后替换掉原计算图中的若干节点。

这在文献里叫 **频繁子图挖掘（Frequent Subgraph Mining, FSM）**，具体到「逐步生长候选模式」这个
形式就是 gSpan / FSG / SUBDUE 那一族的 pattern-growth 方法。

## 结论先行

| 维度 | 判断 |
| --- | --- |
| **通用形式**（任意 DAG、任意重复模式）**照原样实现** | ❌ **不可行**，搜索空间指数爆炸，且用户描述里漏掉了两个必须的机制 |
| **加上正确剪枝的形式**（WL 播种 + 反单调剪枝 + MDL 目标） | 🟡 **可行但工程量大**，是一个「小型编译器 pass」级别的项目 |
| **针对实际场景的降维形式**（串行重复块 → 一维重复子串） | ✅ **强烈推荐**，实测在真实 transformer 上 O(n) 就能 100% 找出全部 6 层 |

下面所有数字都来自 `graph_statistics.py`（日志 `log-graph_statistics.py.log`），
被测对象是 `nn.TransformerEncoder(6 层, d_model=64, nhead=4)` 导出的 ONNX。

---

## 一、先把「图」定义清楚：三个必须先想明白的建模问题

### 1.1 权重不是节点

`Linear` 的权重、`Reshape` 的 shape 常量，在两个「同构块」之间**取值必然不同**。
如果把 initializer 当成图节点，任何两层都不同构，算法直接失效。
所以图里只放 **算子节点**，initializer 是模式的「叶子/参数」，正是最后要堆叠成 `[N, ...]` 的东西
（子任务 1 的 `loop_stacked_weight` 已验证这条路在 TRT 上跑得通）。

### 1.2 ONNX 的输入是**有序**的，不能当无标签图处理

`Sub(a, b) != Sub(b, a)`，`Concat` 的输入顺序决定语义。
所以边必须带 `slot` 属性，同构判定必须是 **edge-labeled** 的。
**这一条直接排除了 `gspan-mining` 这类现成库**（它们面向无序边的通用图数据库）。

### 1.3 属性必须进节点标签，但要分级

`Transpose(perm=[0,2,1,3])` 和 `Transpose(perm=[0,1,3,2])` 不能算同一个模式（折叠成 `Loop` 后
body 只有一份，属性无法逐迭代变化）。但 `Reshape` 的目标 shape 是 initializer 输入，可以不同。
所以节点标签 = `(op_type, 所有 attribute)`，而 initializer 输入允许不同 —— 这正是
`graph_statistics.py::onnx_to_networkx` 的做法。

> 子任务 1 结论 9 已经给出硬约束：折叠成 `Loop` 要求各次重复**拓扑同构 + 非权重属性完全一致**。
> 这条就是「等价」的判据。

---

## 二、原思路的两个致命缺口

### 2.1 缺口一：搜索空间爆炸

「从单个算子开始，不断添加新算子」这个动作，走的就是「所有连通子图」的空间。实测（已 slim 的图，246 节点）：

| k（模式大小） | 连通子图数 |
| ---: | ---: |
| 1 | 246 |
| 2 | 268 |
| 3 | 342 |
| 4 | 494 |
| 5 | 776 |
| 6 | 1278 |
| 8 | 3754 |
| 10 | 11982 |
| 11 | 22137 |

好消息：ONNX 图**极其稀疏**（out-degree 分布 `{1: 228, 2: 11, 3: 6, 0: 1}`，几乎是一条链），
增长因子稳定在 **~1.85x/节点**，远好于稠密图的组合爆炸。

坏消息：**一个 encoder layer 是 41 个节点**。`246 × 1.85^40 ≈ 2.6e12` —— 暴力枚举永远到不了。
而我们真正想要的模式恰恰就是那么大。原始未 slim 的图更糟（522 节点、增长因子 ~3.4x）。

**这不是「慢一点」的问题，是「差 12 个数量级」的问题。必须有剪枝。**

### 2.2 缺口二：「出现多次的模式」这个目标函数是错的

即使能枚举，「出现次数 ≥ 2」也会返回**海量无用模式**：一个 41 节点的块被找到时，
它的 2^41 个连通子模式**全都**「出现 6 次」。你会得到一堆互相嵌套、互相重叠的模式，
而不是「6 个 encoder layer」。

正确的目标不是频次，而是**压缩收益**。SUBDUE 用的 MDL（最小描述长度）准则是现成答案：

```
gain(P) = (|P| - 1) × (occurrences(P) - 1) - overhead(P)
          └ 每次替换省下的节点数 ┘   └ 多出来的 function/Loop 本身的开销 ┘
```

再叠加一个**实例互不重叠**的约束（这本身是一个最大权独立集问题，实践中贪心即可）。
用户描述里「寻找图中出现多次的模式」这一步，必须换成「寻找压缩收益最大的、实例互不重叠的模式集合」。

---

## 三、补上缺口后的可行方案

### 3.1 用 WL 哈希做**播种**，而不是做主力

Weisfeiler-Lehman 子图哈希（`networkx.weisfeiler_lehman_subgraph_hashes`）能在 O(n·k) 时间内
给每个节点算出「k 跳邻域的规范标签」，同哈希 = 邻域同构（单向，有极小假阳性率）。
这是把「枚举所有子图」换成「按哈希分桶」的关键。

但实测发现一个**反直觉且很重要的结论：不能靠加大 WL 半径直接找到整个块。**

| WL 迭代轮数 | 等价类数 | 含重复的类数 | 落在重复类里的节点数 | 最大类大小 | 大小恰为 6 的类数 |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 32 | 29 | 243 | [24,18,18,18,18,12] | 20 |
| 2 | 43 | 36 | 239 | [18,18,12,6,6,6] | 26 |
| 3 | 52 | 39 | 233 | [18,6,6,6,6,6] | 27 |
| 5 | 72 | 41 | 215 | [6,6,6,6,6,6] | 22 |
| 8 | 109 | 41 | 178 | [6,6,6,6,6,6] | 12 |
| 12 | 172 | 27 | 101 | [5,5,5,5,5,5] | **0** |
| 16 | 221 | 15 | 40 | [3,3,3,3,3,3] | **0** |
| 24 | 246 | **0** | **0** | [1,1,1,1,1,1] | **0** |

半径一大，节点的邻域就**碰到图的边界**：第 1 层挨着模型输入、第 6 层挨着模型输出，
它们的大半径邻域天然不同，于是 6 个本应等价的实例被拆散，24 轮以后 246 个节点全成了单例。
（图的直径只有 26，所以 24 轮已经覆盖全图。）

**设计含义**：WL 只能用来产生**小半径的候选锚点**（比如 3~5 轮，此时还有 22 个大小恰为 6 的类），
之后的模式生长必须显式地把**模式边界当作通配符**处理 —— 匹配的是「诱导子图的形状」，
而不是「节点在全图中的上下文」。这是两种完全不同的匹配语义，实现时极易混淆。

### 3.2 反单调剪枝

FSM 的标准武器：若模式 P 的支持度 < 阈值，则 P 的任何超图支持度也 < 阈值，整枝剪掉。
配合 gSpan 的 **DFS 最小编码（minimum DFS code）** 做规范形式，可以保证每个模式只被生成一次，
避免同一个模式沿不同生长顺序被重复枚举 —— 这是 pattern-growth 类算法真正的核心，
也是「自己写一个」时最容易写错、最容易写慢的部分。

### 3.3 实例验证

分桶只是必要条件，最终要用 **edge-labeled 子图同构**确认。`networkx` 的
`DiGraphMatcher(node_match=..., edge_match=...)`（VF2）可以直接用；实例只有几十个节点，代价可忽略。

### 3.4 替换

见子任务 2：`onnxscript.rewriter` 有 `as_function=True`，或者直接
`onnx.helper.make_function` 手工拼。这一步**工作量最小**，不是瓶颈。
折成 `Loop` 时额外要做的是把各实例的 initializer 按迭代序 `np.stack` 成 `[N, ...]`，
并在 body 里插 `Gather(W, iter)` —— 子任务 1 的 `case_loop_stacked_weight` 就是模板。

---

## 四、更好的路线：针对实际场景降维

上面那套是「通用 DAG FSM」的正确做法，但**我们的实际场景比通用问题简单得多**：
transformer / resnet / U-Net 里的重复块是**串行堆叠**的。

把 DAG 按拓扑序拍平成一个算子类型的字符串，串行重复块就变成了**周期性子串**，
于是问题从「NP 难的子图同构挖掘」塌缩成「最长重复子串」—— 后缀自动机 O(n)。

实测（`graph_statistics.py` 第 3 节）：

```
topological op-type sequence length = 246
longest sub-sequence repeated >= 6 times without overlap: length = 41
occurrence positions = [0, 41, 82, 123, 164, 205]
pattern head = ('Transpose','Reshape','Gemm','Reshape','Unsqueeze','Transpose','Squeeze','Gather','Gather','Gather')
coverage = 246 / 246 nodes
induced sub-graphs isomorphic to the first one (label + input slot matched): [True, True, True, True, True]
```

* 周期 **41**，重复 **6** 次，位置完全均匀 —— 精确对应 6 个 encoder layer。
* **覆盖率 246/246 = 100%**，没有一个节点漏掉。
* 6 个实例的诱导子图，用 VF2 带 `label` + `slot` 匹配验证，**全部同构**。
* 整个过程（含这里用的朴素 O(n²) 实现）不到一秒。

**这条路线的注意事项**：

1. **拓扑序不唯一**。这里之所以整齐，是因为图近乎一条链。有分支时必须用**规范拓扑序**
   （比如按小半径 WL 哈希 + 原始节点序做确定性 tie-break），否则同一个块的两次出现可能被排成不同顺序。
2. **它只能找串行重复**。并行重复（比如 multi-head 被展开成 N 条并列分支、MoE 的 N 个 expert）
   在拓扑序里是交错的，抓不到。这类场景要退回 3.1~3.3 的通用路线，或者单独做一个「兄弟分支同构」检测。
3. **必须先做常量折叠**。实测 slim 前 522 节点、21 种算子，slim 后 246 节点、12 种算子（2.1x）。
   `Constant`(129) / `Identity`(63) / `Shape` 这些 shape 计算的管道是纯噪声，
   不折掉的话周期性会被打乱。用 `onnxslim` 或 `polygraphy surgeon sanitize --fold-constants` 即可。

---

## 五、成本 / 收益判断

### 收益（来自子任务 1 的实测）

| 折叠目标 | ONNX 节点 | TRT layer | build 时间 | engine 体积 | 推理延迟 |
| --- | --- | --- | --- | --- | --- |
| local function | ↓ N 倍 | **不变**（被内联） | 不变 | 不变 | 不变 |
| `Loop` + 堆叠权重 | ↓ 到 O(1) | ↓ 到 O(1) | ↓（N=256 时 12.6s → 6.1s） | **不变** | **↑ ~2x** |

所以：

* 想要**可读性 / 文件体积 / 上下游工具处理速度** → local function，零风险，值得做。
* 想要**缩短 TRT build 时间** → `Loop`，但要付约 2 倍延迟，只在特定场景划算。
* **省显存是做不到的**，权重量不变。

### 成本

| 路线 | 工作量估计 | 风险 |
| --- | --- | --- |
| 降维路线（串行重复 + 后缀自动机 + VF2 验证 + `make_function` 替换） | **~1 人周** | 低。规范拓扑序是唯一需要小心的地方 |
| 通用路线（WL 播种 + 最小 DFS 编码 + 反单调剪枝 + MDL + 非重叠选择） | **~1~2 人月** | 高。规范形式和边界语义是经典的坑，性能调优另算 |

### 建议

**先做降维路线**，它能覆盖 transformer / CNN backbone 这些实际会遇到的绝大多数「节点爆炸」模型，
并且能立刻拿到可验证的收益。把它做成两个可选输出后端（local function / `Loop`），
再用「重写后模型 vs 原模型的 onnxruntime 逐元素比对」作为强制的正确性闸门。

只有当遇到降维路线抓不到的并行重复结构时，再考虑投入通用路线。

---

## 六、必须配套的正确性保障

自动改图的风险很高，工具必须自带：

1. **数值等价校验**：改写前后用同一批随机输入过 onnxruntime，逐元素比对（子任务 1、2 的脚本都这么做的）。
2. **`onnx.checker` + shape inference** 必须通过。
3. **TRT 闸门**：改写后的模型必须还能被 `trt.OnnxParser` 解析并 build 成功，
   并且和原模型的 engine 输出一致。
4. **可回退**：任何一步失败就退回原图，绝不输出半成品。
5. **子图内不能有外部引用**：模式实例内部的中间张量如果被实例外的节点消费，
   就不能整体提取（提取后那个张量在函数体内不可见）。这是 outlining 的经典约束，
   实现时必须显式检查，否则会静默产生错图。

## 参考

- [gSpan: Graph-Based Substructure Pattern Mining](https://dl.acm.org/doi/10.5555/844380.844811)
- [Automated Design Space Exploration of CGRA Processing Element Architectures using Frequent Subgraph Analysis](https://arxiv.org/pdf/2104.14155)
- [FLEXIS: FLEXible Frequent Subgraph Mining using Maximal Independent Sets](https://arxiv.org/pdf/2404.01585)
- [GitGraph — Architecture Search Space Creation through Frequent Computational Subgraph Mining](https://arxiv.org/pdf/1801.05159)
- [networkx — Weisfeiler-Lehman graph hashing](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.graph_hashing.weisfeiler_lehman_subgraph_hashes.html)
- [networkx — VF2 isomorphism](https://networkx.org/documentation/stable/reference/algorithms/isomorphism.vf2.html)
