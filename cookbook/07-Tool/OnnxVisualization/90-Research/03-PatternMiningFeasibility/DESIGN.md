# ONNX Outliner 实现设计（第一版）

> 目标：把展开的 ONNX 中重复出现的子图找出来，改造成 local function 形式。
> **阶段一只服务于可视化**，不保证改造后的 ONNX 还能被 TRT 正确编译运行 ——
> 实际工作流里喂给 TRT 的仍是改造前的原始 ONNX。
>
> 本文的每个设计决策都对应 `graph_statistics.py` / `topological_order_experiment.py`
> 里的一个实测结论，不是拍脑袋。

---

## 0. 范围与分期

| 阶段 | 目标 | 验收 |
| --- | --- | --- |
| **M1（本设计的重点）** | 发现重复子图 → 输出 local function 版 ONNX，**给人看** | 6 层 TransformerEncoder：246 节点 → 主图 ≤ 10 节点 + 1 个 41 节点的 function；Netron 能打开并展开 |
| M2 | 支持并行重复、嵌套多层级 | 双塔 / MoE / 展开的 multi-head 也能抓到；能输出 2~3 级层次 |
| M3 | 保语义 | `onnx.checker` 通过、ORT 输出逐元素相同、TRT parse+build 成功且 engine 输出一致 |

**阶段一能放松什么**（这是它便宜的原因）：

* 不用处理权重堆叠、不用 `Loop`（那是 M3 折叠到 `Loop` 时才需要的）；
* function body 内可以不写 `value_info`（Netron 本来也不显示，见 lutzroeder/netron#1447）；
* 可以提供一个「宽松匹配」档位（忽略属性差异），让本来不完全同构的块也归到一类，
  代价是改造后的图**不再等价** —— 阶段一允许，阶段三禁止。

**但阶段一不能放松什么**：

* 输出必须是合法 protobuf 且 Netron 能打开（否则可视化目标本身就没达成）；
* 每个 function 的 body 必须是**真的**从某个实例复制出来的，不能是拼凑的；
* 必须输出一份报告说明「哪些节点被折叠进了哪个 function」，否则用户没法把可视化结果
  映射回原图去定位问题。

---

## 1. 流水线总览

```
model.onnx
  │
  ├─ P0  预处理：常量折叠 / 去 Identity / 规范化名字      （必做，实测能减 2.1x 节点）
  ├─ P1  建图：GraphIR（节点=算子，边=(out_slot,in_slot)，initializer 作为叶子属性）
  ├─ P2  候选发现
  │       ├─ 路径 A：规范拓扑序 → 最长重复子串（后缀自动机）        ← M1 主力
  │       └─ 路径 B：WL 小半径播种 → 实例组「齐步生长」            ← M2
  ├─ P3  候选验证：凸性检查 + VF2 带边标签同构 + 接口一致性
  ├─ P4  实例选择：MDL 收益贪心 + 非重叠 + 收缩后递归（层次化）
  ├─ P5  Outlining：make_function + 调用节点替换
  └─ P6  校验与报告：checker / ORT 比对（M1 只报告不拦截）/ 统计报告
  │
  └→ model-outlined.onnx + report.json
```

---

## 2. P0 预处理

**为什么必做**：实测 6 层 TransformerEncoder，slim 前 522 节点 / 21 种算子，
slim 后 246 节点 / 12 种算子。被折掉的是 `Constant`(129)、`Identity`(63)、`Shape`/`Slice`/`Cast`
这些 shape 计算管道 —— 它们是纯噪声，而且**会打乱周期性**（它们的数量在不同层之间不一定相同）。

做法：

1. `onnxslim.slim()`（本机已装，实测有效）或 `polygraphy surgeon sanitize --fold-constants`；
2. `onnx.shape_inference.infer_shapes`（给 P1 的严格档位提供 dtype/rank）；
3. **建立 `slim 后节点 → 原始节点名` 的映射表**，报告里要用。
   `onnxslim` 一般保留节点名，保不住的走「输出张量名」兜底。

> 可配置 `--no-preprocess` 跳过，用于调试「预处理是否吃掉了我关心的结构」。

---

## 3. P1 图模型

```python
GraphIR:
    nodes:  id -> NodeInfo(op_type, domain, attr_key, const_input, out_dtype, orig_name)
    edges:  (producer_id, consumer_id, out_slot, in_slot)
```

三个建模决策，每个都有理由：

### 3.1 initializer 不是节点，是叶子属性

同构块之间权重取值**必然不同**。若把 initializer 当节点，任何两层都不同构，算法直接失效。
所以 initializer 记在消费它的节点上：`const_input = {slot: (dtype, shape)}`，
**取值不参与匹配，dtype/shape 按档位参与**。

### 3.2 边必须带 `(out_slot, in_slot)` 双向槽位

* `in_slot`：ONNX 算子输入**有序**，`Sub(a,b) != Sub(b,a)`、`Concat` 顺序决定语义。
* `out_slot`：`Split` / `LSTM` 这类多输出算子，接到第几个输出上是不同的结构。

**这一条直接排除了 `gspan-mining` 这类现成库**（面向无序边的通用图数据库）。

### 3.3 匹配严格度分档（配置项 `strictness`）

| 档 | 节点标签 | 用途 |
| --- | --- | --- |
| `L0` | `op_type` | 最宽松，快速看大致结构 |
| `L1` | `+ domain + 全部 attribute` | **M1 默认** |
| `L2` | `+ const 输入的 dtype/shape` | 区分「同结构不同宽度」的块 |
| `L3` | `+ 激活张量的 dtype/rank` | M3 保语义时用 |

档位越低，能归并的块越多、图越好看，但等价性越弱。M1 默认 `L1`，因为它已经能抓到
transformer layer，而且属性一致是 M3 折叠到 `Loop` 的硬前提（子任务 1 结论 9），
早点对齐可以少返工。

---

## 4. P2 候选发现

### 4.1 路径 A：规范拓扑序 + 最长重复子串（M1 主力）

**核心洞察**：串行堆叠的块在拓扑序里就是**周期性子串**，
于是问题从 NP 难的子图挖掘塌缩成 O(n) 的最长重复子串。

实测（`graph_statistics.py`）：6 层 TransformerEncoder，周期 41、重复 6 次、
位置 `[0,41,82,123,164,205]`、覆盖率 **246/246 = 100%**、VF2 验证 6 个实例全部同构。

#### 4.1.1 规范拓扑序是这条路径的命门

任意拓扑序**不行**。`topological_order_experiment.py` 实测三种结构：

| 结构 | 任意拓扑序 | recency 拓扑序 |
| --- | --- | --- |
| 串行块 + 一条独立并行链 | `ADBDCDADBDCDABCABC` → **完全找不到** | `ABCABCABCABCDDDDDD` → 正确找到 `ABC × 4` |
| 块内部有 Q/K/V 并行分支 | `SQKVM × 4` → 正确 | 同样正确 |
| 双塔，各自有重复块 | `AEBFCGAEBFCG…` → **找到一个假模式** `AEBFCG` | `ABCABCABCABC EFGEFGEFGEFG` → 正确找到 `ABC × 4` |

两个结论：

* **块内部的并行是无害的**（块是单入单出区域，整体连续）；
* **块之间的并行是致命的**，而且不只是「漏检」，还会产出**语义上毫无意义的假模式**
  （两个无关塔交错出来的 `AEBFCG`）。

**采用的排序**：`recency_topological_sort` —— 在就绪节点里，优先发射「前驱完成时间最晚」的那个，
也就是「先把当前正在走的分支走完，再去开新分支」。tie-break 用
`(前驱最晚完成时间 desc, 小半径 WL 哈希 asc, 原始节点序 asc)` 保证确定性。

**理论依据**：一组两两不相交的节点集合能在**某个**拓扑序里同时连续
⟺ 把每个集合收缩成单点后仍是 DAG（即每个集合是**凸的**，且集合之间无环）。
recency 排序是在不知道区域边界的前提下逼近这个序的贪心启发式 ——
它是启发式，不是保证，所以 **P3 的图空间验证不可省**。

#### 4.1.2 重复子串搜索

* 数据结构：后缀自动机 / 后缀数组 + LCP，枚举所有重复子串 O(n log n)。
  （目前实验里用的是朴素 O(n²) 实现，246 长度下 <1s；真实大模型上万节点时再换。）
* 候选过滤：非重叠出现次数 ≥ `min_repeat`（默认 2）、长度 ≥ `min_size`（默认 3）。
* **不是只取最长的一个**：按 MDL 收益排序，取一个 → 提交 → 把已占用位置屏蔽 → 再取，
  直到没有正收益。双塔场景就是靠这一步先拿 `ABC` 再拿 `EFG`。

### 4.2 路径 B：WL 播种 + 实例组齐步生长（M2）

路径 A 抓不到**并行重复**（展开的 multi-head、MoE 的 N 个 expert、多分支特征金字塔），
它们在拓扑序里必然交错。这时回到图空间，但**不能照原思路做**。

#### 4.2.1 原思路的两个缺口（回顾）

* 「从单个算子开始不断加算子」走的是「所有连通子图」空间，实测增长 ~1.85x/节点，
  而一个 encoder layer 是 41 节点 → `246 × 1.85^40 ≈ 2.6e12`，**永远到不了**。
* 「出现次数 ≥ 2」这个目标函数是错的：一个 41 节点块被找到时，它的所有连通子模式**都**出现 6 次。

#### 4.2.2 修正：不枚举模式，而是让**实例组齐步生长**

```
1. WL 小半径（3~5 轮）哈希分桶  →  得到若干「锚点组」
2. 对每个大小 >= 2 的锚点组：
       region[i] = {anchor[i]}   （每个实例一个区域）
       repeat:
           对每个实例，按同一个规范顺序取「下一个待扩张的邻居」
           若所有实例取到的邻居 (标签, out_slot, in_slot) 完全一致 → 全部纳入
           否则 → 停止
3. 得到一组同步生长出来的、天然互相同构的区域
```

关键点：生长由**实例之间的一致性**驱动，不是由枚举驱动，所以**没有组合分支**。
复杂度 O(桶数 × 实例数 × 区域大小)，和图规模基本线性。
这正是把用户原思路「维护一个子模式集合不断添加算子」修正成可行形式的地方 ——
维护的不是「模式集合」，而是「实例组」，一次生长只产生一个后继状态而不是若干个。

#### 4.2.3 WL 半径必须小（实测反直觉结论）

不能靠加大 WL 半径直接找到整个块：

| WL 轮数 | 等价类 | 含重复的类 | 大小恰为 6 的类 |
| ---: | ---: | ---: | ---: |
| 3 | 52 | 39 | 27 |
| 5 | 72 | 41 | 22 |
| 8 | 109 | 41 | 12 |
| 12 | 172 | 27 | **0** |
| 24 | 246 | **0** | **0** |

半径一大，邻域就碰到图边界：第 1 层挨着模型输入、第 6 层挨着输出，天然不同，
6 个本应等价的实例被拆散，24 轮后 246 个节点全成单例（图直径仅 26）。
**WL 只能用于小半径播种**，之后的生长必须把区域边界当**通配符**，
匹配「诱导子图的形状」而不是「节点在全图中的上下文」—— 这是两种不同的匹配语义，实现时极易混淆。

---

## 5. P3 候选验证

分桶/子串只是必要条件，提交前必须在图空间过三关：

### 5.1 凸性（convexity）

候选实例 `S` 收缩成单点后不能产生环 —— 否则「先算 S 的一部分 → 出去算别的 → 再回来算 S」
根本无法表达成一次函数调用。

实现：`contracted = nx.contracted_nodes(...)` 后 `nx.is_directed_acyclic_graph()`，O(V+E)/候选。
（也可以用拓扑序下标做剪枝：先看 `[min_idx, max_idx]` 区间里有没有不属于 S 的节点，
没有就直接判定凸，有才走完整检查。）

### 5.2 两两同构

`networkx.DiGraphMatcher(node_match=categorical_node_match("label"), edge_match=<(out_slot,in_slot)>)`。
实例只有几十个节点，VF2 代价可忽略。实测 6 个 transformer layer 实例全部 `True`。

### 5.3 接口一致性

所有实例必须有**相同的外部输入/输出接口**，且接口位置在同构映射下对应：

* 外部输入 = 实例内节点消费的、由实例外产生的张量（含 graph input 和 initializer）；
* 外部输出 = 实例内产生的、被实例外消费的张量（含 graph output）。

给每个实例算一个「接口签名」`[(实例内节点的规范序号, slot), ...]`，签名不同则整组作废。

> **这一关最容易出静默错误**：某个实例内部的中间张量恰好被实例外的某个节点消费了
> （比如某一层的 attention 权重被 debug 输出引出去了），如果不检查就直接 outline，
> 那个张量在函数体内不可见，图就悄悄错了。
> 正确处理不是拒绝，而是**把它提升为额外的 function 输出**（见 P5）；
> 但只有当**所有实例都有对应的引出**时才成立，否则接口不一致，整组作废。

---

## 6. P4 实例选择

目标函数不是「频次」，是**压缩收益**（MDL 思想）：

```
gain(P) = (|P| - 1) × (k - 1) - overhead(P)
          └ 每次替换省下的节点数 ┘  └ FunctionProto 自身的开销 ┘
k = 非重叠实例数，|P| = 模式节点数
```

* 贪心：按 gain 降序，跳过与已提交实例重叠的候选；
* **层次化**：提交一批后，把每个实例收缩成单点，在收缩图上**重跑 P2**，
  于是能得到 `layer → encoder → model` 的多级结构（`max_level` 默认 2，M2 再放开）；
* 记录每一级的 `n_node_before / n_node_after / coverage`，进报告。

---

## 7. P5 Outlining

### 7.1 硬约束：`FunctionProto` 没有 `initializer` 字段

实测 `onnx.FunctionProto.DESCRIPTOR.fields` =
`[name, input, output, attribute, attribute_proto, node, doc_string, opset_import, domain, overload, value_info, metadata_props]`
—— **没有 initializer**。ONNX function 也**不是闭包**，不能引用外层图的名字。

结论：**权重必须作为 function 的输入**，调用点传入各自实例的 initializer 名字。
子任务 1 里 torch 的 `export_modules_as_functions` 就是这么做的
（`Block(x, w1, b1, w2, b2)`），可以直接照抄这个形态。

### 7.2 生成步骤

```
对每个模式 P（实例 I_1..I_k，取 I_1 为模板）：
  1. 按 I_1 的规范内部顺序，确定 function 的 input 列表：
       外部输入（含 initializer）按「接口签名」的顺序排列
  2. output 列表 = 外部输出（含被实例外消费的中间张量、graph output）
  3. body = I_1 的节点，张量名重命名为函数局部名（f_in_0, f_t_3, f_out_0 ...）
  4. opset_import = body 里用到的所有 domain
  5. helper.make_function(domain="cookbook.outlined", name=f"Block{p}", ...)
  6. 对每个 I_j，删掉它的全部节点，插入一个
       make_node("Block{p}", 实际输入名列表, 实际输出名列表, domain="cookbook.outlined")
  7. 在 model.opset_import 里加上 ("cookbook.outlined", 1)
```

### 7.3 必须踩过的坑清单

| 坑 | 处理 |
| --- | --- |
| **函数体去重** | 一个模式**只发一份** `FunctionProto`，k 个调用节点。`onnxscript.rewriter(as_function=True)` 恰恰做不到这点（每个 match 一份，靠 `overload` 区分，实测见 `02-PackageSurvey/log-demo_rewriter_as_function.py.log`），所以这一步自己写 |
| 中间张量被实例外消费 | 提升为额外 function 输出（5.3） |
| graph output 落在实例内部 | 同上，且调用节点的输出名必须保留原名 |
| 多输出算子（`Split`/`LSTM`） | 边必须记 `out_slot`（3.2），否则接口顺序会错 |
| `model.opset_import` 漏加自定义域 | `onnx.checker` 会报错 |
| IR version | function 需要 IR ≥ 8，`overload` 需要 IR ≥ 10。我们只发一份 function、不用 overload，IR 10 足够 |
| 名字冲突 | 所有新名字加统一前缀，最后跑一遍 `onnx_ir` 的 `NameFixPass` |
| 拓扑序 | 替换后主图节点顺序可能失序，最后跑 `TopologicalSortPass` |
| 子图属性（`Loop`/`If` body）里的模式 | M1 **不进入**控制流子图，只处理主图，进报告说明跳过了几个 |

### 7.4 可视化侧的验证

Netron 的 `onnx.js` 里有完整的 function 处理（`context.functions`、`func.callers`、`func.uses`），
调用节点会以 function 名作为节点类型显示，function body 可以单独浏览。
已知限制：function 体内的 `value_info`（shape）不显示（lutzroeder/netron#1447）——
M1 可以顺手把推断出的 shape 写进 `FunctionProto.value_info`，将来 Netron 修了就能直接受益。

---

## 8. P6 校验与报告

| 检查 | M1 | M3 |
| --- | --- | --- |
| `onnx.checker.check_model` | **拦截** | 拦截 |
| Netron 能打开（用 `netron` 的 js 解析路径做冒烟测试） | **拦截** | 拦截 |
| ORT 逐元素比对（同一批随机输入） | 只报告 | **拦截** |
| TRT `parse` + `build` + engine 输出比对 | 只报告 | **拦截** |
| 任一步失败 | **回退到原图，绝不输出半成品** | 同 |

`report.json` 至少包含：

```json
{
  "preprocess": {"n_node_before": 522, "n_node_after": 246},
  "levels": [{"level": 0, "n_node_before": 246, "n_node_after": 7, "coverage": 1.0}],
  "patterns": [{"name": "Block0", "size": 41, "n_instance": 6, "gain": 200,
                "instances": [["原始节点名", "..."], "..."]}],
  "skipped": {"subgraph_node": 0, "non_convex_candidate": 3, "interface_mismatch": 1},
  "verification": {"onnx_checker": "pass", "onnxruntime_max_diff": 0.0, "tensorrt_parse": "pass"}
}
```

`instances` 里存**原始节点名**是硬要求：用户在 Netron 里看到 `Block0`，
必须能查回它对应原图的哪 41 个节点。

---

## 9. 对外接口

```python
from tensorrt_cookbook.onnx_outliner import OutlineConfig, outline

config = OutlineConfig(
    min_repeat=2,          # 至少重复几次才折叠
    min_size=3,            # 模式至少几个节点
    strictness="L1",       # L0/L1/L2/L3，见 3.3
    max_level=2,           # 层次化折叠的层数
    method="auto",         # "serial"(路径A) / "parallel"(路径B) / "auto"
    preprocess=True,
    verify="report",       # "report"(M1) / "strict"(M3)
)
report = outline("model.onnx", "model-outlined.onnx", config)
```

CLI：

```bash
python3 -m tensorrt_cookbook.onnx_outliner model.onnx -o model-outlined.onnx \
        --min-repeat 2 --strictness L1 --max-level 2 --report report.json
```

---

## 10. 里程碑与验收标准

### M1（预计 ~1 人周）

* [ ] P0 预处理 + 映射表
* [ ] P1 GraphIR（含 `out_slot`/`in_slot`、initializer 作叶子、4 档严格度）
* [ ] P2 路径 A（recency 拓扑序 + 重复子串 + 多模式迭代）
* [ ] P3 凸性 / VF2 / 接口一致性
* [ ] P4 MDL 贪心 + 单层折叠
* [ ] P5 Outlining（含 7.3 全部坑）
* [ ] P6 checker + Netron 冒烟 + report.json

**验收**：
1. 6 层 TransformerEncoder：246 节点 → 主图 ≤ 10 节点 + 1 个 41 节点 function，覆盖率 100%；
2. `topological_order_experiment.py` 的三个结构全部得到期望模式（作为回归测试）；
3. Netron 能打开输出文件并展开 function；
4. `report.json` 能把每个 function 实例映射回原始节点名。

### M2（预计 ~1~2 周）

路径 B（WL 播种 + 齐步生长）、嵌套层次、并行重复结构。

### M3（预计 ~1~2 周）

保语义：ORT 逐元素相同 + TRT 编译运行一致；`Loop` 折叠后端（权重堆叠 + `Gather(W, iter)`，
模板见 `01-SubgraphInONNX/main.py::case_loop_stacked_weight`）。
注意子任务 1 实测的代价：`Loop` 折叠会让延迟变成约 2 倍、engine 体积不变。

---

## 11. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| recency 拓扑序在某些图上仍然交错 | 路径 A 漏检 / 假模式 | P3 图空间验证兜底（假模式一定会被凸性或同构检查干掉）；漏检则 M2 的路径 B 补 |
| 预处理吃掉了用户关心的结构 | 折叠结果不符预期 | `--no-preprocess` 开关 + 报告里列出被折掉的节点数 |
| 大模型（1.6 GB 那种）内存/耗时 | 跑不动 | 外部权重不加载（`load_external_data=False`），只按 shape/dtype 匹配；后缀自动机 O(n) |
| 严格度选得不对，一个模式都找不到 | 用户体验差 | 报告里输出「若降到 L0 可多找到 N 个模式」的提示 |
| 控制流子图里的重复被忽略 | 覆盖不全 | M1 明确不做，报告里显式说明跳过数量 |
| 折叠后 Netron 反而更难看（模式太碎） | 达不到可视化目的 | `min_size` / `min_repeat` 门槛 + MDL 收益必须为正 |

---

## 12. 测试集

| 用例 | 来源 | 考察点 |
| --- | --- | --- |
| 三个合成结构 | `topological_order_experiment.py` | 拓扑序回归（漏检 + 假模式） |
| 6 层 TransformerEncoder | `graph_statistics.py` | 主场景 |
| `01-SubgraphInONNX/model-01-flat.onnx` | 已有 | 最小可用例 |
| ResNet 类（多 stage，每 stage 内块数不同） | 待补 | 多模式 + 层次化 |
| 展开的 multi-head / MoE | 待补 | 并行重复（M2） |
| `00-Data/model-large.onnx`（1.6 GB） | 已有 | 规模与内存 |
