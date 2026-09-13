# 子任务 2：现成 Python 包调研

## 问题定义

我们要找的是这样一个能力（编译器里叫 **outlining**，是 inlining 的逆操作）：

> 输入一个被完全展开的 ONNX，**自动（自底向上、无需人工给模式）**找出图中反复出现的子图，
> 把它们提取成共享的子模块（`FunctionProto` 或 `Loop` body），并替换掉原图中的对应节点。

关键词是 **「自动发现」**。「我告诉你要找什么，你帮我替换」是另一回事，那个已经有很多轮子了。

## 结论（一句话）

**没有现成的包能做这件事。** 全部现有工具都只提供两半中的**后一半**（给定模式 → 替换），
或者干脆只做**反方向**（function → 摊平）。「自动发现重复子图」这一半在 ONNX 生态里是空白。

## 逐个包的核查

`check_existing_tools.py` 扫描了本机所有相关包的公开 API（日志见
`log-check_existing_tools.py.log`）。汇总：

| 包 | 相关能力 | 是否满足需求 | 说明 |
| --- | --- | --- | --- |
| `onnx` (1.21) | `helper.make_function` | ❌ 只是**构造** | 给你造 `FunctionProto` 的砖，不告诉你该造什么 |
| `onnx.inliner` | `inline_local_functions` / `inline_selected_functions` | ❌ **反方向** | 只有 inlining，没有 outlining |
| `onnx.utils` | `extract_model(input_names, output_names)` | ❌ 需人工指定边界 | 按你给的 IO 名字切一刀，切完是独立 model 不是 function |
| `onnx.compose` | `merge_models` / `add_prefix` | ❌ | 拼接，不是提取 |
| `onnx_ir` (0.2.1) | `InlinePass`、`CommonSubexpressionEliminationPass`、`DeduplicateInitializersPass` | ❌ | **有 `InlinePass` 但没有对应的 OutlinePass**。CSE 只合并「输入完全相同」的重复计算，我们要的是「输入不同、结构相同」，两回事 |
| `onnxscript.rewriter` | `RewriteRule(target_pattern, replacement_pattern, as_function=True)` | 🟡 **半个** | 见下文，能力最接近，但模式必须手写 |
| `onnx_graphsurgeon` (0.6.1) | `Graph.layer/cleanup/toposort/fold_constants`、`GraphPattern`/`match_all` | ❌ | `GraphPattern` 能做「给定模式 → 找到并替换」，但模式仍要用户手写 |
| `onnxslim` (0.1.96) | 常量折叠 / 算子融合 | ❌ | 一批**硬编码**的融合规则（Conv+BN 之类） |
| `polygraphy` | `FoldConstants` / `extract_subgraph` | ❌ | 同样需要人工指定 IO |
| `onnxruntime.transformers.optimizer` | `FusionAttention` / `FusionLayerNormalization` … | ❌ | 每个 fusion 都是**几百行手写**的模式匹配代码，正是我们想避免的 |
| `onnx-tool` (ThanatosShinji) | `benchmark/do_fusion.py` 的 pattern fusion | ❌ | 同上，模式需人工描述 |
| `networkx.algorithms.isomorphism` | `DiGraphMatcher` / `vf2pp` / `ISMAGS` | 🟡 **底层积木** | 通用子图同构判定，可以拿来做「验证候选」，但不做「发现候选」，也完全不懂 ONNX 语义 |
| `gspan-mining` / `cgspan-mining` / `Gaston` | gSpan 频繁子图挖掘 | 🟡 **底层积木** | 真正的频繁子图挖掘算法，但：面向无标签/简单标签的通用图数据库、纯 Python 很慢、不支持「边有序」（ONNX 的算子输入是**有序**的，`Sub(a,b) != Sub(b,a)`），也没有 ONNX 适配层 |

## 最接近的东西：`onnxscript.rewriter(..., as_function=True)`

这是本次调研里唯一一个**能把匹配到的子图真的变成 ONNX local function** 的现成实现。
`demo_rewriter_as_function.py` 做了实测（日志 `log-demo_rewriter_as_function.py.log`）：

```
Before: main-graph nodes = 25, functions = 0
Matches replaced: 4
After : main-graph nodes = 5, functions = 4
        main-graph node list = [('cookbook','MlpBlock') x4, ('','Identity')]
        function cookbook::MlpBlock(overload='1') body = ['MatMul','Add','Relu','MatMul','Add','Relu']
        function cookbook::MlpBlock(overload='2') body = [...]
        function cookbook::MlpBlock(overload='3') body = [...]
        function cookbook::MlpBlock(overload='4') body = [...]
Max |flat - function| = 0.000e+00
TensorRT parse of the function version: ok=True, network layers=49
```

25 个节点收缩成 5 个，数值逐比特一致，TensorRT 也照收（照例内联回 49 层）。

但是它有三个硬限制：

1. **`target_pattern` 必须由人手写。** 库里没有任何东西会去发现这个模式 ——
   这正是我们缺的那一半。
2. **`as_function=True` 要求 replacement 只有一个节点**，否则直接
   `ValueError: as_function=True is only supported for patterns with a single replacement node.`
   所以写法必须是「替换成一个自定义域的单节点」。
3. **每个 match 生成一个独立的 `FunctionProto`**（靠 `overload` 字段区分，见上面输出里的
   `overload='1'..'4'`），**不会**把结构相同的 body 合并成一份共享 function。
   源码 `onnxscript/rewriter/_rewrite_rule.py` 里 `_get_new_overload(model, domain, name)`
   就是干这个的。想要「一份 body、N 处调用」还得自己再做一遍去重。

## ONNX 官方的动向

`onnx/onnx` 上有一个 RFC [#7301 *Equivalent subgraphs and compiled subgraphs*](https://github.com/onnx/onnx/issues/7301)，
提议在 `GraphProto` 里加 `EquivalentSubGraphProto`，让一段计算可以有多条等价路径
（其中一条可以是「调用一个 local function」或「一段已编译的二进制」）。
但它解决的是**表示**问题，且明确**没有涉及自动发现重复子图**，目前也还在讨论阶段、没有实现。

## 相邻领域的现成算法（可以借用，但都要自己接到 ONNX 上）

| 领域 | 代表 | 与我们的关系 |
| --- | --- | --- |
| 编译器 outlining | LLVM `MachineOutliner`、MLIR 的 outline pass | 思路完全对口：找重复代码序列 → 提成函数。但作用在线性指令序列/region 上，不是 DAG |
| 编译器 loop re-rolling | LLVM `-loop-reroll` | 把展开的循环还原成循环，正是子任务 1 里 `loop_stacked_weight` 那个变换的编译器版本 |
| 图挖掘 | gSpan、Gaston、FSG、**SUBDUE** | SUBDUE 最贴合用户提的思路：用 MDL 准则迭代地生长子结构并替换。但没有维护良好的 Python 实现 |
| 硬件设计空间探索 | *Automated Design Space Exploration of CGRA PE Architectures using Frequent Subgraph Analysis* | 在数据流图上跑 FSM 找高频子图，场景和我们几乎同构 |
| 深度学习编译器 | TVM Relay/Relax 的 `FuseOps`、TASO/PET 的图替换 | 都是「给定重写规则库」或「搜索等价变换」，不是「发现重复结构」 |

## 给子任务 3 的输入

1. **必须自己实现发现这一半**，这是明确的空白，没有轮子可抄。
2. **替换这一半可以直接复用 `onnxscript.rewriter`**（注意上面三个限制），
   或者用 `onnx.helper.make_function` 自己拼，工作量都不大。
3. **可复用的底层积木**：`networkx` 的 `DiGraphMatcher`/`vf2pp`（同构验证）、
   `onnx_ir` 的 IR 与 pass 框架（图操作、拓扑排序、CSE）。
4. **gSpan 这类通用 FSM 算法不能直接用**，主要是两点不匹配：
   ONNX 的算子输入**有序且带属性**，而且我们要的不是「所有频繁子图」而是
   「一组互不重叠、能最大化压缩率的重复模块」—— 后者更接近 SUBDUE 的 MDL 目标。
5. 子任务 1 已经证明：折叠到 **local function** 对 TRT 无副作用，折叠到 **`Loop`** 才有
   build-time 收益但要付 ~2x latency，因此工具应该把这两个后端做成可选的输出模式。

## 参考

- [[RFC] Equivalent subgraphs and compiled subgraphs · onnx/onnx#7301](https://github.com/onnx/onnx/issues/7301)
- [onnx.inliner — ONNX documentation](https://onnx.ai/onnx/api/inliner.html)
- [onnx.helper — make_function](https://onnx.ai/onnx/api/helper.html)
- [microsoft/onnxscript — rewriter](https://github.com/microsoft/onnxscript)
- [ThanatosShinji/onnx-tool](https://github.com/ThanatosShinji/onnx-tool)
- [betterenvi/gSpan — Python 实现，支持有向图](https://github.com/betterenvi/gSpan)
- [gspan-mining · PyPI](https://pypi.org/project/gspan-mining/0.2.2)
- [cgspan-mining · PyPI](https://pypi.org/project/cgspan-mining/)
- [gSpan: Graph-Based Substructure Pattern Mining](https://dl.acm.org/doi/10.5555/844380.844811)
- [Automated Design Space Exploration of CGRA Processing Element Architectures using Frequent Subgraph Analysis](https://arxiv.org/pdf/2104.14155)
- [GitGraph — Architecture Search Space Creation through Frequent Computational Subgraph Mining](https://arxiv.org/pdf/1801.05159)
- [FLEXIS: FLEXible Frequent Subgraph Mining using Maximal Independent Sets](https://arxiv.org/pdf/2404.01585)
- [ZhangGe6/onnx-modifier](https://github.com/ZhangGe6/onnx-modifier)

## 更正（2026-08-25）

上表原先写 `onnx_graphsurgeon`「没有任何 pattern search 入口」，这句不准确 ——
gs 有 `GraphPattern` / `PatternMapping` / `match_all`。**本文结论不变**：
那里的模式仍需用户手写，属于「后一半」，不是自动发现。
另外后续实测还纠正了两点，详见 [`../04-APIChoice/README.md`](../04-APIChoice/README.md)：
gs 的 `Function` 是一等公民（往返无损），且大模型 import 走 lazy values、无额外内存开销。
