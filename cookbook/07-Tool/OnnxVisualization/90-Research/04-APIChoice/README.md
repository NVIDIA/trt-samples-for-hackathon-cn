# 图修改 API 选型：`onnx` 原生 vs `onnx_graphsurgeon` vs `onnx_ir`

> 结论：**`onnx_graphsurgeon` 作主力，`onnx.helper` 作逃生舱**，
> 并且把 API 依赖收拢在流水线的两端（P0/P1 导入、P5 改写），中间的挖掘算法不碰任何一方。

判断不是凭文档，而是**把同一件事（把 4 个重复块 outline 成一份共享 `FunctionProto`）用两套 API 各写一遍**：

* `outline_with_onnx_api.py` —— 原生 `onnx`
* `outline_with_graphsurgeon.py` —— `onnx_graphsurgeon`
* `benchmark_large_model.py` —— 1.6 GB / 6119 节点大模型的时间与内存

两份输出**逐字节等价**（调用节点输入输出、initializer、function 签名、opset_import 完全相同），
ORT 数值 `max diff = 0.0`，`onnx.checker` 都通过。

---

## 1. 代码量：51 行 vs 73 行

| 实现 | 有效代码行 |
| --- | ---: |
| `outline_with_graphsurgeon.py` | **51** |
| `outline_with_onnx_api.py` | 73 |

多出来的 22 行**全部**是这四件事，而且恰好是最容易写出 bug 的四件事：

| 原生 API 必须手写 | gs 自带 |
| --- | --- |
| `producer = {output_name: node_index}` 生产者索引 | `Tensor.inputs` / `Tensor.outputs` 天然双向 |
| 「谁消费了这个张量」要遍历全图 | 同上，O(1) |
| 张量重命名的簿记（哪些是入参、哪些是出参、哪些是内部临时） | 直接换 `Tensor` 对象，名字是次要的 |
| 手写拓扑排序 + 删除死节点 | `graph.cleanup().toposort()` |

原生版里那段手写拓扑排序是典型的「看着简单、边界条件一堆」的代码
（空输入名 `""`、initializer 与 graph input 的初始可见集合、环检测）。
我们的流水线里 P5 每折叠一层就要跑一次，写错一次就是静默错图。

## 2. `Function` 支持：三者都是一等公民（这是最关键的一点）

我原以为 gs 对 `FunctionProto` 支持很弱，实测不是：

```
gs.Function(name, domain=..., nodes=..., inputs=..., outputs=..., opset=..., import_domains=...)
gs.Graph.functions   # list[Function]
```

`gs.Function` 直接继承自 `Graph`，自带 `cleanup()` / `toposort()` / `fold_constants()` / `layer()`，
import→export 往返后 function 数量、body、签名全部保持不变，`onnx.checker` 通过。

`onnx_ir` 也支持，而且 key 是 `(domain, name, overload)` 三元组，比 gs 的 `list` 更严谨。

## 3. 大模型：三者都没问题

`benchmark_large_model.py`（1.52 GB，6119 节点，293 个 initializer）：

```
onnx.load               :   2.49s  peakRSS=  3.8GB
gs.import_onnx          :   0.33s  peakRSS=  3.8GB     <- 没有额外内存
gs.toposort             :   0.06s  peakRSS=  3.8GB
gs.export_onnx          :   0.65s  peakRSS=  3.8GB
ir.from_proto           :   0.13s  peakRSS=  3.8GB
```

gs 的 `Constant` 走 lazy values，**不会**在 import 时把权重全部 materialize 成 numpy，
峰值内存和裸 `onnx.load` 完全一样。这条原本是我对 gs 最大的顾虑，实测被排除。

## 4. 逐项对比

| 维度 | `onnx` 原生 | `onnx_graphsurgeon` 0.6.1 | `onnx_ir` 0.2.1 |
| --- | --- | --- | --- |
| 生产者/消费者索引 | ❌ 自己建、自己维护 | ✅ `Tensor.inputs/outputs` 双向自动 | ✅ `Value.producer()/uses()` |
| 节点邻居导航 | ❌ | ✅ `Node.i(idx)` / `Node.o(idx)` | ✅ |
| 拓扑排序 / 清死节点 | ❌ 手写 | ✅ `cleanup().toposort()` | ✅ `TopologicalSortPass` 等 |
| `FunctionProto` | ✅ `helper.make_function`，完全控制 | ✅ `gs.Function`（继承 Graph） | ✅ key 含 `overload` |
| 用户指定模式匹配 | ❌ | ✅ `GraphPattern` / `match_all` | ✅ `onnxscript.rewriter` |
| 大模型 lazy 权重 | ✅（`load_external_data=False`） | ✅ | ✅ |
| 表达完整度 | ✅ **100%**，proto 就是它自己 | 🟡 少数字段（`attribute_proto` / ref-attr、`metadata_props`）覆盖较薄 | ✅ 接近完整 |
| 控制流子图（`Loop`/`If` body） | ✅ 直接操作 `GraphProto` | ✅ `Graph.subgraphs()` | ✅ |
| pass 框架 | ❌ | ❌ | ✅ 现成一批（CSE / Inline / NameFix / Dedup） |
| 生态位 | ONNX 官方，永远不会错 | **TensorRT 官方**，cookbook 已有 `07-Tool/OnnxGraphSurgeon` | onnxscript / torch 导出器的底座 |
| API 稳定性 | 最稳 | 稳（0.6.x 变化不大） | 🟡 0.2.1，还在动 |

## 5. 选型与理由

### 主力：`onnx_graphsurgeon`

1. **代码少 30%，而少掉的正是易错部分**（生产者索引、重命名簿记、拓扑排序）；
2. **`Function` 是一等公民**，往返无损；
3. **大模型零额外开销**，唯一的顾虑被实测排除；
4. **它是 TensorRT 官方工具**，cookbook 里已经有 `07-Tool/OnnxGraphSurgeon` 这一节，
   读者的知识可以复用，这个工具将来也更可能落在 cookbook 的工具链里；
5. `GraphPattern` / `match_all` 在 M2/M3 想加「用户手写模式」这个入口时可以直接用。

### 逃生舱：`onnx.helper` + 直接操作 proto

gs 覆盖不到的地方走原生 API，具体预计有三处：

* **`attribute_proto` / `ref_attr_name`**：M2/M3 如果要让同一个 function 承载「属性逐实例不同」的块，
  需要 function 参数化属性，gs 的 `Node.AttributeRef` 有雏形但很薄，大概率要落到 proto 层；
* **最终 `onnx.checker.check_model` 与 `shape_inference`**：本来就是原生 API；
* **报告里读原始 proto 的元信息**（`doc_string`、`metadata_props`、producer 信息）。

### 不选 `onnx_ir` 作主力的原因（但保留切换可能）

`onnx_ir` 技术上最现代（pass 框架、`overload` key、更完整的 IR），如果只看工程质量它其实最好。
不选它是两个非技术理由：**0.2.1 版本 API 还在动**，以及**它不是 TensorRT 生态的东西**，
放进 TensorRT cookbook 里解释成本更高。

如果后面 gs 在 `attribute_proto` 上卡住、或者我们需要它那套 pass，切换成本可控 —— 见下。

### 架构上的降险：把 API 依赖收拢在两端

`DESIGN.md` 的流水线里，**挖掘算法（P2 候选发现 / P3 验证 / P4 选择）只跑在我们自己的
轻量只读 `GraphIR` 上**（节点 = `(op_type, domain, attr_key)`，边 = `(out_slot, in_slot)`），
不认识 gs 也不认识 onnx。

所以 API 只出现在两个薄层：

```
P0/P1  某某 API 的 model  ──►  GraphIR（只读，供挖掘）
P5     挖掘结果（节点 id 列表）──►  某某 API 的图改写
```

真要换 API，只动这两层，挖掘算法一行不改。这一点在设计上是刻意的。

## 6. 一处需要更正的旧结论

`02-PackageSurvey/README.md` 里我写过 gs「没有任何 pattern search 入口」，这句话不准确 ——
gs 有 `GraphPattern` / `PatternMapping` / `match_all`。
不过那份调研的**结论不变**：`GraphPattern` 里的模式仍然要由用户手写，
它提供的是「给定模式 → 找到并替换」，不是「自动发现重复子图」。
