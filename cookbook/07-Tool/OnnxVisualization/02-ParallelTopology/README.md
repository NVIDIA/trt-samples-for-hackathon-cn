# 并联 + 跨连拓扑测试

前面所有测试都是**串联**的重复块。这里换成真正有分叉、汇合、跨连的 DAG。

## 拓扑

五个结构相同的 TransformerEncoderLayer（A~E，各自权重不同）：

```
x --> A --+--> B ------+
          |            |--> h = B * C --+
          +--> C --+---+                +--> Add --> D --+
          |        |                                     |--> f = D + E
          +········+--> E ---------------------------->--+
          :
          +········(A --> D 跨连，仅 case_with_skip)
```

原始描述里 `A->D` 和 `h->D` 同时存在，但 transformer block 只吃一个张量，所以两种读法都测：

* **`case_with_skip`** —— `A->D` 是真实的边，D 的输入是 `h + A_out`（标准残差形式）。
  这是更难的一种：**D 的汇合点和其他块都不一样**。
* **`case_transitive_only`** —— `A->D` 只是「D 在 A 的下游」的口语说法，D 只吃 `h`。

## 运行

```bash
python3 main.py    # 通过返回 0，失败返回 1
```

两个变体的模型来自 [`../00-ModelZoo`](../00-ModelZoo/README.md)
（`transformer_branchy` 与 `transformer_branchy_transitive`），本目录不再自己导出。
`../01-BasicUsage/test_outliner.py` 对同样这两个模型也有断言。

## 结果：两种读法都把五个块折进了同一个 function

| 变体 | 输入 | 折叠后 | 覆盖率 | checker | onnxruntime | TensorRT |
| --- | ---: | --- | ---: | --- | --- | --- |
| `with_skip` | 412 节点 | slim 后 206 → **主图 16 节点 + 1 个 39 节点 function ×5** | 94.7% | pass | **diff 0.0** | pass (335 层) |
| `transitive_only` | 411 节点 | slim 后 205 → **主图 15 节点 + 1 个 39 节点 function ×5** | 95.1% | pass | **diff 0.0** | pass (334 层) |

`report-*.json` 里可以确认 5 个实例分别精确对应模块 A / B / C / E / D，没有任何跨模块串味：

```
instance covers module(s) ['A'], 39 nodes, e.g. /A/self_attn/MatMul_output_0
instance covers module(s) ['B'], ...
instance covers module(s) ['C'], ...
instance covers module(s) ['E'], ...
instance covers module(s) ['D'], ...
```

注意实例顺序是 **A, B, C, E, D** —— 这正是 recency 拓扑序的效果：走完 C 之后先把 E 走完再回头做 D。
`rejected` 为空，说明 1-D 搜索给出的候选一次就全对，图空间验证没挡下任何东西。

## 两个观察

### 1. 块是 39 节点而不是 41

串联 transformer 里每层是 41 节点，这里是 39。原因是 **onnxslim 的公共子表达式消除破坏了块的对称性**：
B 和 C 都消费 A 的输出并各自做一遍相同的 `Transpose + Reshape`，slim 把这两份合并成一份，
于是 slim 后各模块的节点数变成 `A:41, B:41, C:39, D:41, E:41`。

工具的表现是**优雅降级**：它找到了五个块的**最大公共核心**（39 节点），
把多出来的 4 对 `Transpose+Reshape` 留在主图里，而不是因为「不完全同构」就放弃。
主图里剩下的 16 个节点就是这 8 个 + `Mul` / `Add` / `Add_1` 这些块间接线 + 2 个边界 Reshape。

### 2. 这个测试挖出了两个真的正确性 bug

用 `--no-preprocess` 跑这个模型时（不折常量，图里保留大量 `Constant` 和 `Identity`），
产出的模型 **`onnx.checker` 通过但数值是错的**。定位后是两个独立的 bug，都已修复并加进
`../01-BasicUsage/test_outliner.py` 的回归用例：

| bug | 现象 | 根因 | 回归用例 |
| --- | --- | --- | --- |
| **`Constant` 属性只按 dtype/shape 比较** | 4 个持有不同常量的块被错误合并，ORT diff = **6.0** | 属性值是**烤进 function body** 的，不像 initializer 输入那样按调用点传。只比 dtype/shape 会把 `[2,16,64]` 和 `[8,16,16]` 判为同一个 | `constant_attribute` |
| **function body 的输入按张量身份重映射** | 参考实例里共享的张量把多个输入槽塌缩成一个，其他调用点静默丢参数，ORT diff = **1.298** | `remap[id(tensor)]` 后写覆盖先写。必须按 `(offset, slot)` 键 | `shared_input` |

两个 bug 都是**默认路径（开预处理）碰不到**的：onnxslim 会把 `Constant` 折掉、把 `Identity` 删掉。
也就是说它们潜伏在代码里，只有这个更复杂的模型 + 关掉预处理才暴露出来。

**这也说明了一件事**：`onnx.checker` 通过完全不代表图是对的。
`OutlineConfig(verify=...)` 里的 onnxruntime 逐元素比对不是锦上添花，是唯一抓得住这类错误的闸门。
