# Model Zoo —— 测试样例集（模型 + 期望结果）

这些模型是在追某个具体问题时一个一个写出来的，好几个原本只活在某个测试函数或临时脚本里。
每个都**孤立地考察图的一个性质**，这正是它们值得留下来的原因 ——
合在一起就是一个「关于 ONNX 里重复结构」的小型语料库。

```bash
python3 build_model_zoo.py            # 生成 model/*.onnx 和 manifest.json
python3 build_model_zoo.py --check    # 再用 onnxruntime 逐个加载验证
```

**28 个模型，onnxruntime 全部可加载。** 生成物在 `model/`，索引在 `manifest.json`。

## 这里是唯一的真相源

| 文件 | 作用 |
| --- | --- |
| **`model_zoo.py`** | **库**：所有构图函数 + `EXPECT` 期望表。谁都从这里拿模型 |
| `build_model_zoo.py` | CLI：把模型写到 `model/`，把 `EXPECT` 一并写进 `manifest.json` |
| `manifest.json` | 索引：节点数 / 体积 / 描述 / **期望模式** / ORT 加载结果 |

三处消费者，都不再自己造模型：

* [`../01-BasicUsage/test_outliner.py`](../01-BasicUsage/test_outliner.py) —— **导入** `model_zoo`，
  对 `EXPECT` 里每一条做一个用例（zoo 全扫），再对其中几个做深度用例；
* [`../01-BasicUsage/main.py`](../01-BasicUsage/main.py) —— `ensure("transformer_6layer")` 等；
* [`../02-ParallelTopology/main.py`](../02-ParallelTopology/main.py) —— 分叉 DAG 的两个变体。

`EXPECT[name]` 长这样，`pattern` 是报告里的模式列表 `(块大小, 实例数)`，按收益降序：

```python
"ambiguous_sibling": dict(config=dict(preprocess=False), pattern=[(4, 4)],
                          note="4, not 3: without edge disambiguation the side branch is left behind"),
```

改了模型忘了改期望，或者改了算法碰巧改变了某个模型的结果，`test_outliner.py` 立刻会红。
反过来，往 zoo 里加模型却忘了写 `EXPECT`，`build_model_zoo.py` 会打印
`WARNING: no entry in model_zoo.EXPECT for [...]` 并返回非 0 —— 不静默。

## 一、重复结构的拓扑（考察「能不能找到」）

| 模型 | 节点 | 期望 | 考察什么 |
| --- | ---: | --- | --- |
| `serial_chain` | 12 | `(3,4)` | 4 个块串联，最简单的情况 |
| `serial_plus_parallel` | 18 | `(3,4) (3,2)` | 串行链旁边一条独立链。**任意拓扑序会交错成 `ADBDCD...`，完全漏检** |
| `two_tower` | 24 | `(3,4) (3,4)` | 两个独立塔。**任意拓扑序会报出交错的假模式 `AEBFCG`** |
| `internal_branch` | 20 | `(5,4)` | 并行发生在**块内部** —— 无害，块整体仍然连续 |
| `fan_out` | 16 | `(3,5)` | N 条并列分支。**能找到，但没有循环依赖，`Loop` 表达不了** |
| `shared_hub` | 17 | `(3,5)` | 每条分支都读中间一个共享张量（成为第 2 个 function 输入） |
| `shared_accumulator` | 17 | `(3,5)` | 分支串在一条共享累加链上 |
| `nested_two_level` | 20 | `(3,6)` | 2 组 × 3 个块，**真正的两级结构**；`--max-level 2` 再多出 `(4,2)` |
| `planted_block` | 11 | `(5,2)` | 随机植入块生成器的一次采样：filler 落进块中间，打断了它在拓扑序里的连续段。`--method serial` 只报 `(4,2)` |

## 二、改写器的陷阱（考察「改完对不对」）

这三个都会产出 **`onnx.checker` 通过但数值错误**的模型，只有逐元素比对能发现。

| 模型 | 节点 | 期望 | 陷阱 |
| --- | ---: | --- | --- |
| `shared_input` | 16 | `(3,4)` | **参考实例**把同一个张量喂给三个输入槽，其他实例不这样。按张量身份重映射 function body 会把这三个槽塌缩成一个，其他调用点静默丢参数 |
| `constant_attribute` | 12 | **无模式** | `Constant` 属性 dtype/shape 相同但**值不同**。属性值是**烤进 body** 的（不像 initializer 输入按调用点传），只比 dtype/shape 会错误合并。注意只在 `preprocess=False` 时成立，onnxslim 会把它们变成 initializer |
| `ambiguous_sibling` | 17 | `(4,4)` | 同一个值喂给**两个同类型**的节点。`(方向, slot, slot, 标签)` 分不开这两条边，遇到歧义就跳过会把侧分支丢在外面 —— 期望是 4 而不是 3 |

## 三、shape 敏感算子（考察「属性 vs 输入」）

算法有一条核心前提：**节点标签 = `(op_type, 全部属性)`，属性不同即不同模式**。
原因是**属性值被烤进 function body**，而输入张量按调用点传 —— 所以
「形状信息放在属性里」和「放在输入里」的算子，判定结果正好相反。
这三个样例把这条前提的两个方向都钉住。

| 模型 | 节点 | 期望 | 考察什么 |
| --- | ---: | --- | --- |
| `transpose_perm` | 12 | `(3,2)` ×2 | 四个分支两种 `Transpose.perm`。`perm` 是**属性** → 必须切成**两个**模式，不能合成一个 `(3,4)` |
| `concat_axis` | 12 | `(3,2)` ×2 | 同上，但 `axis` 会改变**输出形状** —— 错误合并连图都跑不起来 |
| `reshape_shape_input` | 16 | `(4,4)` | **反方向**：`Reshape` 的目标形状是**输入张量**，按调用点传，所以四个不同目标**必须合成一个**模式 |

前两个是负向对照，第三个是正向对照 —— 少了它，一个「把标签收得更紧」的修复
会连带把本该合并的也拆开，而且不会有任何测试报警。

**为什么这三个值得单独列一节**：把 `--strictness` 降到 `L0`（标签里不含属性）后，
前两个会被错误合并，而**两道结构性闸门全部放行**：

| L0 下 | `onnx.checker` | TensorRT parse | onnxruntime |
| --- | --- | --- | --- |
| `transpose_perm` | **pass** | **pass**（12 层） | **mismatch**，`max_abs_diff` 0.114 |
| `concat_axis` | **pass** | **pass**（12 层） | **error**：`operands could not be broadcast` |

这是本项目第三次撞上同一件事：`onnx.checker` 通过、TensorRT 能 parse，
都不代表图是对的，**只有逐元素数值比对拦得住**。
`../01-BasicUsage/test_outliner.py` 里 `transpose_perm_L0` / `concat_axis_L0`
两组用例把这个对比整个锁住了 —— 包括「L0 必须错」和「必须是数值闸门发现的」。

> 顺带一条实用结论：**`--strictness L0` 对含属性的算子不安全**，只适合确知
> 全图算子都没有语义相关属性的场合。默认的 `L1` 才是该用的。

## 四、`Loop` / `If` 相关（考察「能不能折成控制流」）

| 模型 | 节点 | 期望 | 考察什么 |
| --- | ---: | --- | --- |
| `loop_scan_output` | 13 | `(3,4)` | 一个 loop-carried 输出 + 一个离开链的输出 → 后者必须变成 **scan output** 再逐迭代切片回传 |
| `loop_two_carried` | 19 | `(4,4)` | 块把**两个**值交给下一迭代 → **两个循环变量**。只支持一个的话几乎所有真实模型都折不了 |
| `loop_with_repeats` | **1** | `(3,4)` | 重复块只在 `Loop` **body** 里。主图只有一个节点，只看顶层什么都找不到 |
| `if_with_repeats` | **1** | `(3,4) (3,4)` | `If` 的两个分支各自含重复块 |

## 五、从 PyTorch 导出的（真实形态）

| 模型 | 节点 | 体积 | 期望 | 说明 |
| --- | ---: | ---: | --- | --- |
| `flat_mlp` | 16 | 5 KB | `(4,4)` | 4 个独立权重的 MLP 块，完全展开 |
| `flat_mlp_as_function` | **4** | 4 KB | **无模式** | 同样 4 个块，用 `export_modules_as_functions` 导出 —— **PyTorch 已经帮你 outline 好了**。原有 function 必须被原样带过 |
| `transformer_6layer` | 522 | 264 KB | `(41,6)` | 6 层相同的 encoder，**整个项目的主目标** |
| `transformer_two_stage` | 524 | 276 KB | `(41,6)` | `(3 层 + tanh) × 2`，两级 transformer |
| `transformer_branchy` | 412 | 1016 KB | `(39,5)` | 5 层 encoder 组成的分叉/汇合/跨连 DAG，`D` 吃 `h + A` |
| `transformer_branchy_transitive` | 411 | 1016 KB | `(39,5)` | 同一个 DAG 的另一种读法，`D` 只吃 `h`，`A` 只是传递可达 |
| `moe_6expert` | 23 | 200 KB | `(4,5)` | 6 条并列 expert 分支求和。块是 `Gemm/Relu/Gemm/Add`，而 expert 0 没有 `Add`，所以是 5 个实例不是 6 |
| `loop_body_repeats` | 2 | 1 KB | `(3,4)` | 真正的 `Loop`，body 里含 4 个重复块 |
| `loop_shared_weight` | 2 | 2 KB | **无模式** | trip count 来自运行期输入的 `Loop`，**TensorRT 会把它当 shape input**。body 里只有一个块，没得折 |

> `loop_body_repeats` 用到了子任务 1 的一个发现：**trip count 是编译期常量的 `for` 会被导出器展开**，
> 只有数据依赖的 trip count 才能保住 `Loop`。所以外层 `for` 用运行期的 `n`（保住 Loop），
> 内层 `for` 用常量 4（被展开，从而在 body 里造出 4 个重复块）。

## 已知缺口

这批样例是攒下来的，不是设计出来的，所以：

* 大部分合成模型用的是 shape 无关的一元/二元算子（`Relu`/`Add`/`Sum` 之类），
  好处是拓扑干净、不受 shape 推断干扰。shape 敏感算子由上面第三节的三个样例覆盖
  （`Transpose.perm` / `Concat.axis` / `Reshape` 的目标形状），
  但只覆盖了「属性 vs 输入」这一条轴；`Split`、`Gather.axis`、`Squeeze` 的
  `axes` 输入、动态 shape 下的 `Reshape(-1)` 都还没有用例；
* 没有量化模型（QDQ）、没有动态 shape、没有 fp16/bf16；
* `planted_block` 只存了一次采样，完整的随机生成器在
  [`../03-ParallelRepeat/stress_test.py`](../03-ParallelRepeat/stress_test.py)，
  可以按需要生成任意多个（`model_zoo.planted_block_with_answer()` 也直接暴露了它，
  并额外返回植入答案）；
* `transformer_6layer` 用 `nn.TransformerEncoder(layer, 6)` 生成，它用 `deepcopy` 克隆同一层，
  所以 **6 层权重完全相同**。对折叠没有影响（折叠只看结构），但拿它做 engine 体积对比会失真 ——
  [`../08-LoopBackend`](../04-LoopBackend/README.md) 因此自己重新随机化每层权重。
