# 子任务 5：torch 的两套 ONNX 导出器对「子图」的支持

## 问题

`OnnxVisualization` 这个工具存在的前提是「torch 导出的 ONNX 是完全展开的」。
那么反过来问一句：**torch 自己能不能导出带子模块的 ONNX？** 如果能，这个工具就没必要做了。

结论是**旧导出器能、新导出器不能**，而新导出器已经是默认。所以工具仍然必要，
而且随着 `dynamo=True` 铺开会更必要。

## 两套导出器

| 特性                                   | 旧系统 `dynamo=False`                     | 新系统 `dynamo=True`（PyTorch 2.9 起默认）    |
| -------------------------------------- | ----------------------------------------- | --------------------------------------------- |
| 底层机制                               | TorchScript tracing                       | `torch.export.ExportedProgram` + onnxscript    |
| 状态                                   | 每次调用都抛 `DeprecationWarning`         | 推荐方式                                       |
| `export_modules_as_functions`          | ✅ 支持，但文档标注 *Deprecated option*   | ❌ **静默忽略**（不报错、不警告）              |
| 模块层级子图（local function）         | ✅                                        | ❌                                             |
| 控制流子图（`If` / `Loop` 的子图属性） | ✅（`torch.jit.script` + Python `if`）    | ✅（`torch.cond`）                             |
| 模块层级信息                           | 直接体现在图结构里                        | 只留在节点 `metadata_props` 里                 |

## 两个范例

```bash
python3 main-legacy.py   # -> log-main-legacy.py.log
python3 main-dynamo.py   # -> log-main-dynamo.py.log
```

两个脚本用**同一个模型**（`Net` = 3 个 `Block`，每个 `Block` = Linear-ReLU-Linear-ReLU）
和同一份权重，各自跑 4 个 case，最后都用 onnxruntime 与展开版逐元素比对。

### `main-legacy.py`（旧系统）

| Case                | 主图节点 | function 数 | 子图节点 | 与 flat 最大误差 |
| ------------------- | -------: | ----------: | -------: | ---------------: |
| `flat`              |       12 |           0 |        0 |                0 |
| `function_selected` |    **3** |           1 |        0 |                0 |
| `function_all`      |    **1** |           3 |        0 |                0 |
| `control_flow_if`   |        5 |           0 |        4 |          （n/a） |

+ `export_modules_as_functions={Block}` → 主图 12 节点收成 **3 个 `Block` 调用节点 + 1 份
  `FunctionProto`**。三个调用点**共享同一份 body**，权重不是烤进 body 的，而是作为
  function 的额外输入在调用点传进去（`FunctionProto` 没有 `initializer` 字段，也不是闭包）：

  ```
  Function input  : ['onnx::Gemm_20', 'block_list.2.fc1.weight', 'block_list.2.fc1.bias',
                     'block_list.2.fc2.weight', 'block_list.2.fc2.bias']
  Call site name  : ['/block_list.0/Block', '/block_list.1/Block', '/block_list.2/Block']
  ```

+ `export_modules_as_functions=True` → 每个 `nn.Module` 都变成 function，于是**嵌套**成
  `Net` → `Block` → `Linear`，主图只剩 **1 个节点**。Netron 里看着最干净，
  代价是要点三层才能看到真正的算子。

+ **deprecation 的实际表现**：`export_modules_as_functions` 本身**不单独报警**，
  只有旧导出器整体抛 `DeprecationWarning`（提示 2.9 起新导出器已成默认）。
  所以「这个参数被废弃了」这件事在运行期是**看不见**的，只写在文档里。

+ `control_flow_if`：Python 的 `if` 在 tracing 下会被直接执行掉、只留下走到的那一支，
  必须 `torch.jit.script` 才能保住，导出为 `If` 节点的 `then_branch` / `else_branch` 两个子图。

### `main-dynamo.py`（新系统）

| Case                | 主图节点 | function 数 | 子图节点 | 与 flat 最大误差 |
| ------------------- | -------: | ----------: | -------: | ---------------: |
| `flat`              |       12 |       **0** |        0 |                0 |
| `function_ignored`  |       12 |       **0** |        0 |                0 |
| `metadata`          |       12 |       **0** |        0 |                0 |
| `control_flow_cond` |        3 |           0 |        4 |          （n/a） |

+ **`function` 一列全是 0**：新导出器在任何情况下都不会产出 local function。

+ **`export_modules_as_functions` 是静默忽略，不是报错**（`function_ignored`）。
  参数**仍在 `torch.onnx.export` 的签名里**，调用被接受，**不抛任何警告**，
  产物与不传该参数时**逐字节相同**：

  ```
  `export_modules_as_functions` still in the signature: True
  Warning mentioning `export_modules_as_functions`     : False
  Byte-identical to the export without the parameter   : True
  ```

  这是最需要注意的一点：一个从旧导出器移植过来的脚本**照跑不误，只是子模块悄悄没了**。

+ **模块层级信息去哪了：节点的 `metadata_props`**（`metadata`）。
  图是平的，但每个节点都还带着自己来自哪个模块：

  ```
  namespace                      : ': __main__.Net/block_list.0: __main__.Block/...'
  pkg.torch.onnx.class_hierarchy : "['__main__.Net', '__main__.Block',
                                     'torch.nn.modules.linear.Linear', 'aten.linear.default']"
  pkg.torch.onnx.name_scopes     : "['', 'block_list.0', 'block_list.0.fc1', 'linear']"
  ```

  用 `name_scopes[1]` 分组就能把 12 个平铺节点还原成 3 个 `Block`：

  ```
  block_list.0    ['Gemm', 'Relu', 'Gemm', 'Relu']
  block_list.1    ['Gemm', 'Relu', 'Gemm', 'Relu']
  block_list.2    ['Gemm', 'Relu', 'Gemm', 'Relu']
  ```

  **Netron 不会用这些信息来组织可视化**，但对本工具是个现成的提示来源（见下面「对本工具的影响」）。

+ **一个坑：降 opset 会把 metadata 全部抹掉。**

  ```
  {}                    -> opset=20, metadata key = [namespace, class_hierarchy, fx_node, name_scopes, stack_trace]
  {'optimize': False}   -> opset=20, metadata key = [同上]
  {'opset_version': 18} -> opset=18, metadata key = [同上]
  {'opset_version': 17} -> opset=17, metadata key = []          <-- 全没了
  ```

  原因不是 `optimize`（关掉也一样有），而是**版本转换**：目标 opset 低到 onnxscript 的
  version converter 处理不了时，会回退到 ONNX C API 的转换器，那一条路径**不搬运
  `metadata_props`**。想留住模块信息就别顺手写 `opset_version=17`。

+ `control_flow_cond`：`torch.cond` 正常导出为 `If` + 两个子图，`flag` 正负两个取值分别命中
  `Relu` / `Tanh` 分支，说明分支真的都在图里（Python `if` 在 `torch.export` 下会被
  specialize 掉，只留一支）。

## 结论：对本工具的影响

1. **新导出器不给模块层级子图，而且是默认。** 靠 torch 自己产出可读 ONNX 这条路在
   `dynamo=True` 下走不通，`OnnxVisualization` 这类**后处理 outlining** 是唯一选择。
2. **旧导出器能给，但不能依赖**：参数已标 deprecated、无替代品，且旧导出器整体在退场。
   顺带一提，它的产物正是 `00-ModelZoo` 里 `flat_mlp_as_function` 那种「输入模型本身已含
   local function」的来源 —— 主 README「抓到的正确性 bug」第 3 条就是被这种输入炸出来的。
3. **`metadata_props` 是一条没被利用的捷径。** 当前的挖掘算法完全从拓扑结构出发，
   不看 metadata。对 torch 导出的模型，`name_scopes` / `class_hierarchy` 可以当**种子**用
   （比 WL 播种靠谱得多，见 `03-PatternMiningFeasibility`）。没做，因为：
   (a) 只对 torch 导出的模型有效，TF / Paddle / 手写 ONNX 都没有；
   (b) 经 onnxslim 之类的预处理或降 opset 之后 metadata 可能已经不在了；
   (c) 模块边界不等于最优的 outline 边界（`Linear` 每个都单独成模块，折出来毫无意义）。
   结构挖掘是无条件可用的那条路，metadata 只能是锦上添花。
4. **控制流子图两套都支持**，这与主 README 里「`Loop` / `If` 的 body 里也要挖」
   （`--no-subgraph` 开关）对得上：子图是会真实出现的，不是假想情况。

## 环境

PyTorch 2.13.0a0+9186a08b2c.nv26.07、onnx 1.21.0（`nvcr.io/nvidia/pytorch` 镜像）。
两个脚本都只用 onnxruntime 做数值校验，不建 TensorRT engine ——
「带子模块的 ONNX 能不能过 TensorRT」在 [`01-SubgraphInONNX`](../01-SubgraphInONNX/README.md) 里已经测过（结论：都能）。
