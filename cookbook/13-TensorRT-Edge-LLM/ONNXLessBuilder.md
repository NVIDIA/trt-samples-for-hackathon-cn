# A3：绕开 ONNX 直接建 engine —— 讨论文档

> **这是一篇供讨论的文档，不是一个范例。** 按约定先写出来给你看，决定要不要做、做成什么样之后再动代码。
> 来源是 [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) 的
> `docs/source/user_guide/getting_started/direct-engine-builder.md` 与 `experimental/`，候选条目见
> [`../99-Todo/trtellm.md`](../99-Todo/trtellm.md) 的 A3。

## 一、上游那条路是什么

Edge-LLM 有两条前端，**共用同一套 C++ 部署运行时**：

| 路径 | 命令 | 状态 |
| :-- | :-- | :-- |
| 受支持 | `tensorrt-edgellm-export`（导出 ONNX）→ 各组件的 C++ builder | 正式 |
| 实验 | `tensorrt-edgellm-build`（**不写也不解析 ONNX**） | 实验性 |

实验路径的描述是：

> `tensorrt-edgellm-build` compiles every TensorRT engine required by a checkpoint without writing
> or parsing ONNX. The command reads checkpoint configuration and tensor metadata directly, selects
> model-owned Python component definitions, and writes **the same engine and runtime-artifact
> layout** consumed by the existing C++ executables.

关键在最后半句：**产物布局完全一致**。也就是说这不是另起一套部署，而是把「前端怎么产生 engine」这一步
换掉，后端一个字节都不用动。上游自己仍然把 ONNX 路径标为"受支持"，直接路径标为"实验性"。

## 二、为什么这件事对 cookbook 有意义

**我们一直在做的就是这件事，只是从来没把它当成一个立场。**

cookbook 的绝大多数例子用的是 `build_mnist_network_trt(tw)` —— 直接调 `network.add_convolution_nd`、
`add_activation`、`add_matrix_multiply` 把网络搭出来，从不经过 ONNX。而业界的默认答案是
「PyTorch → ONNX → TensorRT」，我们的 `03-Workflow/` 也是这么教的。

于是产生一个我们从未回答的问题：**既然有 ONNX 这条标准路径，为什么 `02-API/Layer/` 里六十多个例子
全都在手搭网络？** 目前的隐含答案是「因为要演示单个层的 API」，这对教学成立，但读者会合理地推断
「生产里不这么干」。

Edge-LLM 提供了一个**生产项目绕开 ONNX 的真实论证**，正好给这条路线提供背书。这是 A3 的全部价值
——它是一篇立场文档的素材，不是一段可搬的代码。

## 三、绕开 ONNX 能得到什么、失去什么（基于上游文档与我们已有的实测）

| | ONNX 路径 | 直接构建 |
| :-- | :-- | :-- |
| 算子覆盖 | 受 parser 支持范围限制，不支持就要写 plugin 或改图 | 只受 TensorRT API 限制 |
| 自定义算子 | 要注册 ONNX schema + 导出翻译 + plugin，见 `05-Plugin/ONNXParserWithPlugin` 的 `case_pytorch_to_onnx_to_trt` | 直接 `network.add_plugin_v3`，少两层 |
| 中间产物 | 一个可检视、可用 Netron 看、可用 onnx-graphsurgeon 改的文件 | 没有中间产物 |
| 调试 | 可以在 ONNX 层二分（`07-Tool/OnnxGraphSurgeon`） | 只能在 TensorRT network 层二分（`07-Tool/NetworkPrinter`） |
| 权重 | 烤进 ONNX 或走 external data | 直接从 checkpoint 读，可以省掉一次完整复制 |
| 大模型的代价 | 导出一次 ONNX 就是一次全量权重落盘 | 没有这一步 |
| 生态 | ORT、polygraphy、各类工具都能消费 | 无 |
| 失败模式 | parser 报 "unsupported operator"，位置明确 | API 调用点报错，位置更明确但需要自己维护映射 |

**对边缘设备而言最后两行是决定性的**：一个 30B 模型导出 ONNX 意味着几十 GB 的落盘和再读取，而
Jetson 上既没有那个磁盘预算也没有那个时间预算。这解释了为什么是 **Edge**-LLM 而不是别的项目先做
这件事。

## 四、我们可以做成什么样（三个选项，我倾向 B）

### 选项 A：一篇纯文档

把第二、三节扩写成 `03-Workflow/README.md` 的一节，或者 `11-TensorRT-Model-Connect/Glossary/`
旁边的一页「两条路径怎么选」。零代码，零维护成本。

**问题**：cookbook 的风格是「能跑的东西 + 实测数字」，一页纯论述和周围格格不入。

### 选项 B：一个对照例子（推荐）

新建 `13-TensorRT-Edge-LLM/TwoFrontends/`，用**同一个 MNIST 模型**走两条路，然后逐项对比：

```txt
case_via_onnx     : torch -> onnx -> parse -> engine
case_direct       : 从同一份 checkpoint (model-trained.npz) 直接 network.add_* -> engine
case_compare      : 两个 engine 的输出、层数、层名、engine 字节数、构建耗时、中间产物大小
```

可以测到的真实数字：

+ **两个 engine 的输出是否逐比特相同** —— 大概率不同，因为 parser 会做常量折叠和算子融合，而手搭
  网络给的是另一种层序。这本身就是个好结论。
+ **层名完全不同** —— ONNX 路径带 `node_conv2d` 这类来自导出器的名字，手搭路径带我们自己设的
  `Convolution1`。这解释了 `07-Tool/EngineVisualization` 里为什么层名对不上 ONNX。
+ **中间产物**：ONNX 路径多一个 `.onnx` 文件，可以报它的字节数 —— 在 MNIST 上微不足道，但可以据此
  外推说明为什么 30B 模型上这一步是决定性的。
+ **构建耗时** —— 含不含 parse 的差别。

这些全都在现有素材上可测，不需要任何下载，且和我们已有的 `01-SimpleDemo`（同一个网络两种搭法）
形成呼应而不是重复 —— `01-SimpleDemo` 比的是"API 建网 vs 解析 ONNX"的**写法**，这里比的是
**两条路径的产物和代价**。

**成本**：一个目录，三个 case，大约 200 行。

### 选项 C：照搬上游的组件化构建器

把 Edge-LLM 那套「模型自带 Python 组件定义，builder 按 checkpoint 元数据选组件」的结构复刻一个
小型版本。

**不建议**。那是一套框架设计，和 cookbook「一个目录讲一件事」的结构冲突；而且它真正的价值在 136 个
模型的规模上才显现，MNIST 上只会显得过度设计。

## 五、需要你拍板的三件事

1. **选 A / B / C 哪个？** 我倾向 B。
2. **如果选 B，放哪里？** `13-TensorRT-Edge-LLM/TwoFrontends/` 还是 `03-Workflow/` 下面？我倾向前者
   —— 它的来源是 Edge-LLM，放在一起便于以后追溯；但它讲的是通用问题，放 `03-Workflow/` 更容易被
   需要的人找到。
3. **要不要带上权重这一维？** 上游强调「运行时读原始 checkpoint，初始化时准备一次最终布局的权重」。
   我们可以顺带演示 `trt.Weights` 直接包 numpy 数组、不经 ONNX initializer 的做法 —— 但那和
   `02-API/Network/weight_transport.py` 已有的内容部分重叠（包括那条「`trt.Weights` 包的是指针，
   值在 `build()` 时才读」的坑）。

## 六、一条限制，先说在前面

上游这条路径**自己标注为实验性**，受支持的仍是 ONNX 路径。所以无论做成什么，结论都不能写成
「绕开 ONNX 更好」，只能写成「这是两条路径各自的代价，以及什么时候第二条更划算」。
`03-Workflow/` 作为 ONNX 路径的主干不受影响。
