# 子任务 1：含子模块的 ONNX 与 TensorRT 的兼容性

## 问题

从 torch / TF / Paddle 导出的 ONNX 是**完全展开**的：一个 32 层的 transformer 会被摊平成
几千个 MatMul / Softmax / Reshape / Transpose 节点。我们想知道，如果 ONNX 里保留「子模块」
这一层抽象，TensorRT 还认不认、还能不能编译和运行。

## ONNX 中表达「子模块」的两种机制

| 机制 | 载体 | 谁会产生它 | 语义 |
| --- | --- | --- | --- |
| **Local function** | `ModelProto.functions`（`FunctionProto` 列表）+ 主图里 `domain` 为自定义域的节点 | `torch.onnx.export(..., export_modules_as_functions={Block})` | 纯粹的**宏 / 内联展开**，没有运行时语义，每次调用可以带不同的权重 |
| **Sub-graph 属性** | `Loop` / `Scan` / `If` 节点的 `GraphProto` 属性 | TorchScript 的 `for` / `while` / `if`（trip count 必须是运行期量） | 真正的**控制流**，body 只有一份，迭代之间靠 loop-carried dependency 串起来 |

> 还有 ONNX Runtime 特有的 `com.microsoft` fused op、以及 opset 里的 standard function（如
> `Gelu`、`LayerNormalization` 有 function body），但那些是「算子定义自带 body」，不是用户可以
> 自由构造的子模块。

## 实验

* `main.py` —— 用同一份 PyTorch 权重生成 5 种 ONNX，逐个过 TensorRT parser + builder + runtime，
  和展开版的输出做逐元素比对。
* `02-scaling.py` —— 把重复块数 N 从 2 扫到 256，对比「展开」和「折叠成一个 Loop」两种表示在
  ONNX 节点数、TRT layer 数、parse/build 耗时、engine 体积、推理延迟上的差别。

运行：

```bash
python3 main.py         # -> log-main.py.log
python3 02-scaling.py   # -> log-02-scaling.py.log
```

## 结果 1：TensorRT 全部都能编译和运行

`main.py`（N_BLOCK=4, N_C=8, N_B=2，TensorRT 11.1.0.106）：

| Case | 主图节点 | function 数 | 控制流节点 | 子图节点 | TRT layer | 与展开版最大误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `flat` 展开基线 | 16 | 0 | 0 | 0 | 48 | 0 |
| `local_function` | 4 | 1 | 0 | 0 | **48** | 0 |
| `loop_static_trip` | 16 | 0 | 0 | 0 | 36 | (权重不同，见下) |
| `loop_dynamic_trip` | 2 | 0 | 1 | 5 | **25** | (权重不同，见下) |
| `loop_stacked_weight` | 1 | 0 | 1 | 11 | **34** | 0 |

结论：

1. **Local function：parser 接受，但会被完全内联。** 主图只有 4 个 `__main__::Block` 节点，
   parse 出来仍是 48 个 TRT layer，和展开版一模一样。parser 还会打印
   `A node named Gemm_0 already exists, the output tensors of this new instance will not be queryable`
   —— 这正是内联时节点重名的证据。
   → 好处只在 **ONNX 文件本身**（可读性、节点数、Netron 里能折叠），对 TRT 的 build 时间和
   engine 没有任何影响。
2. **Loop：parser 会真的映射成 `ILoopLayer`。** 网络里出现
   `TRIP_LIMIT` / `RECURRENCE` / `LOOP_OUTPUT` 这些 layer type，body 只实例化一次。
3. **一个反直觉的坑：trip count 是编译期常量的 `for` 循环会被导出器直接展开。**
   即使套了 `torch.jit.script`，`for _ in range(self.n)`（`self.n` 是 int 属性）在
   `loop_static_trip` 里得到的 ONNX **一个 `Loop` 节点都没有**，16 个节点全是摊平的。
   想拿到 `Loop`，trip count 必须是运行期张量（`loop_dynamic_trip` 里 `n` 是一个 int64 标量输入）。
4. **`loop_dynamic_trip` 的 `n` 在 TRT 里是 shape input tensor**，必须用
   `profile.set_shape_input()` 设置、用 `context.set_tensor_address()` 绑定 host 内存
   （即 cookbook 里的 `TRTWrapperShapeInput`），不能当普通 device 输入用。
   n = 1 / 3 / 5 都能跑通，且与 onnxruntime 完全一致。
5. **`loop_stacked_weight` 是最关键的一例**：它把「4 个各自独立权重的 block」重写成
   *一个* `Loop`，body 里用 `Gather(W_all, iter)` 取当前迭代的权重（`W_all` 形状 `[N, C, C]`）。
   TRT 编译通过、运行结果与展开版**逐比特相同**。
   这正是子任务 3 想自动化的那个变换，说明目标表示在 TRT 上是可行的。

## 结果 2：折叠成 Loop 的收益与代价

`02-scaling.py`（N_C=256, N_B=8，**B200**，2026-09-06 重测）：

```
   N |  Node(F) Node(L) |  Layer(F) Layer(L) |  Parse(F)s Parse(L)s |  Build(F)s Build(L)s |   Eng(F)MB  Eng(L)MB |   Lat(F)ms  Lat(L)ms |    MaxDiff
   2 |       13      12 |        25       34 |      0.003     0.003 |       9.25      4.37 |        1.1       1.1 |      0.021     0.042 |  1.426e-05
   4 |       25      12 |        49       34 |      0.003     0.004 |      12.15      4.53 |        2.1       2.1 |      0.031     0.075 |  1.192e-07
   8 |       49      12 |        97       34 |      0.004     0.005 |      15.21      4.13 |        4.1       4.1 |      0.059     0.139 |  1.816e-05
  16 |       97      12 |       193       34 |      0.006     0.007 |       9.32      3.90 |        8.1       8.1 |      0.117     0.269 |  1.192e-07
  32 |      193      12 |       385       34 |      0.106     0.010 |      13.91      4.07 |       16.2      16.2 |      0.232     0.520 |  5.960e-08
  64 |      385      12 |       769       34 |      0.220     0.225 |      23.38      4.23 |       32.4      32.2 |      0.462     1.029 |  1.127e-05
 128 |      769      12 |      1537       34 |      0.430     0.396 |      41.90      4.36 |       64.7      64.4 |      1.008     2.061 |  1.043e-07
 256 |     1537      12 |      3073       34 |      0.906     0.835 |      79.39      4.29 |      129.4     128.6 |      2.326     4.141 |  1.764e-05
```

（F = flat 展开，L = 单个 Loop + 堆叠权重；parse 时间受磁盘 I/O 影响，抖动较大，只看量级）

* **图规模**：Loop 版恒为 12 个 ONNX 节点 / 34 个 TRT layer，与 N 无关；展开版线性增长。
* **build 时间**：展开版 9.3 s → **79.4 s**（N=256）且还在涨；Loop 版恒为 **~4.2 s**。
  这里 ~4 s 是本机的固定开销。**这一条在 B200 上比 H100 上强烈得多**：H100 是 6.3 s → 12.6 s
  （2.0x），B200 是 9.3 s → 79.4 s（**8.6x**）。折叠成 `Loop` 省下的 build 时间从「几秒」变成了
  「一分多钟」。
* **engine 体积**：两者基本相同 —— 权重量没变，只是换了个存放形式。**折叠不省显存**。
* **推理延迟**：**Loop 版稳定慢 ~2 倍**。原因是 body 里的权重变成了运行期 `Gather` 的结果，
  不再是常量：TRT 无法做常量折叠、无法按权重挑选专用 kernel、无法跨迭代做层融合，
  而且每次迭代都要真的搬一次权重。
* **数值**：**不再逐比特相同**。H100 上所有 N 的 MaxDiff 都是 `0.000e+00`；B200 上在
  `1.19e-07` 到 `1.76e-05` 之间跳动，且与 N 无单调关系。量级仍是 fp32 累加顺序的差别、远小于
  fp16 分辨率，但「两种表示逐比特一致」这句话在 B200 上是错的 —— 两边挑到的 kernel 不同了。

## 对后续子任务的结论

1. 「把重复子图折叠成子模块」在 TRT 上**技术可行**，两种表示都能编译运行。
2. 如果目标是 **ONNX 文件本身的可读性 / 体积 / 工具链处理速度**，用 **local function** 即可，
   对 TRT 完全透明（会被内联），零风险、零性能损失。
3. 如果目标是 **缩短 TRT build 时间 / 缩小网络规模**，必须用 **`Loop`**，
   但要接受 ~2x 的推理延迟代价和权重被迫「去常量化」。**在 B200 上这个 trade-off 更划算**：
   build 从 79 s 降到 4 s，而延迟代价仍是 ~2x。这是一个明确的 trade-off，
   适合「build 时间难以忍受、latency 不敏感」或者「权重本来就要 streaming」的场景。
4. 折叠成 `Loop` 要求被折叠的各次重复**拓扑完全同构、且非权重的属性（axis、perm、eps…）完全一致**，
   差异只能体现在可以堆叠成 `[N, ...]` 的权重上。这条约束直接决定了子任务 3 里模式匹配的
   「等价」判据要怎么定义。
