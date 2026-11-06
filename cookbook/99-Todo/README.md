# 99-Todo

+ cookbook 的待办清单与调研笔记。

## 1. 待办 —— 还剩什么

只剩一项。**M4 和 M11 已于 2026-09-07 落地**；只剩 P5，而它卡在 Torch-TensorRT 的升级上，不在我们这边。

+ **M4** -> [`07-Tool/QDQPlacementAutotune/`](../07-Tool/QDQPlacementAutotune/README.md) —— 用真实
  TensorRT 延迟计时的 Q/DQ 放置搜索：14 次 engine 构建换来 **1.174 倍**（0.110 -> 0.090 ms）。同时
  记下了两个诚实的负面结果：在这么小的模型上，模式缓存**什么也没省**（开不开都是 14 次基准测试）；
  而且搜索是**随机的** —— 带缓存的 model-B 那一行一次读出 1.388 倍、另一次 1.149 倍，所以
  「有缓存 vs 无缓存」的延迟差**不能**被当成缓存的效果来读。
+ **M11** -> 并入
  [`03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT`](../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/README.md)
  成为 `case_autocast_bf16` + `case_node_sensitivity` —— 在一个从不溢出的模型上（opset 19 -> 22），
  BF16 比 FP16 **离 FP32 远 6.0 倍**；而 `data_max` 扫描显示有三个阈值产出**逐字节相同的图**，却报出
  不同的误差。

### 1.0 Engine 可视化：收敛到两套系统

当前有三套实现把 engine 的 layer-info JSON 变成图，计划是收敛到两套。对比表在
[`07-Tool/EngineVisualization/README.md`](../07-Tool/EngineVisualization/README.md)；这里记的是工作。

**V1. 控制流 —— 已完成。** `export_engine_as_onnx` 不再拒绝带 `Loop` 的 engine，并且把 `model-loop`
的**全部 8 条**依赖都表示了出来（6 条前向 + 2 条回边），而另外两个写出器只保留 5 条且一声不吭。两种
模式，都由 `07-Tool/EngineVisualization/export_as_onnx.py` 在 `model-loop.onnx` 和 `model-for.onnx`
上覆盖：

+ **默认**把回边保留为真实的环。`onnx.checker` 会拒绝它 —— ONNX 的定义要求单静态赋值 —— 但
  **Netron 能正确画出来**，而这些文件是用来读的、不是用来执行的。已用一个手工搭的三节点环状 ONNX
  验证过，该文件由测试现场重建而非提交进仓库。
+ **`b_break_cycle=True`** 把每一次写入重命名为一个 SSA 版本，并加入 `BackEdge` 标记节点，得到一个
  能通过 `onnx.checker` 的 DAG。任何建立在 `onnx_graphsurgeon` 之上的东西都需要它，否则会抛
  `Cycle detected in graph!`。不设为默认是因为它更难读：那些标记是死胡同，得按名字去查；而且去掉回边
  之后，图常常会裂成互不相连的几块。

不含控制流的 engine 在两种模式下逐字节相同。

仍未解决的是**结构化**形式：一个带真实子图 body 的 ONNX `Loop`，那样 Netron 就能把 body 折叠起来。
这件事比乍看起来难得多 —— **engine 里根本没有子图可供还原。** TensorRT 把循环编译成一条扁平的指令流，
带一个显式的 `cjmp` 层和一条回边，所以要产出 ONNX `Loop` 就等于反编译：要从分支推断 body 的边界，还要
把 recurrence 映射回 carried dependency。嵌套循环、多出口、body 里再嵌一个 `If`，每一条都让事情更复杂；
而一个**细微错误的重建比我们现在输出的那个诚实的环更糟**，因为它看起来是对的。不排期。


### 1.1 下一步：TensorRT 11.2 升级

**2. `04-Feature/RefitObserver` 应该可以甩掉它外挂的那套 TensorRT。** 它现在通过 `TRT_OBSERVER_PATH`
对着一个解包的 **11.3** tarball 编译，因为 `IRefitterObserver` 是 11.2 才有的、而系统安装的版本更老。
在 11.2 上系统头文件应该就够了 —— 把 `TRT_OBSERVER_PATH` 这层间接和它 `unit_test.yaml` 里的
`LD_LIBRARY_PATH` 要求一起退役掉。仍然是 **C++ only**：11.3 的 cp312 wheel 里没有 `set_refit_observer`，
所以在假定存在 Python 绑定之前，先去查 11.2 的 wheel。

**3. 复查 API 移除导致的修复。** 有两个例子在 11.0 移除弱类型时挂了，已在那轮扫描中迁移。11.2 可能
移除更多：
  + `02-API/Layer/QDQStructure/main.py` —— `BuilderFlag.INT8`（已移除），以及 `add_quantize` /
    `add_dequantize` 现在要求显式的 `output_type`；
  + `07-Tool/trtexec/main.sh` —— `--fp16` / `--int8` / `--best` / `--precisionConstraints` 全部被
    拒为 "Unknown option"。
  `trt-strong-typing-migration` skill 是参考。它的 `scripts/migrate.py --write`
  **会重排整个文件并抹掉 SPDX 头**（它是通过 AST unparse 往返的），所以拿它看 diff 和结论，然后手工改。

**4. 每一次都要先重装 pip 附加包**，见 §1.2。

**5. P5** 仍然卡在 Torch-TensorRT 而不是 TensorRT 上，这次升级不会推动它。

### 1.2 环境供给：容器在会话之间会重置

`tensorrt_rtx`、`tensorrt-lean` 和 `tensorrt-dispatch` **不会**在重置后幸存，每次全量扫描前都必须重装：

```bash
pip install tensorrt_rtx tensorrt-lean tensorrt-dispatch
```

Graphviz（`/usr/bin/dot`，apt 装的）到目前为止都活下来了。**一次全绿的扫描并不能证明附加包装上了**：
那 16 个 `optional-package` 用例会守卫自己的 import 并带着一行 `[SKIP]` 以 0 退出，所以一次没装附加包
的运行照样报 257/257。事后要逐个 log 去 grep `^[SKIP]` 这个标记来核实 —— 不要 grep "skip" 这个词，它会
在 `10-TensorRT-RTX/01-ComputeCapability` 的正常输出里出现（"skip setting"），造成假阳性。

## 2. 值得携带的发现

### 2.1 测量的教训

有三个错误反复出现到值得专门写一次。每一个都产出了**自信而错误**的结果，看起来和真实测量一模一样。

**热降频。** 这台机器上的 GPU 0/1 在持续负载下会到 88-91 °C，并从约 1900 MHz 掉到 **352 MHz**
（`clocks_throttle_reasons.active = 0x20`，软件热降频，功耗 437 W 而上限是 1000 W）。同一个 engine
在凉的 GPU 上测出 2.8 ms，在热的上测出 11.0 ms；有一次扫描**报出 111.8 ms 而正确答案是 10.9 ms**，
还把一个看似的交叉点挪动了一倍。`08-Advance/ContextParallelism` 现在会挑最凉的 GPU、等它们凉下来，
并在每个延迟旁边打印 SM 时钟。**这台机器上任何背靠背取得的延迟数字，如果没有同时记录时钟，都值得怀疑。**

**编排开销淹没被测量。** `08-Advance/MultiTask` 和
`06-DLFrameworkTRT/Torch-TensorRT/DistributedInference` 一开始都把并发测成了**退步**（0.35x、0.65x、
0.79x），因为每次迭代两道 Python 屏障的开销比被计时的 GPU 工作还大。修法是把工作量放大，并改测吞吐
（屏障只付一次）而不是每轮延迟。`MultiTask` 现在显式测量屏障的地板值并打印在每个结果旁边。
**当每次迭代的工作量逼近它周围同步的开销时，这个基准测试就不再是关于 TensorRT 的了。**

**被探针选择做局的对比。** `04-Feature/OnnxPTQMethod` 曾经只在「离群点被**种进去**的那份数据」上比较
entropy 与 max 校准 —— 在那种数据上 `max` 按构造必赢。加上一个普通探针之后**排名反转了**。另外，
`04-Feature/LowBitQuantization` 曾经在数据显示相反结论的情况下断言「块缩放胜过位数」。
**只要一个对比能被输入的选择做局，就要把它的两面都测。**

还有一个小一点的：`08-Advance/ResourceProbe` 曾经断言它当时恰好观察到的那个具体的析构症状。那个症状是
非确定性的（要么以 0 退出但记录 5 条错误，要么静默 SIGSEGV），所以断言是不稳定的。**断言不变量，不要
断言观察到的现象。**

### 2.2 我们正在扛着的上游缺陷与绕行方案

这些都**没有修好** —— 例子只是绕开它们。每一条都应该在下一次 ModelOpt / Torch-TensorRT / TRT 升级后
重测，第一条应当上报上游。

| 是什么                                                                                                                                                                                                                                                                                                | 在哪里咬人                                                  | 例子改用什么做法                                                                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **ModelOpt 的 Q/DQ 自动调优器在 TRT 11 上用不了它自己的默认基准后端。** `autotune/benchmark.py:361` 读 `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`，而显式 batch 成为唯一模式后该枚举已被移除。那个 `AttributeError` 被吞掉，只报成 `Failed to initialize TensorRT benchmark`，什么都没说          | `07-Tool/QDQPlacementAutotune`                              | 传 `--use_trtexec` 走 `TrtExecBenchmark` 那条路，可以工作。**应与 `plugin_creator_list` 一并上报 ModelOpt**                               |
| **ModelOpt 0.44.0 与 TRT 11 不兼容。** TRT 11 移除了 `IPluginRegistry.plugin_creator_list`，而 ModelOpt 仍在读它                                                                                                                                                                                      | `05-Plugin/ONNXPTQWithPlugin`                               | 自带 `apply_modelopt_trt11_shim()`，把该属性作为 `all_creators` 的别名加回去。**应上报 ModelOpt** —— 这个 shim 是为了让例子能跑，不是修复 |
| **`trt_plugins=` 在这里不可用。** 即使过了 shim，ONNX Runtime 在**加载 session 时**就会拒绝含 TensorRT 自定义算子的图；`calibration_eps=["trt"]` 也没用。三条报错，没有一条说出自己的原因                                                                                                             | `05-Plugin/ONNXPTQWithPlugin`                               | 校准一个可用 ONNX 表达的替身，再把得到的 Q/DQ 移植到 plugin 图上                                                                          |
| **低于 8 bit 的格式根本到不了 engine。** Torch-TRT 的转换器直接拒绝 NVFP4 / MXFP8 / INT4-AWQ；ONNX 路线对**全部五种**格式都失败；把 ModelOpt 量化过的 torch 模块导出到 ONNX 对每种格式都失败；INT4 weight-only 能量化但在 TRT 构建时因块大小整除失败（`inputSize % scaleSize == 0`，3211264 / 25600） | `04-Feature/LowBitQuantization`、`04-Feature/OnnxPTQMethod` | 把负面结果原样发布。**这套栈上只有 INT8 和 FP8 能到达 engine**                                                                            |
| **TRT-LLM 的 plugin 拒绝 CUDA 13**，所以**张量并行**演示不了（仅限 TP；同目录的数据并行在 CUDA 13 / 4×B200 上正常，2026-10-14 复测四卡逐位一致）                                                                                                                                                                                                                                              | `06-DLFrameworkTRT/Torch-TensorRT/DistributedInference`     | 只做数据并行（4 卡 3.11x）；TP 被记录为受阻                                                                                               |
| **`GroupNormalizationPlugin` 已被移除**（TRT ≥ 10.7，Blackwell）                                                                                                                                                                                                                                      | `07-Tool/OnnxGraphSurgeon/14_fold_exporter_subgraphs.py`    | 改为发出原生的 ONNX `GroupNormalization`（opset 21）而不是 plugin 节点                                                                    |
| **E8M0 块缩放构建不起来。** 用一个 2x2 测试隔离过 —— 裸的 Q/DQ 对能正常构建，**E8M0 这个 scale 才是阻塞点**；FP4 的块 QDQ 则被直接拒绝                                                                                                                                                                | `02-API/Layer/QDQStructure/main.py`（`case_block_*`）           | 作为实测边界记录在案                                                                                                                      |

### 2.3 静默失败 —— 这一轮最有价值的产出

这些没有一个会抛异常。每一个都会给出一个看似合理的数字和一个错误的答案。

+ **`refit_module_weights` 接受结构不同的模型。** 用一个 8 层的模块去 refit 一个 6 层的 engine 是
  **被接受的**；之后这个 engine 与该模块的**前 6 层完全一致**（差值 0.0），与它真正的输出则毫无关系。
  多出来的层被一声不吭地丢掉。这是本页最危险的一条 —— `06-DLFrameworkTRT/Torch-TensorRT/Refit`。
+ **`use_explicit_typing=True` 与 `enabled_precisions=...` 同时出现会被静默接受**，而不是被拒绝。
  我预测会报错，预测错了 —— `06-DLFrameworkTRT/ModelOptimizer`。
+ **给 shape tensor 绑一个设备指针是 SIGSEGV，不是报错。** shape tensor 在主机侧读取，而没有任何地方
  检查这个指针 —— `02-API/ShapeTensor`（该用例在子进程里跑，正是因为它会杀掉进程）。
+ **`MAX` / `MIN` 归约会静默丢掉 NaN**，而 NumPy 会传播它（8.0 / 5.0 对 `nan`），所以 max 归约
  **不是**一个 NaN 检测器 —— `04-Feature/CornerCase`。
+ **TensorRT 可以记录 5 条析构错误却仍然 `exit 0`**，当它的 CUDA 上下文先被销毁时 —— 对任何基于退出码
  的检查都不可见。而且症状是非确定性的：有时是带错误的 exit 0，有时是静默 SIGSEGV ——
  `08-Advance/ResourceProbe`。
+ **相同的权重在 plan 里会被去重**，把一个 18 MiB 的 engine 压成 2 MiB，于是任何建立在「重复权重」之上
  的加载/流式测量都失去意义 —— `08-Advance/MultiDevice`。
+ **`polygraphy run --trt --fp16` 在 `POLYGRAPHY_USE_TENSORRT_RTX=1` 下会丢掉该标志并通过。**
  同样的请求走 `CreateConfig(fp16=True)` 则会抛异常。只看退出码的检查分不出「标志被丢掉」和「标志被
  执行」—— `07-Tool/Polygraphy/TensorRTRTX`。
+ **TF32 默认开启，是手工搭建的网络「对不上」时的头号嫌疑人。** 它在一个正确的网络和 torch 之间制造了
  2.643e-04 的差距，而 float32 下是 8.941e-08，**2956 倍**的差异，读起来和一个转置错的权重块一模一样
  —— `02-API/Network`、`04-Feature/RefitObserver`。
+ **`trt.Weights` 包的是一个指针。** 值是在 `build()` 时读的，不是在 `add_*()` 时。在这中间改动数组会
  静默改变 engine；让它被回收则是一次没有任何提示的 use-after-free —— `02-API/Network/weight_transport.py`。
+ **TensorRT-RTX 的 `set_compute_capability` 在 `num_compute_capabilities` 未赋值时返回 `False` 而不是
  抛异常**，随后构建会悄悄以当前设备为目标 —— `10-TensorRT-RTX/08-DeployTimeRuntimeConfig`。
+ **`IRuntimeConfig` 不会让它的 runtime cache 保活。** 在 `set_runtime_cache()` 之后丢掉最后一个 Python
  引用，进程会段错误，静默地 —— 同一个例子。
+ **两种误差模式可以有相同的 max-abs-diff**，而其中一个是 80/80 个元素都错、另一个只错 1/80。单一的
  标量容差分不出它们 —— `07-Tool/Polygraphy/DebugWorkflow`、`07-Tool/AccuracyCheck`。

## 6. 参考

### 6.1 NVIDIA 生态仓库

+ 我们在 2026年10月12日调研了下面几个 repo，将其中有用的工具吸纳进了 cookbook。后续如果再次考察这些 repo，就不用再看他们在该日期之前的提交了

| 仓库                                                                                                    | 状态                             |
| ------------------------------------------------------------------------------------------------------- | -------------------------------- |
| [NVIDIA/healthcare-on-tap-TRT-TRITON-demo](https://github.com/NVIDIA/healthcare-on-tap-TRT-TRITON-demo) | 不提取，太老（2021-02）          |
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)（旧名 TensorRT Model Optimizer）    | 提取完成                         |
| [NVIDIA/object-detection-tensorrt-example](https://github.com/NVIDIA/object-detection-tensorrt-example) | 不提取，太老（2019-06）          |
| [NVIDIA/Stable-Diffusion-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT)     | 部分提取（2024-03）              |
| [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)                                 | 部分提取                         |
| [NVIDIA/TensorRT-Incubator](https://github.com/NVIDIA/TensorRT-Incubator)                               | 部分提取                         |
| [NVIDIA/tensorrt-laboratory](https://github.com/NVIDIA/tensorrt-laboratory)                             | 不提取，太老（2020-09）          |
| [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)                                           | 部分提取                         |
| [NVIDIA/TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect)                       | 部分提取                         |
| [NVIDIA/TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI)                             | 暂时不做（没有TensorRT-RTX SDK） |
| [NVIDIA/TensorRT-RTX](https://github.com/NVIDIA/TensorRT-RTX)                                           | 部分提取                         |
| [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)                                                   | 部分提取                         |

### 6.2 其他 TensorRT 生态仓库

| 仓库                                                             | 在 cookbook 中的状态      |
| ---------------------------------------------------------------- | ------------------------- |
| [pytorch/TensorRT](https://github.com/pytorch/TensorRT)          | 部分提取                  |
| [tensorflow/tensorrt](https://github.com/tensorflow/tensorrt)    | 部分提取，太老（2025-02） |
| [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)          | 未调研                    |
| [jetson-inference](https://github.com/dusty-nv/jetson-inference) | 未调研                    |
