# TensorRT-RTX-EP-ABI 可吸纳内容清单

对 [`NVIDIA/TensorRT-RTX-EP-ABI`](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) 的调研，找出值得搬进
cookbook 的材料。

调研版本：commit `afe88d1`（tag `rel-0.4.2`，2026-09-21），182 个文件 / 3.9 MB。Apache-2.0，每个文件
都带 `SPDX-License-Identifier: Apache-2.0`，与 cookbook 同许可证，带署名改编没有问题。注意该仓库声明
**不接受外部贡献**，所以修复要走 NVIDIA 内部渠道而不是提 PR。

## 状态，2026-10-12：全部暂不做

曾经按 wili 的决定做过一个"编译 so + 在 onnxruntime 中跑一个最简单的模型"的端到端例子
（`../12-TensorRT-RTX-EP-ABI/`），**该目录已于 2026-10-12 整体清理**，本仓库的所有条目暂不做。
概况与进展记在 [`README.md`](README.md) 的 §3.8。

| 档 | 条目 | 状态 |
| --- | --- | --- |
| — | 端到端：编译 .so + ORT 中注册并跑模型 | **已清理**，例子目录不再存在 |
| A | A1 / A2 / A3 ONNX→TensorRT 兼容性重写 pass | **暂不做**，保留在本清单中待议 |
| B | B1 / B2 / B3 / B4 | **暂不做**。B1 的"每形状一个 CUDA graph"已由另一条线落地，见 `08-Advance/CudaGraph` 的 `case_graph_per_shape` |
| C | C1 / C2 / C3 / C4 / C5 | **暂不做** |
| D | 否决项 | **不做** |

**被清理的原因是供给而不是技术判断。** 那个例子的第 1/2/3/5 步（前置检查、clone、取 ONNX Runtime
SDK、注册并跑模型）都验证通过，唯独第 4 步构建跑不成：

+ **TensorRT RTX 的 C++ SDK 无法自动获取** —— developer.nvidia.com 需登录，
  `developer.download.nvidia.com/compute/tensorrt-rtx/redist/` 是 404。
+ **`pip install tensorrt_rtx` 不能替代** —— 实测装上 1.6.1.120 之后全盘递归搜 `NvInfer*.h` 零命中，
  只有 `libtensorrt_rtx.so.1` 和 `libtensorrt_onnxparser_rtx.so.1`；PyPI 上也没有头文件专用包
  （`tensorrt-rtx-headers` / `tensorrt_rtx_headers` / `tensorrt-rtx-dev` 全 404）。而
  `-DTRT_RTX_ROOT=` 要的是 `include/` 加 `lib/`。
+ **`NVIDIA/TensorRT-RTX` 仓库也不是 SDK** —— 那是样例仓库，不含 `NvInfer*.h`，它自己的 CMake 还在等
  别人提供 `TRTRTX_INCLUDE_DIR`。
+ 即便拿到 SDK，**本机 B200 也是错的硬件** —— TensorRT-RTX 面向 Ampere 及以后的 RTX 卡。

等哪天手上有 RTX 卡和 SDK 了，从本清单重新开始即可。

## 这个仓库是做什么的

它是 **TensorRT-RTX 的 ONNX Runtime execution provider**，以独立插件
`onnxruntime_providers_nv_tensorrt_rtx.{dll,so}` 的形式发布。它实现的是 ORT 1.23.0 引入的
**EP ABI**（`OrtEpFactory`、`OrtEp`、`OrtNodeComputeInfo`、`OrtDataTransferImpl`），因此**只依赖
ORT 的 SDK 而不需要和 ONNX Runtime 一起编译**——这正是这个仓库存在的理由：在 EP ABI 之前，写一个 EP
必须在 ORT 树内构建。

一句话说清它的形状：ORT 把一个子图交给它 → 转成 ONNX proto → 跑一组重写 pass → 交给 TensorRT-RTX 的
ONNX parser → 建 engine → 还给 ORT 一个可调用对象。所以它本质上是**一份生产级、开源的"ONNX 进、
TensorRT engine 出、然后跑起来"的完整范例**，和 `03-Workflow/` 教的是同一条流水线，但带着真实客户端
部署会撞上的每一个边角情况。

和我们跟踪的企业版 TensorRT 有三点定位差异：

+ **面向客户端而非数据中心**：RTX 30xx 以上，约 200 MB 包体，在终端机器上 JIT 建 engine（秒级），
  engine 缓存可跨不同 RTX 卡复用。
+ **Windows 是一等公民**，某些地方甚至是首要平台：D3D12/Vulkan interop、CiG（CUDA in Graphics）、
  对所加载 DLL 的 Authenticode 签名校验。
+ **用的是 TensorRT-RTX** 而非企业版 TensorRT。我们这边只有 `10-TensorRT-RTX/` 一个目录同主题。

即便我们不写 EP，它仍然值得读：`src/` 那 21000 行基本上是"把真实 ONNX 喂给 TensorRT parser 会出什么
问题"的答案集，每一条都带着修法和一句"它故意不修什么"。

| EP 版本 | ORT | TRT RTX | 新增 |
| --- | --- | --- | --- |
| 0.1 | 1.24.0+ | 1.4.x | Windows |
| 0.3 | 1.25.0+ | 1.5.x | Linux、weight streaming、CiG interop、ORT 版本协商 |

## A 档 —— ONNX→TensorRT 兼容性 pass

这是整个仓库最值钱的部分，我们完全没有对应物。每个 pass 都是绕开 TensorRT parser 某项限制的图重写，
而每个头文件都写明了：限制是什么、可证明等价的重写是什么、**以及它拒绝改动哪些情况**。

| # | 内容 | 来源 | 目标位置 | 判断 |
| --- | --- | --- | --- | --- |
| A1 | **按策略做 Q/DQ lowering。** parser 只接受 ONNX Q/DQ 的一个*子集*——对称、`zero_point == 0`、int8/uint8 围绕 Gemm/MatMul/Conv，这些是它的量化感知融合器调好的模式；非对称 Q/DQ 和 int32/uint32 作为 DQ 输入一律拒绝。该 pass 只把非原生模式重写成等价的 ONNX 算术，**原生模式完全不碰，好让 TRT 还能融合**。它还会主动拒绝那些数值上不安全的重写（量化到 int16/uint16 且 scale 是 fp16/bf16；量化到 int32/uint32 时浮点重写无法精确表示整数范围）。 | `src/qdq_lowering.{h,cc}`（2272 行） | `07-Tool/OnnxFP8QDQConvert/`、`02-API/Layer/QDQStructure/`、`07-Tool/QDQPlacementAutotune/` | **这里最强的一条。** 我们有三个 Q/DQ 范例，没有一个说清楚 parser 实际会融合哪些、会默默拒绝哪些。要搬的是那张*策略表*，不是 2272 行代码。 |
| A2 | **三个小而文档完备的重写 pass。** `Clip(x, -inf, +inf) -> Identity`、`Clip(x, min, +inf) -> Max`（因为 TRT 把 Clip 映射到 activation 的 alpha/beta，必须是有限值）；带 dilation 的 `MaxPool`/`AveragePool` 降解成 slice+Max / slice+Add+除（TRT-RTX 不接受 dilation != 1）；`bool -> Cast(uint8) -> Cast(T)` 规范化成 int32（TRT 不接受 UINT8 作为中间张量）。 | `src/clip_bound_compatibility.*`（480）、`src/pooling_dilation_compatibility.*`（607）、`src/logical_output_compatibility.*`（262） | 新建 `03-Workflow/OnnxCompatibilityShims/`，或作为 `07-Tool/OnnxGraphSurgeon/` 的 case | **强且好移植。** 这三个是纯 ONNX proto 重写，用 `onnx_graphsurgeon` 重写能短很多。每个都带 WebNN/Chromium 的来历说明，解释了*为什么会存在这种图*，这部分占了一半的教学价值。 |
| A3 | **它们共同的方法论，而且是写明的。** 每个 pass 只降解"能证明等价"的子集，并点名未处理的情况：padding/ceil 导致的部分窗口需要额外的掩码与除数逻辑，故保持原样；缺省的 Clip 边界按 dtype 的有限极值而非无穷解释，故不碰。Q/DQ 那个头文件还多一条规则——重写要在**能力发现**和**engine 构建**两处**用相同输入各跑一次**，任何漂移都会导致 TRT 拒绝 ORT 已经分配给该 EP 的节点。 | 三个头文件，`src/qdq_lowering.h:22-31` | A1/A2 落地处的 README 正文 | **采纳这个叙述框架。** "这是重写、这是它拒绝做的、这是原因"正是我们的行文风格，而那条"跑两次"的不变量是一类相当不显然的 bug。 |

## B 档 —— 对我们已有特性的真实世界延伸

| # | 内容 | 来源 | 目标位置 | 判断 |
| --- | --- | --- | --- | --- |
| B1 | **一*组* CUDA graph，按 annotation id 索引。** 不是一个捕获好的图，而是 `unordered_map<annotation, cudaGraphExec_t>`，并保留了 "skip" 和 "default" 两个特殊 id。每个形状或配置捕获一次，每次运行查表。 | `src/cuda_graph.{h,cc}` | `08-Advance/CudaGraph/` | **已落地**（见 `08-Advance/CudaGraph` 的 `case_graph_per_shape`）。它和 TRTMC 那条"捕获的图绑定在捕获时的形状上"互为表里：TRTMC 的答案是重新捕获，这里的答案更好——每个形状缓存一个。 |
| B2 | **自定义 `IGpuAllocator`，配一个真实的失败故事。** `nv_use_sync_gpu_allocator=1` 用 cudaMalloc 支撑的 BFC arena 替换 TensorRT-RTX 默认的 `cudaMallocAsync` 路径，原因是异步分配**在 CiG 下会直接失败**，并且在 Windows 上会在显存仍然充足时耗尽进程虚拟地址空间。它装在 **runtime 和 builder 两处**，连执行上下文的 scratch（`setDeviceMemoryV2`）也走同一个 arena，确保没有任何路径能摸到 `cudaMallocAsync`。device arena 建不起来时记一条警告回退到默认分配器，而不是让 session 失败。 | `src/gpu_sync_allocator.h`、`src/ep_arena.{h,cc}`（1751）、`src/cuda_mempool_arena.*`（614）、`doc/TECHNICAL_NOTES.md:11-144` | `04-Feature/GPUAllocator/` | **强。** 我们的例子赋了 `runtime.gpu_allocator`，并在注释里提了执行上下文；没有覆盖 builder、没有覆盖 device memory scratch，也没有说明为什么要换分配器。那份技术笔记有 130 行就是在说这个原因。 |
| B3 | **用持久化的记录表做 weightless refit。** 用 `IRefitterObserver::onRefittableWeight()` 记录每个可重填权重是如何从 ONNX 图推导出来的，把 parser 所有的 `nvonnxparser::RefitRecord` 深拷贝成自有结构，存为 EPContext 节点的一个属性。加载时直接用 `IRefitter::setNamedWeights` 重填，**完全不需要原始 ONNX**。 | `src/weightless_refit.{h,cc}`（367）、`doc/TECHNICAL_NOTES.md:145-296`、`tests/test_tensorrt_rtx_weightless_refit.cpp`、`examples/cxx/40_ep-context/sample_weightless*.cpp` | `04-Feature/RefitObserver/`、`04-Feature/WeightStripping/` | **强，而且是直接命中。** `RefitObserver/README.md` 已经记录了 `IRefitterObserver` 是 TRT 11.2 新增、C++ only、无 Python 绑定。这是该 API 的一个在产消费者，展示了那份记录到底是**用来干什么**的。它的技术笔记还有"失败行为（不自动回退）"和"已知限制"两节。 |
| B4 | **EPContext：把 engine 嵌进 ONNX 文件。** `com.microsoft::EPContext` 算子，带 `embed_mode`、`ep_cache_context`、`hardware_architecture`、`ep_sdk_version`、`partition_name` 等属性。文件模式和缓冲区模式都有示例。 | `src/onnx_ctx_model_helper.{h,cc}`（977）、`examples/cxx/40_ep-context/` | 新建一个 `07-Tool/` 例子，或作为 `07-Tool/Onnxruntime/` 的一个 case | **值得做一个例子。** 我们整棵树里 EPContext 一次正经出现都没有。它是业界把预编译 engine 塞进 ONNX 文件的标准做法，而且和我们的 `export_engine_as_onnx` 恰好相反——后者是把 engine 的*结构*导成 ONNX 供查看，前者是把 engine 的*字节*放进 ONNX 供运行。这个对照值得写。 |

## C 档 —— 测试与工具技巧

| # | 内容 | 来源 | 目标位置 | 判断 |
| --- | --- | --- | --- | --- |
| C1 | **用 CUPTI 证明零拷贝。** 按版本列表探测并动态加载 `libcupti`，注册 activity 回调，统计整轮运行中 HtoD memcpy 记录条数，断言为零。**用仪器而非计时去验证一个关于内存流量的论断。** | `tests/test_tensorrt_rtx_zero_copy_cupti.cpp`（695） | `08-Advance/PinnedMemory/`、`04-Feature/OutputAllocator/`，或作为 `tests/` 的一种技法 | **技法很强，工作量中等。** 我们所有断言都在证明数值和形状对，没有一个在证明"某种开销没有发生"。那个"按版本名列表探测动态库"的小技巧也可复用。 |
| C2 | **provider 选项文档写在头文件里，参考手册自动生成。** `src/tensorrt_rtx_provider_options.h` 在每个选项常量上带 `/// @brief`、`/// @par Type`、`/// @par Default`、`/// @par Accepted values`；一个 190 行的脚本把这些标记解析成 RST 选项参考。单一事实源，不会漂移。 | `docs/generate_provider_options.py`、`src/tensorrt_rtx_provider_options.h`（300） | `build-Copyright-and-README.py`、`07-Tool/ListAPIs/` | **如果以后有了旋钮表就采纳。** 我们已经从源码生成 README 内容，这是同一套做法用在配置面上。 |
| C3 | **路径穿越校验，并把自身限制写了下来。** 只做词法检查，头文件直言："Symlinks are NOT resolved: a symlink whose textual name passes validation but resolves outside the intended directory is not caught here. This is a known limitation."，并配有专门的测试文件。 | `src/utils/path_validation.h`、`tests/test_tensorrt_rtx_path_traversal.cpp`（183） | `92-LocalFile/PlanTool/`，以及任何接受用户路径的地方 | **小，值得抄的是这个态度。** 我们的 `plan_tool.py` 从命令行取路径；TRTMC 的 `test_impact.py` 也有一个同类的 `_safe_relative`。两个独立仓库给出同一条教训，这本身就是信号。 |
| C4 | **用 C++ 构造 ONNX 测试模型。** 1053 行 proto 构造辅助代码，含手写的 fp32→fp16 转换，使 C++ 测试无需 Python 即可产出模型。 | `tests/test_tensorrt_rtx_model_builder.h` | `02-API/Layer/*/main.cpp` 一类的测试 | **知道有这回事即可，不搬。** 我们用 Python 构造测试 ONNX，对 cookbook 是正确选择。将来若有纯 C++ 例子需要模型，再回头看。 |
| C5 | **Windows 下加载前校验 DLL 签名。** 基于 `wintrust` 的 Authenticode 校验加延迟加载钩子，未签名的依赖不予加载。 | `src/secure_load.cc`（283）、`src/utils/security.h` | — | **只读不搬。** Windows 专有且超出 cookbook 范围，但它是这几个仓库里唯一一处把"刚才加载的是哪个 .so/.dll"当作安全问题来处理的代码。 |

## D 档 —— 看过，不建议搬

+ **EP ABI 实现本体**（`src/tensorrt_rtx_execution_provider.cc` 6151 行、`tensorrt_rtx_provider_factory.cc`
  2403 行）—— `OrtEpFactory` / `OrtEp` / `OrtNodeComputeInfo` 的管线代码。教的是 ONNX Runtime 扩展而
  不是 TensorRT。除非我们以后要加"把 TensorRT 集成进别的框架"一章，否则不在范围内。
+ **五个 C++ sample**（`examples/cxx/10..40`）—— device selection、device tensor、异步数据传输、
  sync stream、EP context。结构不错，但五个里有四个教的是 ORT 的 V2 device API。只有
  `40_ep-context` 对 cookbook 有价值，已由 B4 收录。
+ **D3D12 / Vulkan / CiG interop**（`tests/test_tensorrt_rtx_{dx,vk}_*`）—— 图形 interop 确实是 RTX
  的真议题，但需要我们无法测试的 Windows 图形栈。
+ **`CODING-GUIDELINES.md`**（19 KB）—— 详尽的 C++ 风格指南。我们已有 clang-format 强制执行，这里没
  有能改变任何决定的东西。
+ **`build.sh` / `build.bat`**（16 KB / 25 KB）—— 以 SDK 根目录驱动的 CMake 包装脚本，带 wheel 打包。
  我们的 `setup_environment.sh` 解决的是另一个问题。**注意**：`12-TensorRT-RTX-EP-ABI/run.sh` 并没有
  重新实现它，而是直接调用它——理由见那个例子的 README。

## 跨仓库的一条观察

和 TensorRT-Model-Connect 的调研合起来看，三个独立的 NVIDIA 仓库在三件我们沉默的事情上达成一致：**校验用户提供的
路径**（这里的 C3，那边的 `test_impact.py`）、**CUDA graph 绑定在捕获时的形状上**（这里的 B1，那边的
`trt_module_impl.cpp`）、**说清楚一个重写或一次测量拒绝覆盖什么**（这里的 A3，那边的 profiling
混淆变量清单）。这三条改起来便宜，而且不依赖更重的条目最后筛不筛得过。
