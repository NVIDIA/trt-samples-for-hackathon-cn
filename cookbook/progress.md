# Cookbook 工作进度

> 断线续做用。新会话先读本文件，找到「下一步」一节直接接着干。
> 路径若无特殊说明均相对 `/work/trt-samples-for-hackathon-cn-wili/cookbook/`。

## 当前任务：扩展 `07-Tool/Polygraphy/More/`（**已全部完成**，2026-08-27）

**目标**：现有 `07-Tool/Polygraphy/` 有 9 个 CLI 子目录，Python API 侧只有一个 `API/`
（`main.py` 是 `CreateConfig` 的 kwarg 清单，`gs_workflow.py` 是写着
`# Do something with gs APIs` 的空壳）。在 `More/` 下补齐 Python API 的范例。

**源码参考**：`/work/trt/TensorRT-GitHub/tools/Polygraphy`（仓库 0.49.27）。
**装的版本**：**0.50.3**，比仓库新，对照时注意。

### 环境（2026-08-27 确认）

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA H100 PCIe 80GB（**单卡**） |
| TensorRT | 11.1.0.106 |
| polygraphy | 0.50.3 |
| torch / torch_tensorrt | 2.13.0a0+9186a08b2c.nv26.07 / 2.14.0a0 |
| onnx / onnx_graphsurgeon / onnxruntime-gpu / onnxslim | 1.21.0 / 0.6.1 / 1.29.0 / 0.1.96 |
| triton | 3.7.1 |

注意：`from cuda.bindings import runtime as cudart`（老写法已失效）。

**⚠️ 容器重启会丢包**（2026-08-27 B5 开工时遇到）：`onnxruntime` / `onnx_graphsurgeon` /
`onnxslim` 不在镜像里，是前面会话 pip 装的。容器一重启就没了，
而 `import tensorrt_cookbook` 依赖它们 → 所有 main.py 都 import 失败。恢复：

```bash
pip install onnxruntime-gpu onnx_graphsurgeon onnxslim
```

### 任务清单与进度

按 A → B → C 顺序做。**每做完一个立即回来更新本表并跑 runner。**

| # | 主题 | 核心 API | 状态 |
| --- | --- | --- | --- |
| A1 | 懒加载器 vs 立即求值两种风格 | `EngineFromNetwork` vs `engine_from_network` | ✅ `More/01-LazyVsImmediate/` |
| A2 | 跨框架输出比对 | `Comparator.run` + `CompareFunc.simple`/`.indices` | ✅ `More/02-ComparingBackends/` |
| A3 | 新的比对函数 | `CompareFunc.distance_metrics`/`quality_metrics`/`perceptual_metrics` | ✅ **已并入 A2**（上游 `01_comparing_frameworks` 本就覆盖，拆开是重复） |
| A4 | 与 TensorRT 原生 API 互操作 | `@polygraphy.func.extend` | ✅ `More/04-ExtendInterop/` |
| A5 | 用 TRT network API 手搭网络 | `CreateNetwork` + `extend` | ✅ `More/05-BuildNetworkByHand/` |
| A6 | INT8 校准 | `trt.Calibrator` | ✅ `More/06-Int8IsNowExplicit/`（**API 已移除，改为迁移指南**） |
| A7 | 动态 shape 与多 profile | `Profile` + `TrtRunner(optimization_profile=)` | ✅ `More/07-ProfilesAndDynamicShapes/` |
| A8 | 在真实数据集上验证 | 直接用 runner，不用 `Comparator` | ✅ `More/08-ValidatingOnADataset/` |
| A9 | 保存/复用推理输入输出 | `RunResults` / `IterationResult` / JSON | ✅ `More/09-SavedInputsAndResults/` |
| A10 | PyTorch 张量直接进 runner | `TrtRunner` 收发 `torch.Tensor`（含 BF16） | ✅ `More/10-PyTorchTensors/` |
| B1 | TRT network 导出成 ONNX-like | `OnnxLikeFromNetwork` | ✅ `More/11-NetworkAsOnnxLike/` |
| B2 | 战术记录与回放 | `TacticRecorder`/`TacticReplayer`/`TacticReplayData` | ✅ `More/12-TacticsAndReproducibility/`（**API 已移除，转向 timing cache**） |
| B3 | 逐层精度控制 | `SetLayerPrecisions`/`SetTensorDatatypes`/`SetTensorFormats`/`PostprocessNetwork` | ✅ `More/13-PerLayerPrecision/` |
| B4 | 调试张量 | `MarkDebug` | ✅ `More/14-DebugTensors/` |
| B5 | 零拷贝设备内存 | `polygraphy.cuda.DeviceArray`/`DeviceView`/`Stream` | ✅ `More/15-DeviceMemory/` |
| B6 | 插件参考实现比对 | `PluginRefRunner` | ✅ `More/16-PluginReference/` |
| B7 | 版本兼容 / lean runtime | `LoadRuntime` + `CreateConfig(version_compatible=)` | ✅ `More/17-VersionCompatibility/` |
| C1 | 用 Polygraphy 基类写自己的 CLI | `polygraphy.tools` | ✅ `More/18-WritingACliTool/` |
| C2 | 给 `polygraphy run` 加自定义 backend | extension module | ✅ `More/19-CustomBackend/` |

### 约定（沿用 `06-DLFrameworkTRT/Torch-TensorRT/` 的做法）

- 每个候选一个子目录 `More/<Name>/`，含 `main.py` + `README.md` + `unit_test.yaml`
- `main.py` 用 `@case_mark`（来自 `tensorrt_cookbook`）分 case
- **不照抄上游范例**：先跑一遍上游写法、测它宣称的效果，把**对不上的地方**作为主要产出
- 每个结论都要有实测数字或断言撑着，不写没有证据的话
- 不往仓库里放 `.onnx` / `.trt`（`cookbook/.gitignore` 已挡住 `*.onnx` `*.trt` `*.json`）
- 模型优先用 `00-Data/model/` 下已有的，或代码现造

### 已完成的产出

- **A1 `More/01-LazyVsImmediate/`**（4 个 case，runner 60s 通过）
  - 懒 0.05 ms vs 立即 11240 ms；前者返回 `EngineFromNetwork`，后者返回 `ICudaEngine`
  - **坑**：懒加载器**没有记忆化**，调用两次 build 两次（8.35s + 8.29s，对象不同），无警告
  - 懒加载器可 `deepcopy`/`pickle`，`ICudaEngine` 不行 → 这就是懒 API 存在的理由
  - 改网络：立即式直接改；懒式要 `@func.extend`
  - 注意：Polygraphy 每次 build 都打印整张 config 表，脚本里用
    `G_LOGGER.module_severity = G_LOGGER.ERROR` 压掉

- **A2 `More/02-ComparingBackends/`**（5 个 case，runner 25s 通过）**含原 A3**
  - 真实差异 `max |trt-ort| = 7.451e-09`（fp32 舍入）；上游用 identity 模型，两边永远相等，
    读者看不到比对失败长什么样。这里所有比对都对着这个真实数字跑
  - `simple(atol=1e-8)` PASS / `1e-9` FAIL —— 一个数量级就翻盘
  - `distance_metrics` L2=9.80e-09 cos=1；`quality_metrics` PSNR=144.69dB SNR=140.52dB；
    收紧后同样 FAIL —— 它们**不是更宽松**，是判断整个张量而非最差单点
  - **坑1**：`CompareFunc.indices` 是给 Top-K 输出用的，直接套到 logits 上会把浮点当索引比，
    必然 FAIL。正确用法 `PostprocessFunc.top_k(k={"y": (5,1)})` 再比。
    `k` 要按输出名限定，`z` 是 1-D，`top_k` 在 axis=1 上会抛 `AxisError`
  - **坑2**：`perceptual_metrics` 缺 `lpips` 包时**仍然返回 PASS**，那个 PASS 无意义。
    脚本里改成检测到缺包就 SKIP
  - **发现**：`CreateConfig(fp16=True)` 在 TRT 11 上**抛 `PolygraphyException`**
    （ONNX 解析出的网络是 STRONGLY_TYPED）。与 Torch-TRT 的 `enabled_precisions`
    静默忽略**正好相反** —— 已在两边 README 交叉引用
  - 注意：`CreateConfig` 的校验发生在**应用到 network 时**，光 `CreateConfig(fp16=True)()`
    只会得到 `TypeError: missing 2 required positional arguments`，要真 build 才看得到拒绝

- **A4 `More/04-ExtendInterop/`**（4 个 case，runner 42s 通过）
  - `@func.extend(Loader())` 机制：函数收 loader 的产物、不用 return，链仍是懒的
  - `config.set_flag(BuilderFlag.REFIT)` 与 `CreateConfig(refittable=True)` 等价
  - **重要发现**：**`trt.BuilderFlag` 里 `FP16`/`INT8`/`BF16` 在 TRT 11 全被移除**，
    只剩 `TF32`。所以上游这个范例里 `config.set_flag(trt.BuilderFlag.FP16)` 那一行
    在 TRT 11 上直接 `AttributeError`——**上游范例本身跑不了**。
    也说明 A2 里 `CreateConfig(fp16=True)` 的拒绝不是保守，是底层 API 没了
  - **坑**：`TrtRunner` 复用输出缓冲区。留着 `outputs["y"]` 的引用再 `infer()` 一次，
    那个引用会**变成新结果**，旧值无声丢失。要跨调用留存必须 `copy.deepcopy`

- **A5 `More/05-BuildNetworkByHand/`**（4 个 case，runner 通过）
  - `CreateNetwork()` 默认就是 **STRONGLY_TYPED**，且没有关掉它的 flag
  - **闭环**：FP16 靠 `add_input(dtype=trt.float16)` + fp16 权重得到，**不碰 config**。
    这是 02/04 那条「FP16 不再是 builder flag」的正面答案
  - 类型不匹配（bfloat16 输入 + float16 权重）**build 直接失败**，
    报 `ElementWiseOperation SUM must have same input types`——弱类型时代会静默插 cast

- **A6 `More/06-Int8IsNowExplicit/`**（3 个 case，runner 通过）
  - **整个 INT8 校准 API 在 TRT 11 上已删除**：`BuilderFlag.INT8`、`IInt8Calibrator`、
    `IInt8EntropyCalibrator2`、`IInt8MinMaxCalibrator` 全部 REMOVED
  - `CreateConfig(int8=True)` → `PolygraphyException`；
    `CreateConfig(calibrator=Calibrator(...))` → `AttributeError`（**从 TRT 内部抛的**）
  - **关键**：`from polygraphy.backend.trt import Calibrator` **能 import、能构造**，
    只有 build 时才失败 → 照上游 `04_int8_calibration_in_tensorrt` 写的代码看着没问题、一跑就炸
  - 替代路径：QDQ 显式量化。`00-Data/model/model-trained-int8-qat.onnx` 有 8+8 个 QDQ 节点，
    scale 是 initializer，**不向 builder 传任何东西**
  - 实测 float vs qat：logits 差 **9.96**（本该差），但 **argmax 相同** ——
    量化模型该问的是排序是否保住，不是逐元素容差

- **A7 `More/07-ProfilesAndDynamicShapes/`**（4 个 case，runner 通过）
  - 一个 engine 装 3 个 profile，权重只存一份
  - `TrtRunner(optimization_profile=N)` 绑定后，**适用范围是该 profile 的，不是 engine 的并集**
    （profile 0 拒绝 batch 4，尽管 profile 1 支持）
  - **诚实的负面结果**：pinned 0.601ms vs dynamic 0.607ms = **1.01x**，
    小模型上 profile 调优几乎不影响。上游只建三个 profile 就收尾，从不测量
  - profile 只改 tactic 不改数学：`max |diff| = 7.451e-09`（与 A2 同一个舍入）
  - 模型用 `00-Data/model/model-trained.onnx`，输入是 `['nBS',1,28,28]` 动态 batch

- **A8 `More/08-ValidatingOnADataset/`**（4 个 case，runner 通过）
  - 数据集 `00-Data/data/TestData.npz`：500 张 MNIST，**标签是 one-hot**，要 `argmax`
  - **更重要的理由不是大小**：`Comparator` 只比较 runner 之间，**没有放 label 的地方**。
    直接循环得到 accuracy **194/200 = 97.0%**
  - 大小论断实测：`Comparator.run` 保留 200 个 IterationResult 共 **9600 B（48 B/迭代）**。
    外推：10000 迭代下 MNIST 仅 0.4MB，但分割模型 (1,3,1024,1024) 要 **120 GB**。
    注意：**用 `ru_maxrss` 测不出来**（高水位 + MNIST 输出太小），要直接累加 `nbytes`
  - 批处理：batch 1 → 135.9ms，batch 50 → 11.6ms，**11.7x**，准确率不变
  - 坑：默认 profile 只接受 batch 1，批处理要先建够宽的 `Profile`

- **A9 `More/09-SavedInputsAndResults/`**（4 个 case，runner 通过）
  - 上游只演示**加载** CLI 产生的文件；这里从 Python **生成**再读回，往返 bit-exact
  - `load_json` 用于普通对象，Polygraphy 自己的类型用 `save`/`load`
  - **CLI↔Python 打通**：`RunResults.load` 能直接读 `polygraphy run --save-outputs` 的产物。
    注意 CLI 的 runner 名带时间戳（`trt-runner-N0-08/27/26-00:52:01`），**不要写死**
  - `RunResults.add(iterations, runner_name=)` 可把两个进程的结果并成一个对象再比对
  - **查证过的事实**：Polygraphy **默认 data loader 是确定性的** —— 两个独立进程
    生成的输入逐比特相同（`DataLoader(seed=None)` 但实际有固定种子）。
    这才使得跨进程比对有意义。初稿我写成「不该一致」，与输出的 True 自相矛盾，已改
  - CLI 调用要用 `polygraphy` 可执行文件，`python3 -m polygraphy` 会报
    `No module named polygraphy.__main__`

- **A10 `More/10-PyTorchTensors/`**（5 个 case，runner 通过）
  - numpy 进 → ndarray 出；torch 进 → Tensor 出（默认在 CPU，
    `copy_outputs_to_host=False` 则留在 cuda:0）
  - **未文档化的坑**：`copy_outputs_to_host=False` 返回什么，
    由 runner **第一次** infer 的 feed dict 决定，不是当前这次。
    先喂 numpy 再喂 torch → 返回 `DeviceView`（没有 `.device`/`.cpu()`），
    下游 torch 代码在离原因很远处炸 → **一个 runner 只喂一种数组类型**
  - BF16 端到端可用（手搭 bf16 网络 + `torch.bfloat16` 张量，exact match）
  - **关于「NumPy 没有 BF16」要小心**：干净解释器里 `np.dtype('bfloat16')` 抛 TypeError，
    但 `ml_dtypes`（本镜像的传递依赖）被 import 后就能解析。依赖它是碰运气。
    `torch.bfloat16` 无需注册——这才是用 torch 的稳妥理由
  - 上游这个范例同样用了 `Calibrator`+`int8=True`，**在 TRT 11 上跑不了**（第三个失效的上游范例）

- **B1 `More/11-NetworkAsOnnxLike/`**（3 个 case，runner 通过）
  - **parser 插了一堆东西**：源 ONNX 12 节点 → TRT network 27 层。
    `Gemm`→`MATRIX_MULTIPLY`+`ELEMENTWISE`，`ArgMax`→`TOPK`+`SQUEEZE`，
    `Reshape` 拖出 `SHAPE`/`CAST` 链。build log 里的层名对不上 ONNX 时就该看这个
  - 确实不是合法 ONNX：`onnx.checker` 报
    `No Op registered for CONVOLUTION with domain_version of 11`
  - **initializer=0 不等于没权重**（我初稿写错了，实测纠正）：TRT 层的所有参数
    都变成 ONNX **attribute**，权重也一样（首个 `CONVOLUTION` 带 kernel=800 floats、
    bias=32 floats）。protobuf 存 `FLOATS` 列表远不如 initializer 的 `raw_data` 紧凑，
    所以**导出文件比源模型还大**：15.6 MB vs 12.5 MB。大模型上要当心
  - **与 cookbook 自己的 `export_network_as_onnx` 正面对比**（同一个 `Loop`+`If` 网络）：
    Polygraphy 在 `gs.export_onnx` 里抛
    `ValueError: Could not infer the attribute type from the elements of the passed Iterable`，
    cookbook 的写出全部 38 层。**有控制流就只能用 cookbook 那个**
  - `unit_test.yaml` 的 `clean` 要删 `*.onnx`（16 MB 产物）

- **B2 `More/12-TacticsAndReproducibility/`**（5 个 case，runner 65s 通过）
  - **tactic replay 整条路没了**：`trt.IAlgorithmSelector` 及整个 `IAlgorithm*` 家族
    在 TRT 11 全部移除，`IBuilderConfig.algorithm_selector` 也没了。
    Polygraphy 那边同时标了 deprecated（0.55.0 移除）。
    `TacticRecorder(...)`/`TacticReplayer(...)` **构造就抛**
    `AttributeError: module 'tensorrt' has no attribute 'IAlgorithmSelector'`
  - 但 `TacticReplayData()` 能构造（只是个 OrderedDict，不碰 TRT）→
    与 A6 的 `Calibrator` 同一种坑。**第 4 个跑不了的上游范例**
  - **重要实测坑**：`EngineFromNetwork(..., save_timing_cache=path)` 是**只写的**。
    build 后写盘（还会和已有文件 combine），但**从不读回来喂给下次 build**。
    无 cache 8.43s / 传 kwarg 第一次 8.36s / 第二次 8.38s = **1.00x，完全没加速**，
    而文件是满的、内容也对 → 典型"看着对、实际没生效"
  - 正确做法：`CreateConfig` 没有 load 的 kwarg，要用 `@func.extend` 手动
    `config.set_timing_cache(config.create_timing_cache(bytes), ignore_mismatch=False)`
    → 8.38s → **7.05s，1.19x**
  - **测量方法学**：进程内第一次 build 含 CUDA 初始化（11.22s vs 8.4s）。
    不先跑一次丢弃的 warm-up，就会看成"第二次快了 = cache 生效了"——会得出完全相反的结论
  - **怎么才能发现**：`BuilderFlag.ERROR_ON_TIMING_CACHE_MISS`，
    没命中就直接 build 失败（`Assertion !errorOnMiss failed`）。CI 里该开
  - `ITimingCache.queryKeys()`/`query()` 就是现在的 tactic 记录（本例 30 条，
    含 tacticHash + timingMSec）。**强制换用另一个 tactic** 要 `EDITABLE_TIMING_CACHE`，
    `04-Feature/TimingCache/` 已覆盖，本例不重复、只交叉引用

- **B3 `More/13-PerLayerPrecision/`**（6 个 case，runner 43s 通过）
  - **前提**：TRT 11 里 `create_network()` 不传任何 flag 也已是 `STRONGLY_TYPED`（flags=1），
    **弱类型网络彻底没了**，也没有关掉它的 flag
  - **坑**：`NetworkFromOnnxPath(strongly_typed=False)` **被静默忽略**，
    flag 查回来仍是 True。无 warning 无 error
  - 四个 loader 三种结局：
    | loader | 结果 |
    | `SetLayerPrecisions` | 干净拒绝：`PolygraphyException: layer precision ... not available on TensorRT version 11.1.0.106` |
    | `SetTensorDatatypes` | 裸 pybind 报错：`AttributeError: property of 'ITensor' object has no setter` |
    | `SetTensorFormats` | **可用**，format 是布局不是类型 |
    | `PostprocessNetwork` | **可用**，通用逃生口 |
  - `trt.ILayer.precision`/`precision_is_set` 已移除；
    `BuilderFlag` 里 `OBEY_PRECISION_CONSTRAINTS`/`PREFER_PRECISION_CONSTRAINTS` 也没了；
    `ITensor.set_dynamic_range` 同样没了
  - **两种移除方式的对照**：`SetLayerPrecisions` 报错带版本号（好），
    `SetTensorDatatypes` 只说"某属性没 setter"（差，离病因两层）。
    这个对照本身值得记——A6 的 `Calibrator`、B2 的 `TacticRecorder` 都属后者
  - **format 也被 dtype 反锁**：`LINEAR`/`CHW32`/`HWC` 能建且 engine 上可见；
    `CHW4`(int8)/`HWC8`(fp16) 报
    `has dataType Float unsupported by tensor's allowed TensorFormats`，
    而"改 dtype"正是 `SetTensorDatatypes` 干不了的事 →
    **解析 ONNX 得到的网络，可达 format 集合由 ONNX 文件决定**
  - `PostprocessNetwork(network, func)` 收任意函数、保持懒求值；
    用它 mark 中间 ReLU 输出成额外 output（`['y','z','relu']`），
    是那三个 loader 都做不到的

- **B4 `More/14-DebugTensors/`**（5 个 case，runner 85s 通过）
  - 端到端可用：`MarkDebug` 标网络 + `trt.IDebugListener` 收回调，
    取到 `relu` 的 (1,32,28,28)、range [0, 272.04]，
    而 **engine I/O 仍是 `['x','y','z']`** —— 这就是它相对 mark_output 的唯一卖点
  - 回调里的指针**只在回调期间有效**，要留就得在回调里 copy 出来
  - **坑（我初稿写错、实测纠正）**：`set_tensor_debug_state(name, True)` 是**空操作**，
    `MarkDebug` 已经把状态置 True（`get_debug_state` 查得到）。
    真正必需的只有 listener，**忘了 listener 是静音失败**：推理正常、输出正确、
    captured 为空 dict，读起来像"这张量没被写过"。
    `False` 才是有用的方向——不重建 engine，按需静音单个张量
  - **代价实测（全程没开 listener/debug state）**：
    baseline 0.605ms/125440B → `MarkDebug` 0.943ms/201216B = **1.56x**，
    `mark_output` 0.740ms/75264B = 1.22x。
    **只要标了就付钱，且付在 build 期**（可观测的张量不能被 fuse 掉）→ 调试专用 build
  - `mark_unfused_tensors_as_debug_tensors=True` **确实生效**（捕到 9 个 vs False 的 0 个）。
    **但 `network.is_debug_tensor` 查不到**——哪些张量"未被 fuse"要 build 完才知道。
    我最初就是查错了地方、差点误判成"静默忽略"（与 B3 的 `strongly_typed` 不同，**这个是真生效的**）
  - 捕回来的名字是 TRT fuse 后的内部名（`__myln_k_arg__bb1_3_myl4`），
    只能看出"某个值不对"，定位不到原模型的层

- **B5 `More/15-DeviceMemory/`**（6 个 case，runner 63s 通过）
  - 接口全表：`DeviceView` 只有 `copy_to`/`numpy`/`ptr`/`shape`/`dtype`/`nbytes`；
    `DeviceArray` 多 `copy_from`/`view`/`resize`/`free`/`raw`/`allocated_nbytes`；
    `Stream` 只有 `synchronize`/`free`/`ptr`。`view()` 返回**同一个 ptr**，不分配不拷贝
  - **A10 的伏笔收口**：那个没有 `.device`/`.cpu()` 的对象就是 `DeviceView`，上表就是它的全部
  - **零拷贝是真的**：`context.get_tensor_address("x")` 作证 ——
    numpy 喂 → 绑 runner 自己的 buffer（0x4578020000）；
    `DeviceView` 喂 → 绑我们的地址（0x4578000000），输出逐位相同
  - **数字（本例唯一值得给的）**：4 种组合 × 2 个模型
    | 模型 | host→host | dev→dev | 比 |
    | MNIST 3 KB | 0.582 ms | 0.397 ms | 1.46x（省 0.185 ms） |
    | 手搭 25 MB `y=x+x` | 9.328 ms | 0.290 ms | **32.1x**（省 9.0 ms，97% 是拷贝） |
    → MNIST 上省的几乎不是带宽、是两次拷贝+同步的固定开销。**技术随字节数缩放**，
    1 MB 以下不值得为它承担下面两个生命周期坑
  - **坑1 输出 `DeviceView` 是 runner 的 buffer**：`copy_outputs_to_host=False` 返回的
    view 指向 output allocator，**下次 infer 地址不变**。留着它跨一次 infer，
    读到的是**第二次**的结果。比 A4 的 host 版更难发现（view 根本没有"内容"可看）
  - **坑2 view 不持有内存**（两种死法都是 owner 自己干的、都无声）：
    `free()` 后下一次分配落回同一地址 → stale view 读到别人的张量（实测 1.0 → 9.0，无任何报错）；
    `resize()` 变大内部是 free+malloc → 之前的 view 也悬空，新分配落在别处时读它
    **直接 SIGSEGV**（子进程验证 returncode -11，3/3 稳定，Python 层没有 traceback）。
    注意这与 `with DeviceArray(...) as a:` 的习惯冲突：view 逃出 with 块就已经悬空
  - **坑3 `stream=` 在 pageable 内存上根本不异步**：256 MB H2D，
    numpy 的 `copy_from(buf, stream)` **阻塞 22.6 ms** 才返回，pinned(torch) 只要 0.102 ms。
    后果：忘了 `synchronize()` 在 numpy 上**观察不到**（D2H 同步前数据 100% 已到），
    换成 pinned 立刻炸（同步前只到 2.5%）→ **优化的那天旧 bug 才开始发作**
  - `resize()` 从不缩小分配：`resize((1,3))` 后 `nbytes 12` 而 `allocated_nbytes 24`
  - **读 `.dtype` 本身已 deprecated**（0.55.0 移除）：现在返回 numpy dtype，将来返回
    Polygraphy `DataType` → `np.empty(..., dtype=view.dtype)` 是有保质期的代码

- **B6 `More/16-PluginReference/`**（5 个 case，runner 29s 通过）
  - **名字两个词都不对**：`PluginRefRunner` 不加载任何插件，且 `OP_REGISTRY` 只有
    `Identity`/`InstanceNormalization`/`MeanVarianceNormalization` 三个。
    它跑**整张图**，一个没注册的节点就停 —— 连 `Conv`、`Mul` 都不支持
  - **最贵的坑：`LoadPlugins` 用 `ctypes.CDLL`，注册不了 `IPluginCreatorV3One`**。
    实测 creator 数：启动 16 → `LoadPlugins` 后仍 16 → `init_libnvinfer_plugins` 后 44
    （AddScalar 仍不在）→ `trt.get_plugin_registry().load_library()` 后 45，**才有**。
    `LoadPlugins` 还会 INFO 打印 "Loading plugin library"，报错在两层外的 ONNX parser：
    `Plugin not found, are the plugin name, version, and namespace correct?`
    —— 把人引去查名字/版本/namespace。**CLI 的 `--plugins` 同一条路，同样加载不了**
    （实测 `polygraphy run --trt --plugins ...` exit 1）。
    正确写法就是 `tensorrt_cookbook/utils_plugin.py` 里已经在用的 `registry.load_library()`
  - **`register` 没有被导出**：必须 `from polygraphy.backend.pluginref.references import register`。
    签名 `(attrs, *inputs)`，**返回 list**（每个 node output 一项）；常量输入已是 numpy
  - **比对通过说明不了什么**：参考实现硬编码 `1.0`，模型 attr 恰好也是 1.0 → PASS；
    换成 attr=5.0 的模型 → FAIL，`max |diff| = 4.0`。参考一直是错的，只有第二个模型能看见。
    `OP_REGISTRY` 是普通 dict，重复 register **静默覆盖**
  - **真实模型要先切子图**：`gs` 改 `inputs`/`outputs` + `cleanup()`。
    坑：`graph.copy()` 产生**新的 tensor 对象**，拿旧 `Variable` 去赋值会让 `cleanup()` 抛
    `Encountered a node not in the graph` → 必须 `copy.tensors()` 按名字取
  - 用的插件是 `05-Plugin/ONNXParserWithPlugin/AddScalarPlugin.so`（`main.py` 缺了会自己 make）

- **B7 `More/17-VersionCompatibility/`**（5 个 case，runner 150s 通过）
  - **`version_compatible=True` 就是把 lean runtime 塞进 plan**：
    13.190 MB → 118.036 MB（8.95x），差值 104.847 MB，
    而 `libnvinfer_lean.so.11.1.0` 正好 104.845 MB（差 1528 B）。
    **是固定 ~105 MB 不是百分比**：手搭一层网络 0.011 MB → 104.856 MB = **9221x**
  - `exclude_lean_runtime=True` 把体积**完全**还原（与 baseline 差 0 B），
    然后由 `LoadRuntime(路径)` 在加载时补上 runtime。N 个 engine 共用一个 105 MB 库
  - `exclude_lean_runtime` 不配 `version_compatible` → Polygraphy 自己拦下并说清怎么办
    （与 B3 那种 pybind 深处报错形成对照）
  - **代价在加载不在推理**：反序列化 3.71 → 127.70 ms = **34.4x**；
    推理 0.578 vs 0.572 ms，差 6 us 是噪声。所以伤的是进程启动/镜像体积，不是吞吐
  - **裸 `trt.Runtime` 反序列化 VC engine 返回 `None`（不是抛异常）**：
    `Cannot deserialize engine with lean runtime since getEngineHostCodeAllowed() is false`。
    Polygraphy 的 `EngineFromBytes` 会替你设 `engine_host_code_allowed = True`
    （包在裸 `try/except AttributeError` 里）。
    **名字反直觉**：**含** lean runtime 的那个被裸 runtime 拒绝，**排除**的那个反而能加载
  - `hardware_compatibility_level` 是另一条轴：`AMPERE_PLUS` 只多 0.46 MB、延迟看不出差别
    （诚实结论：MNIST 用不到被限制掉的 kernel，不代表 transformer 也这样）
  - **一条过期建议**：开 VC 时 Polygraphy 提示要加 `NATIVE_INSTANCENORM`，
    TRT 11 上实测加不加**产物字节数完全一样**（104860708 B）

- **C1 `More/18-WritingACliTool/`**（5 个 case，runner 17s 通过）
  - 产出一个真的工具 `plan-size`（可执行文件，50 行）：建 engine 并报告耗时与 plan 大小
  - **订阅 arg group 就白拿 CLI 选项**：空工具 7 个（`LoggerArgs` 是基类自动订阅的）→
    +`DataLoaderArgs` 15 个 → 9 个订阅 **95 个**。
    `plan-size` 自己只声明 `--json`，却能吃 `--version-compatible`（B7 的 118 MB 从它跑出来）
  - **坑：依赖关系只写在 docstring 的 `Depends on:` 里，没有任何校验**。
    漏订阅 `OnnxInferShapesArgs` → 参数解析成功、模型读完、然后
    `KeyError: <class ...OnnxInferShapesArgs>`，从 `ArgGroups.__getitem__` 抛出，7 层深
  - `main()` 里是 `sys.exit`，进程内跑要自己拼
    `setup_parser` / `parse_args(列表)` / `parse` / `run`
  - 基类是 `run_impl`，**上游范例覆盖的是 `run`**（能跑，但跳过了打印模块版本、None→0）；
    两个都不写 → `NotImplementedError`。没有类 docstring → `AssertionError`（`-O` 下变成空 help）
  - **不能给 `polygraphy` 可执行文件加子命令**：`tools/registry.py` 是 import 时执行的
    一串硬编码 `try_register_tool`，没有 entry point。整个 CLI 唯一的扩展点是
    `polygraphy.run.plugins`（见 C2）

- **C2 `More/19-CustomBackend/`**（7 个 case，runner 27s 通过）
  - 产出可安装包 `extension/polygraphy_cookbook_ref`：加 `--cookbook-ref`，
    一个 NumPy 解释器，支持比 `--pluginref` 多的 op，且能选 float32/float64 工作精度
  - **必须真安装，PYTHONPATH 不够**：entry point 来自已安装的 dist 元数据。
    包能 import、选项就是不出现，**没有任何提示**。
    半吊子状态：build 一次会留下 `*.egg-info`，那也是元数据 → `pip uninstall` 后
    PYTHONPATH 又"能用了"，其实是残留产物（main.py 卸载时会一并删掉）
  - entry point group 名固定 `polygraphy.run.plugins`，只对 `polygraphy run` 生效，
    `convert`/`inspect` 都看不到
  - **runner 是被生成的脚本调起来的**：`add_to_script_impl` 是往脚本里加行，
    `--gen-script -` 可以把那段代码打出来
  - **float64 参考的价值**：模型 `(x/3 + 1e7) - 1e7`，float32 下 onnxrt 与 float32 参考
    `max_absdiff=0` 完全一致 —— 两个后端一致**不等于**结果对。
    换 float64 参考后 `max_absdiff=0.29271, max_reldiff=1`（全军覆没），
    TRT 与 onnxrt 给出完全相同的错误答案 → 错在模型不在后端
  - **import 即注册**：`__init__.py` 里 `register("AddScalar")`，装上之后
    `--pluginref` 就能跑 AddScalar（B6 说的"没有 CLI flag"的唯一出路）；
    卸载后立刻恢复报错。反过来说，装扩展模块 = 执行它的代码
  - **选项名必须加前缀**：与内置选项重名会在**构建 parser 时**抛
    `ArgumentError: conflicting option string`，于是 `polygraphy run --help`
    对该环境的所有人都坏掉

### 下一步

**A/B/C 三组 19 个候选已全部完成**（A3 并入 A2，其余 18 个各有一个目录，
编号 `More/01-` 到 `More/19-`）。

**CLI 侧的复查与补齐也已完成**（2026-08-27，见下节），`MultiDevice/` 也已建好。剩下的方向：
- **`MultiDevice/` 的 case 06：多卡实跑**。ONNX 改写部分（case 01~05）单卡已跑通并进 CI，
  缺的是每 rank 一个进程把 `model-TP_tp2_rank<i>.onnx` 建成 engine 跑在 GPU i 上，
  再与单卡 ONNX 的结果比对。**等多卡环境**
- `polygraphy plugin autotune`：需要 TensorRT 能真正加载并计时的插件，`toyPlugin` 只有 pattern
  没有实现，目前只抓了 help
- 把 B6 发现的 `LoadPlugins` 用 ctypes 加载不了 V3 插件这件事反馈给上游

## CLI 目录复查（2026-08-27）

10 个 CLI 目录在 TRT 11.1.0.106 + polygraphy 0.50.3 上**逐个实跑，rc 全为 0**，
问题不在"跑不了"，而在下面三类。已改完并重新生成了全部 `result-*.log`，
`tests/run_tests.py --case 07-Tool/Polygraphy/*` 11 个用例全过。

**实测：CLI 上已死但 help 里还在的选项**（用了才炸，这是最坑的）

| 选项 | 结果 |
| --- | --- |
| `--fp16` / `--bf16` / `--fp8` | `PolygraphyException: ... not available on TensorRT version 11.1.0.106` |
| `--int8` | `AttributeError: module 'tensorrt' has no attribute 'IInt8EntropyCalibrator2'` |
| `--precision-constraints` | 抛（`config.py:226 try_set_flag`） |
| `--layer-precisions <真实层名>:float16` | 抛；**写不存在的层名则静默 PASS** |
| `--save-tactics` / `--load-tactics` | `AttributeError: ... 'IAlgorithmSelector'` |
| `polygraphy debug precision` | 整个子命令死 |
| 仍可用 | `--tf32`、`--sparse-weights`、`--strongly-typed`、`debug build`、`debug repeat` |

连带：`inspect tactics` / `inspect diff-tactics` 在 TRT 11 上**永远拿不到输入文件**
（唯一生产者 `--save-tactics` 已死）。这三处的 help 抓取保留了，但 `unit_test.yaml` 里加了注释说明。

**修掉的硬伤**

- `Inspect/main.sh` 建了个 `model-trained-FP16.trt`，**命令里没有 `--fp16`**（删标志时漏删文件名），
  而且后面没人用它 —— 白建一个和上一个完全同精度的 engine。已删，改成 `--visual --save-visual` 那一步
- `Convert/data_loader.py` 是 INT8 校准步骤删掉后留下的孤儿（全仓库无人引用）。
  移到 `Run/`，用 `polygraphy run --data-loader-script` 救活；`Convert/` 的 `*.Int8Cache` 残留与 clean 规则一并处理
- `Plugin-TODO/` → **`Plugin/`**：原来 `MODEL_ADDSCALAR` 实际指向 `model-trained.onnx`，
  又往 `--plugin-dir` 拷了个 `.so`（`polygraphy plugin` 只读 `pattern.py`，从不加载 `.so`），
  三个日志全是 `{}`；且 `plugin match` 不给 `-o` 会把 `config.yaml` **写进 `00-Data/model/`**。
  现在 `build_toy_subgraph.py` 造一个能匹配 `toyPlugin` 的图（另植两个诱饵子图，让拒绝理由可见），
  `list`/`match`/`replace` 全部产出真实结果（`{'toyPlugin': 1}` → 5 节点折成 1 个 `CustomToyPlugin`），
  并补了 `unit_test.yaml`（此前 runner 根本不跑这个目录）
- `More/p3.py`~`p7.py`：B5 的临时探针脚本被误提交进仓库（`d8a00d36`），已删
- `Data/main.sh` 缺 `echo "Finish"`；`README.md` 里 `pip install polygraph\` 拼错
- `API/main.py` 那份 `CreateConfig` 全参数清单少了 **`runtime_platform`、`tiling_optimization_level`**；
  7 个已死参数逐个加了 DEAD 注释并指向 `More/` 对应目录
- `API/gs_workflow.py` 从空壳改成真的工作流：gs 把动态 batch 钉成 `[7,2,3,4]` →
  `fold_constants` 把整条 `Shape→ReduceProd→Gather→Concat` 折成常量，**8 → 2 节点**

**补上的覆盖缺口**（都能进 CI、单卡即可）

| 位置 | 新增 |
| --- | --- |
| `Surgeon/` | `weight-strip` / `weight-reconstruct`（重建出的是 **proxy 权重**，不是原值） |
| `Inspect/` | `inspect model --visual --save-visual`（81 KB / 100 KB 独立 HTML）。**坑：只给 `--save-visual` 不给 `--visual` 什么都不写、也不报错** |
| `Debug/` | `debug build --until 3 --artifacts`、`debug repeat`。**坑：产物进 `polygraphy_artifacts/{good,bad}/`，不是 `good/`、`bad/`** |
| `Data/` | `data concat`（沿迭代轴拼，与 `merge` 的输入/输出轴正交） |
| `Run/` | `--data-loader-script`（真实 MNIST 而非随机数）、`--warm-up`、`--check-error-stat`、`--postprocess y:top-1 --compare indices`（`More/02` 那个坑的 CLI 正解） |
| `Convert/` | `--convert-to onnx`（**没有 `--fold-constants`，会报 Unrecognized Options**）、`--fp-to-fp16`（只改权重与内部张量，I/O 仍是 float32，边界插 `Cast`） |

**`MultiDevice/`（2026-08-27 已建，多卡执行部分待办）**

关键发现：**切分是模式驱动的，CP 和 TP 找的根本不是同一个东西**
（`polygraphy/tools/multi_device/subtool/shard.py`）：

| | CP | TP |
| --- | --- | --- |
| 切什么 | **序列** | **权重** |
| 认什么模式 | `MatMul(Q,K)→Softmax→MatMul(·,V)` | **SwiGLU MLP**（以 `Sigmoid` 为锚），或 `AttentionPlugin` 节点 |
| 产物 | 1 个文件，initializer 不变，插 **6** 个 `DistCollective` | **每 rank 一个文件** `_tp<N>_rank<i>.onnx`，`w_gate`/`w_up` (8,16)→(8,8) 按列切、`w_down` (16,8)→(8,8) 按行切，插 **1** 个 `all_reduce` |

- `00-Data/model/` 里**没有任何模型含这两种模式**，所以 `build_transformer_block.py` 现造一个
  10 节点的 block（attention + SwiGLU MLP，`B=1 S=4 H=8 I=16`），且是能在 onnxruntime 跑的真模型
  —— 将来多卡比对时单卡参考是免费的
- **最大的坑：模型里没有那两个模式时，整条流程静默变成"复制一份"**。拿 MNIST 跑，
  hints 里 `attention_layers: []`、`inputs/outputs` 空，sharder 报成功，输出 12 节点 → 12 节点、
  0 个 collective，**全程无任何 warning**。要看的是 JSON 里的 `attention_layers`，不是退出码
- **rank 数来自 `--nb-rank` 而不是 `--gpus`**：只给 `--gpus 2`，TP 会老老实实写一个
  `_tp1_rank0` 的原样副本；`--gpus` 只填 `dist_collectives.group_size`
- `template shard-hints` 的 `-o` **必须 `.json` 结尾**，且这个检查发生在模型已经加载分析**之后**
- `shard --one-shot` 与「先 `shard-hints` 再 `-s hints.json`」结果一致（实测 16 节点 / 6 个
  collective 完全相同），只有三个输入张量的访问顺序不同
- `DistCollective` 是 NCCL 集合通信、不是标准 ONNX，onnxruntime 跑不了 → **case 06 必须多卡**，
  已在 `main.sh` 末尾写清缺什么、去哪找多卡管线（`05-Plugin/NcclPlugin/`、`08-Advance/MultiDevice/`）

## `07-Tool/trtexec` 复查（2026-08-27）

对照上游 `samples/trtexec/`（本地 checkout `/work/trt/TensorRT-GitHub`）与**实际安装的 trtexec
11.1.0.106** 逐项核对。**装的版本比目录里假设的新**：`Help.txt` 还是 v11.0.0.114 抓的，
而 README 里写着「11.2 才有、本机没跑过」的两个特性，11.1 就已经有了 —— 于是那两步真的被执行了，
**而且两步都是坏的**：

| 问题 | 证据 |
| --- | --- |
| 步骤 11 精度校验 **直接失败**，`set -e` 让整个 main.sh 中断（后面步骤全没跑） | `[E] When using --refPair, you need at least two pairs of I/O.` —— 例子里只给了一个 `--refPair=0` |
| 步骤 12 调优表达式**知识性错误** | `[E] Failed to parse --tuneBuildRoutes expression: Unknown knob: -builderOptimizationLevel` |

**tuner 的 knob 不是 trtexec 的 build 选项**，而是 209 个**编译器内部 knob**
（`--helpBuildRoute`，tuner_version 2.19.45），如 `-conv_lowering=[on|off]`、`-kgen:tiling=[0|1|2]`。
改对之后实测：`fast` 展开 4 条路线、`full` 展开 6 条；4 条路线 gpu_time 只差 **1.00%**
（28.36~28.64 us），即 MNIST 上花 1 分钟搜索**什么也换不到** —— 诚实结论写进 README。
tuning cache 是 JSON-lines，第一行 metadata 里的 `default_build_route` 是看 TRT 默认值的最好入口。

精度校验改对后补了三件事：`--accuracyThreshold` **是必填**（一旦用了 `--loadRefOutputs`/`--refPair`）；
单对时**不要**加 `--refPair`；以及五种算法在**同一份错配**上的读数完全不可比 ——
L0=1.0、L1=0.001432、L2=0.000003、LInf=0.003492、Cos=0.001497，
所以 `--accuracyThreshold=1e-3` 下 L2 **通过**而其余四个失败。还加了一个**故意失败**的用例
（喂 a 的输入配 b 的参考输出），否则「一直 PASS」证明不了校验在工作。

**关于「TRT 11 废除 FP16 等选项」**（本次任务的起点）：trtexec 侧**不需要删任何东西**，
main.sh 本来就没用；而且 trtexec 的处理比 polygraphy 干净得多 ——
`--fp16`/`--int8`/`--best`/`--precisionConstraints`/`--layerPrecisions` **在 help 里根本不存在**，
用了直接 `[E] Unknown option: --fp16`；polygraphy 则是 help 里还在、build 时才抛。已在两边 README 交叉引用。

**新增覆盖**（对齐上游 README 的 Example 5/6 + TRT 11 特性）：

- 步骤 13 `--stronglyTyped`：上游 Example 6 教你加这个 flag，**TRT 11 上是空操作** ——
  不加任何 flag 的默认 build 就打印 `Precision: Strongly Typed`（见 result-02.log）
- 步骤 14 多流吞吐（上游 Example 5）：`--infStreams` 1/2/4 → 15473 / 26030 / 42889 qps，
  中位延迟 0.0632 → 0.0737 → 0.0894 ms，**2.77x 吞吐换 1.4x 延迟**。
  （上游用的 `--streams` 仍被接受，但文档里已改名 `--infStreams`）
- 步骤 15 权重剥离：13,250,084 B → 157,308 B（**84x**）。
  **坑：剥离后的 engine 照样能加载、能跑、报 PASSED，输出全 0，没有任何 warning**；
  且 CLI 侧 `--refitFromOnnx` 不打印任何日志、输出仍是 0，`--dumpRefit` 什么都不输出 ——
  能用的是 Python 路径 `04-Feature/WeightStripping/`（同一模型，refit 后输出完全恢复）

其余：`Help.txt` 重新抓（v11.0.0.114 → 11.1.0.106）；`unit_test.yaml` 加 `slow` 标签与
`timeout: 3600`（tuner 每条路线 fork 一个子 trtexec）。
上游 `tracer.py` 用的 `startInMs`/`inMs`/`outMs` 与现版 trtexec 的 `startH2dMs`/`h2dMs`/`d2hMs`
不一致这件事，之前 `parse_export_json.py` 已经兼容，README 里保留并补齐成一张「上游哪些地方过期了」的表。

## `07-Tool/TritonServerDeploy`（2026-08-27）

原来只有一个 `print("Finish")` 的空壳，现在是完整的部署范例：`main.py` 按
`start_server` / `wait_for_server` / 客户端请求 / `finally` 收尸的形状分四个阶段。

| 阶段 | 需要 | 本容器 |
| --- | --- | --- |
| 1 建 engine + 铺 model repository | `tensorrt` | ✅ |
| 2 本地跑一遍拿参考答案 | `+cuda-python` | ✅ |
| 3 起 tritonserver → KServe v2 请求 → 与参考比对 | **`tritonserver` 二进制** | ⏭️ 跳过 |
| 4 关服务、不留进程 | — | 属于阶段 3 |

**本容器是 `nvcr.io/nvidia/pytorch`，没有 tritonserver 也没有 docker。**
查证过：PyPI 上的 `tritonserver` 包只有 Python binding 的 `.so`（整包 25 个文件，
无 `libtritonserver.so`、无 backend），`pip install` 装不出服务端。
**但后来把服务端弄起来了，四个阶段全部实跑通过，见下一节。**

为了不让「跑不了的那一半」烂掉（正是 trtexec 里抓到的那类问题），加了 `test_client.py`：
用只讲 KServe v2 的 stub server 顶替 tritonserver，驱动**真实的客户端代码** ——
就绪轮询（前两次故意 503）、metadata、infer 请求编码/响应解码、与参考比对、
`finally` 里的 terminate，**16 项检查全过**。测不到的只有「Triton 认不认这份 `config.pbtxt`」，
作为部分替代，最后一个 case 拿 config 与 engine 的真实张量形状交叉校验。

`config.pbtxt` 的三个坑（都写进 README）：
- `max_batch_size > 0` 时 **Triton 占用第一维**，`dims` 写单样本形状
  （engine 的 `x` 是 `(-1,1,28,28)`，config 里写 `[1,28,28]`）—— 测试专门校验这条
- 每样本标量（输出 `z`）不能写空 `dims`，要 `dims: [1]` + `reshape { shape: [] }`
- `max_batch_size` 不能超出 engine 的 profile（本例 min1/opt4/max16）

客户端用纯 `requests` 走 KServe v2 JSON（不引入 `tritonclient`，方便直接拷进 Triton 容器），
README 里给了 `tritonclient` 的等价写法。另强调 **plan 与 Triton 容器的 TensorRT 版本必须一致**，
否则 `Engine plan file is generated on an incompatible version`。runner 用例通过（14s）。

### 免 docker、免 root 跑起 tritonserver（2026-08-27 实测成功）

**不需要编译源码**。容器镜像本质就是 HTTP API 后面的一堆 tar，NGC 对
`nvidia/tritonserver` 发匿名 token，所以直接把需要的两个目录扒出来就行。
脚本固化在 `07-Tool/TritonServerDeploy/install-tritonserver-without-docker.sh`：

| 步骤 | 要点 |
| --- | --- |
| 匿名 token | `https://nvcr.io/proxy_auth?scope=repository:nvidia/tritonserver:pull`。**token 约一分钟就过期**，必须每个 blob 重新取；用过期 token 拿到的是 JSON 错误体，`tar` 报 "not in gzip format" —— 我一开始就是被这个坑了，白扫了好几层 |
| 选镜像 | `26.07-py3` 的 `TRT_VERSION=11.1.0.106`，**与本容器的 TensorRT 完全一致** —— 这才是 main.py 建的 plan 能被加载的原因 |
| 解包 | 46 层里只有一层（1.9 GB）含 `opt/tritonserver`，只取 `bin/`、`lib/`、`backends/tensorrt/` → 装完 **53 MB** |
| 补两个库 | 二进制还缺 `libb64.so.0d`（Ubuntu universe）与 `libdcgm.so.4`（CUDA 源的 datacenter-gpu-manager-4-core），从 `.deb` 里 `dpkg-deb -x` 取出即可，**不需要 root** |
| 写脚本时踩的两个 shell 坑 | ① `set -o pipefail` 下 `tar -tzf ... \| grep -q` 中 grep 提前关管道 → tar 被 SIGPIPE 打死 → 整条管道返回 141，于是**真正含服务端的那一层被判成没有**（要先把清单落盘再 grep）；② `find A -o -name B -exec cp` 的 `-exec` 只绑定到最后一个 `-name`，`libb64` 被找到但没被拷 → 服务端起不来（要加括号） |
| `--backend-directory` | 默认写死 `/opt/tritonserver/backends`；装到别处会报 `unable to find backend library for backend 'tensorrt'`，**报错里不提目录**。已让 `main.py` 从二进制路径推出这个参数 |

脚本从零跑一遍：装完 **60 MB**，`All shared libraries resolve`。实测端到端（H100 / TRT 11.1.0.106 / Triton 2.71.0-26.07）：
`Server ready after 1.0 s` → HTTP 推理 `13.14 ms` → 与本地 TRT 参考
`max |diff| = 0.000e+00, same argmax = True` → `Server stopped with code 0`。

两条限制：宿主 glibc 要够新（本机 Ubuntu 24.04，与镜像一致）；这**不是受支持的安装方式**，
除 `bin/`/`lib/`/一个 backend 外什么都没有（Python / PyTorch / ONNX-Runtime backend 都不在），
要跑多后端还是得用容器。源码编译（`build.py --no-container-build`）也可行但要拉整条工具链，
在已经验证扒包可行的情况下没必要。

## FP16 ONNX 用量统计（2026-08-27）

**`00-Data/model/` 里一个 FP16 ONNX 都没有**（`model-half-mnist.onnx` 的 "half" 是「半张图」
不是半精度，命名本身误导）。需要它的地方分四类：

**A. 已经坏了的**（用了 TRT 11 已删除的 `trt.BuilderFlag.FP16`，一跑就 `AttributeError`；
`trt.BuilderFlag` 现在只剩 `TF32`）。五个都是 `enabled: false`（框架没装），**CI 抓不到**：

| 例子 | 状态 |
| --- | --- |
| `03-Workflow/JAX-ONNX-TensorRT` | `case_normal(is_fp16=True)` 活跃 |
| `03-Workflow/Mindspore-ONNX-TensorRT` | 活跃 |
| `03-Workflow/Paddlepaddle-ONNX-TensorRT` | 活跃 |
| `03-Workflow/TensorFlow2-ONNX-TensorRT` | 活跃 |
| `03-Workflow/OneFlow-ONNX-TensorRT` | 同样代码，fp16 那行已注释 |

**B. 名字叫 FP16、其实是 FP32 的**
- `07-Tool/trex/get_data.py` 的 `model.fp16.*` 用**普通 FP32 ONNX** 建（注释里也承认精度由图决定），
  于是 CompareEngines 实际在比 INT8-QAT vs FP32
- 顺带发现两处**写死的旧仓库路径**（少了 `-wili`，目录不存在）：
  `07-Tool/trex/get_data.py:36`、`07-Tool/trex/11-ProcessEnginePipeline/main.py:45`

**C. 自己会造 FP16 ONNX 的 —— 现成参考实现**

| 例子 | 工具 |
| --- | --- |
| `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` | **`modelopt.onnx.autocast`**，含 `op_types_to_exclude` |
| `07-Tool/FP16Tuning` | **`modelopt.onnx.autocast.convert_to_mixed_precision`**，逐节点搜索 |
| `07-Tool/Polygraphy/Convert`（今天新增） | `polygraphy convert --fp-to-fp16`（底层 onnxconverter_common） |

**D. 不需要 FP16 ONNX 的**（走 TRT network API / 插件 / torch）：`02-API/CudaEngine`、
`02-API/Layer/Cast`、`04-Feature/DataFormat`、`05-Plugin/UseFP16`、`05-Plugin/CuteDSLPlugin`、
`Polygraphy/More/05`、`More/13`，以及 `tests/NetworkSerialization`、`06-DLFrameworkTRT/Torch-TensorRT`。

**未验证**：`06-DLFrameworkTRT/ONNXRuntime-TensorRT` 的 `trt_fp16_enable`（ORT-TRT EP 选项）在
TRT 11 上是报错还是静默忽略 —— 本容器的 onnxruntime 加载不了 TRT EP
（`Please install TensorRT libraries...`），需要对 TRT 11 构建的 onnxruntime-gpu 才能定论。

**缺口**：最主线的 `03-Workflow/pyTorch-ONNX-TensorRT` **完全没有 FP16 case**。

**三条生成路径的产物不一样，不是口味问题**：

| 路径 | 产物 | I/O dtype |
| --- | --- | --- |
| `modelopt.onnx.autocast` | 混合精度，敏感节点留 FP32，无需校准数据 | FP32（边界插 Cast） |
| `polygraphy convert --fp-to-fp16` | 权重与内部张量全 FP16（实测：8 个 initializer 转 FP16、多 2 个 `Cast`） | FP32（边界插 Cast） |
| `torch.onnx.export(model.half(), ...)` | 纯 FP16 | **FP16** |

**建议**：默认用 **ModelOpt AutoCast** 生成 `00-Data/model/model-trained-fp16.onnx`
（cookbook 已有两处在用、保留敏感节点、不需要校准数据）；若还要覆盖「I/O 就是 FP16」的场景
（对应 `05-Plugin/UseFP16`、`DataFormat` 那类），再用 `torch.half()` 导出一个
`model-trained-fp16-pure.onnx`，两个文件把「混合精度」与「纯半精度」两种形态都摆出来。
A、B 两类**尚未动手修**。

## `08-Advance/EmptyTensor`（2026-08-27）

原来是 `EmptyTensor-TODO` 空壳，现在是**围绕真实场景**的例子：检测器按分数过滤后一个框都不剩、
推理服务塞进来一个 0 行的 batch —— 这两件事天天发生。TensorRT 本身处理得很好，
例子讲的是**外围程序**出错的三种方式（全部实测于 TRT 11.1.0.106）：

| case | 结论 |
| --- | --- |
| `case_no_detection` | `Greater`→`NonZero`→`Gather` 的检测尾巴。没有框通过时输出 `(0,4)`，同一个引擎同一个调用在 3 个框通过时输出 `(3,4)` —— **空结果是正常结果**，不需要 host 侧的 if 分支 |
| `case_empty_tensor_needs_a_valid_address` | **`cudaMalloc(0)` 成功但返回地址 0**；把 NULL 绑成输入后 `enqueueV3` 返回 `False` 且什么都不跑，输出缓冲区保持原样 |
| `case_reduce_over_empty_axis` | 空轴归约：`SUM=0`，但 **`MAX=-inf`、`AVG=NaN`** |
| `case_profile_must_cover_zero` | shape 里的 0 必须落在 profile 内，`min=1` 时 `set_input_shape([0,2])` 返回 `False` 并保留旧 shape |

**这个例子自己踩了自己要讲的坑**：第一版第三个 case 是通过 cookbook 的 `TRTWrapperV1` 跑的，
而 `utils_class.py` 的四个缓冲区分配处都写着 `cudaMalloc(n_byte)` —— 张量为空时 `n_byte=0`、
拿到 NULL 地址、`enqueueV3` 拒绝执行、输出缓冲区没被写过，于是读回来是三个 0，
差点得出「MAX over nothing = 0」这个**错误结论**。绑对之后才是 `-inf` / `NaN`。
已把 `tensorrt_cookbook/utils_class.py` 的 **4 处** `cudaMalloc(n_byte)` 改成
`cudaMalloc(max(n_byte, 1))` 并加注释（这是共享代码的真 bug，任何喂空张量的例子都会中招）。
回归：`01-SimpleDemo`、`02-API/Layer/NonZero`、`04-Feature/DebugTensor`、
`08-Advance/MultiOptimizationProfile` 均通过。

顺带修了 `04-Feature/EmptyTensor-TODO/unit_test.yaml` 里 `name:` 写成 `08-Advance/EmptyTensor`
的复制粘贴错误（会与本例重名）。

## `08-Advance/MIG`（2026-08-27，决定不写成例子）

`MIG-TODO/` 里那 20 行（跑 `nvidia-smi -L`、数实例、打印一句 `export CUDA_VISIBLE_DEVICES=`）
已删除，目录改名 `MIG/`，**只留一个 README**。理由：MIG 是宿主机配置，
**进程里看一个切片就是一张小卡，没有任何 TensorRT API 因它而不同**，写成例子等于抄 nvidia-smi 手册。

**唯一属于 TensorRT 的那条**（已写进 README）：**engine 要在将要部署的那个 MIG profile 上 build**。
TRT 在 build 时按当时可见的 SM 数与显存挑 tactic（tile 尺寸、split-k、occupancy），
本机实测 profile 两端差 8 倍：`7g.80gb` = 114 SM / 79.25 GiB，`1g.10gb` = **14 SM / 9.75 GiB**。
整卡上 build、切片上服务的 engine 照样加载、照样算对，**只是 tactic 选择彻底盲了**，
而且没有任何东西会报告 —— 最难查的那类性能 bug。`--memPoolSize=workspace` 同理。

**未做的测量**（等有开了 MIG 的机器）：同一个 ONNX 分别在 `7g.80gb` 和 `1g.10gb` 上 build，
都放到 `1g.10gb` 上跑，比吞吐。差距明显则升格为例子；差距在噪声内同样是有用结论，笔记就保持笔记。

本机实测记录：H100 PCIe 80GB **支持 MIG 但 `mig.mode.current=Disabled`**；
容器内 `nvidia-smi -i 0 -mig 1` 报 `Insufficient Permissions`（非 root，且切换 MIG 模式/建实例
本就是宿主机操作，要求 GPU 上无进程）。容器只能在 `docker run` 时被分到已有切片，
**运行期不能重新分区**；切片间 `P2P: No`，NCCL 那套不适用。

## `08-Advance/GreenContext`（2026-08-27，MIG 的进程内替代品）

问「值不值得给 MIG 写例子」时顺出来的方向：**green context（CUDA 12.4+）把一张卡的 SM 切开，
给出一个绑定到分区的 stream，在那个 stream 上启动的一切都被限制在这些 SM 里**。
对 TensorRT 而言不需要任何新 API —— `execute_async_v3(green_stream)` 就是全部集成，
而这正是它危险的地方。全部实测于 H100 PCIe（114 SM）/ TRT 11.1.0.106 / CUDA 13.3。

**1. 分区是真的，而且免费**（8 层 1024×1024 matmul，SM bound）

| stream | 延迟 | 相对整卡 | SM 比 |
| --- | --- | --- | --- |
| default（114 SM） | 0.137 ms | 1.00x | 1.00x |
| green 16 SM | 0.546 ms | 3.98x | 7.12x |
| green 32 SM | 0.299 ms | 2.17x | 3.56x |
| green 64 SM | 0.166 ms | 1.21x | 1.78x |
| **green 114 SM** | 0.141 ms | **1.02x** | 1.00x |

最后一行是关键：**切出全部 SM 的分区不花钱**，机制本身零开销。切分有粒度
（`minSmPartitionSize` = `smCoscheduledAlignment` = 8），要按返回值算实际拿到多少。

**2. 用途：吵闹邻居**（同进程内一个延迟敏感 engine + 一个吃满吞吐的 engine）

| 场景 | median | p95 |
| --- | --- | --- |
| 单独跑 | 0.036 ms | 0.038 ms |
| 有背景任务，都在 default stream | 0.161 ms（**4.53x**） | 0.309 ms（**8.23x**） |
| 有背景任务，green 32 SM / 82 SM | 0.066 ms（1.86x） | 0.115 ms（3.05x） |

**这就是 MIG 的招牌实验，但不用 MIG**：同进程、免 root、免宿主机配置、免重启容器，随建随拆。

**3. 隔离有个洞：TRT 的 aux stream 会逃出分区**（本例存在的主要理由）

背景 engine 被钉在自己的 82 SM、延迟任务在不相交的 32 SM 上，然而：

| 背景 engine 的 build 选项 | 延迟 median | p95 |
| --- | --- | --- |
| `max_aux_streams = 0` | 0.064 ms（1.16x） | 0.103 ms（1.79x） |
| `max_aux_streams = 4` | 0.117 ms（2.15x） | 0.222 ms（**3.86x**） |
| **默认（-1，TRT 自己定）** | 0.112 ms（2.04x） | 0.211 ms（**3.66x**） |

TRT 的 aux stream 是从**当前 context** 创建的，不是 green context —— 而默认值就是 -1，
**不需要谁主动要求就会发生**。要隔离就得用 `max_aux_streams = 0` build，代价是牺牲引擎内并发。

**4. 白捡的 19%：在哪跑就在哪 build**

| engine | 在 32 SM 上的延迟 |
| --- | --- |
| 整卡上 build（第 1 次） | 0.301 ms |
| 整卡上 build（第 2 次，对照组） | 0.301 ms |
| **在 32 SM green context 里 build** | **0.243 ms** |

两次相同条件 build 只差 **0.001 ms**，而分区内 build 的效应是 **0.058 ms（19%）= 抖动的 58 倍**。
**我第一次得出的结论是错的**（早期原型显示"完全相同"），是加了对照组才判定的 —— 对照组现在留在例子里。

机制值得记：**TRT 问不到分区大小** —— 它读的 `cudaGetDeviceProperties.multiProcessorCount`
在 green context 里**仍然报 114**，只有驱动层的 `cuCtxGetDevResource` 知道是 32。
真正适应分区的是**经验式的 tactic 搜索**：候选 kernel 是在当前 context 里计时的，
所以在分区内 build 就量到了分区的真实行为，选出了不同的赢家。
这条正好是 `MIG/README.md` 里那句「在部署用的 profile 上 build」的**实测版**。

**5. 生命周期坑**：在 green context 里 build 的 engine 持有该 context 的 CUDA 资源，
先销毁 context 会让 TRT 析构函数报 `Error Code 1: Cuda Runtime` 然后**在进程退出时 SIGSEGV**，
离出错点很远。case 4 因此故意不释放分区。（case 1~3 可以正常释放：它们的 engine 在 primary
context 里，只有 stream 来自分区。）

**测量方法学教训**：中途有一次崩溃留下的僵尸进程占着 GPU，导致基线从 0.055 ms 变成 1.446 ms、
整组数字失真；发现后 kill 掉重测。跑这类隔离实验前必须先 `nvidia-smi --query-compute-apps` 确认卡是干净的。

## `08-Advance/TensorRTGraphSurgeon`（2026-08-27）

原来只有一个空 README。现在做的是**在 `INetworkDefinition` 层面动刀** —— ONNX parser 之后、
builder 之前，与 `07-Tool/OnnxGraphSurgeon`（parser 之前动 ONNX）互补。
实测于 H100 / TRT 11.1.0.106，模型 `x -> Add(1.0) -> Relu -> Mul(2.0) -> y`，张量 [8,512,512]。

**能用的 API 就这么点**：`num_layers`/`get_layer` 走图、`add_*` 加层、`layer.set_input` 改连线、
`mark_output`/`unmark_output` 挪边界，**没有 `remove_layer`**。

| case | 结论 |
| --- | --- |
| 1 走一遍 parse 结果 | 3 个 ONNX 节点变成 **7 个 TRT 层**；**ONNX 节点名活了下来**（`node_add` 等），这是后面能按名字找层的前提；多出来的 CONSTANT/SHUFFLE 是 parser 在广播标量 |
| 2 追加一层 | `unmark_output` + 加 NEG + `mark_output`；忘了 unmark 不会报错，只是网络变成两个输出 |
| 3 删一层 | 没有删除 API，做法是**把消费者接到生产者的输入上**，让 builder 丢掉没人读的层。`num_layers` 仍是 7（层对象还在），**证据在数字里**：Relu 没了之后出现负值 -4.0 |
| 4 换成自己的 plugin | 用户点名要的那个：**ONNX 本来就能跑**，仍想换成自己的实现。`add_plugin_v3` + `set_input` 改线即可，**输出逐位相同** |

**case 4 的代价必须量**：engine 层数 1 → 2，kernel 时间 0.0060 → 0.0128 ms = **2.13x**。
这不是 plugin 本身慢，而是 TRT 原本把 `Add+Relu+Mul` 融成了**一个 kernel**，plugin 融不进去，
一趟内存访问变成两趟 —— 在带宽受限的链上正好约 2 倍。**换可融合的算子要先算这笔账。**
（计时故意绕开 `TRTWrapperV1.infer`：它的 H2D/D2H 拷贝是 kernel 时间的 ~150 倍，会把效应全盖住。）

**两个生命周期坑，都是撞出来的**：
- **插件库加载后，先建一个 engine 再释放它（重新绑定变量就够），下一次
  `trt.get_plugin_registry().get_creator(...)` 会 SIGSEGV** —— 可复现、无 Python traceback、
  崩在 libnvinfer 里。我用 A~H 八种顺序做了二分才定位到触发条件（H 崩、G 不崩：区别只在于第一个
  wrapper 是否已被释放）。例子因此**先加载库、先取 plugin 对象，并把两个 wrapper 都留着**。
  只记录触发条件与规避方法，未下根因结论。
- 改线之后只能靠**输出数值**验证，不能看层数（层数不变）。

与 `GreenContext` 的坑正好是镜像：那边是先销毁 green context 导致 TRT 析构崩。

## `90-Misc/Number` 数据类型表复查（2026-08-27）

对照 `torch` 2.13、`ml_dtypes` 0.5.4 的 `finfo` 与逐 bit `view()` 复查 `mannuscript.md` 和
`output/*.md`。**所有数值结论均为实测，不是查文档抄的。**

### `mannuscript.md` 中改掉的错

| 位置 | 原值 | 实测 | 说明 |
| ---- | ---- | ---- | ---- |
| FP8E8M0 `0x80` | `2^0=1.0` | **2.0** | bias 127，`0x7F` 才是 1.0 |
| FP8E8M0 NaN | ❌ | **✓** | `0xFF` 是唯一的 NaN（`fnu` 的 `n` 就是 NaN） |
| FP8E4M3 Inf | ❌ | **✓** | 该行描述的是 IEEE 风格的 `ml_dtypes.float8_e4m3`，它保留 Inf、上限 240 |
| FP8E4M3 "OCP standard" | 该行 | **移到 fn 行** | OCP OFP8 的 E4M3 就是 max=448 的 `fn`，不是 240 那个 |
| UINT64 `0x80` | 1.15e+19 | **9.22e+18** | $2^{63}$ |
| Min Positive 列 | 混用 | 拆成 **Min Normal / Min Subnormal** | 原表 FP64 给的是 min normal、FP4E2M1 给的是 min subnormal（0.5），同一列两个含义 |
| FP9E5M4 max | 57344 | 63488 | 抄了 E5M2 的上限；该行连同 E5M0 一并改对 |
| FP7E4M3 | 7 bit / 1-4-3 | **删除** | 1+4+3=8，位数自相矛盾，且查不到这个格式 |
| NVFP8 | 独立一行 | **删除** | NVIDIA 没有叫 NVFP8 的格式；FP8 是 per-tensor scale，不是 block 格式，已在概念表里写清 |
| MXFP8 max | 57344 | 448 或 57344 | OCP 允许 E4M3fn 与 E5M2 两种 element |

补充：加了 **FP6E3M2 / FP6E2M3 / MXFP6** 三行（`output/` 里本来就在生成 FP6，表里却没有）；
FP4E2M1 的 PyTorch 名字是 `float4_e2m1fn_x2`（**打包是类型的一部分**）；新增一节讲 E4M3 的三个
变体（`e4m3` / `e4m3fn` / `e4m3fnuz`，同一 layout 三套取值），以及 `fn`/`uz`/`u` 后缀的含义。

### `output/*.md` 中改掉的错（都改在 `build-number-md.py` 里）

- **FP8E8M0 的 `0xFF` 印成了 `3.402824e+38`**，应为 NaN；最大值是 `0xFE` = 1.70e38。根因是 spec 里
  写了 `has_nan=False`。
- **FP4E2M1 的「Largest number < 1」印成 `1 - 2^{-2}` = 0.75**，而 FP4E2M1 根本没有 0.75。原代码只给
  FP6E2M3 开了特例，其实条件是 `q == 2`（bias=1 时 1.0 下面一档就是次正规数）。
- **IEEE 格式的 NaN 行重复**：qNaN 与「NaN」是同一个 bit pattern，改成非 IEEE 格式才印。
- **`E = 2 - 2^{q-1} = -0`**：q=2 时模板里硬写的负号，改成带符号整数。
- **`Integer.md` 根本不是生成的，而且是错的**：UINT8/INT8 行填的是 16 位的范围（65535 / 32767 /
  -32768），INT16/UINT16 整行缺失，INT64 最小值被抄成 `-9223372036854775800`（末位截断），INT2 最小值
  写成 -1。改为由 `build_integer_md()` 现算。
- `build-number-picture.py` 里 `FP32(E11M23)`、`FP16(E5M11)` 两个标签写错，且 FP16 的位宽给成
  `[1,5,11]`（=17 bit），图会画错。
- `README.md` 里的运行命令写成了 `build_number_md.py`（下划线），实际文件名是连字符。

### 验证

把重新生成的 6 张全值表（FP8E4M3/E5M2/E8M0、FP4E2M1、FP6E2M3/E3M2）逐 bit pattern 与 `ml_dtypes`
对拍：**256/256、16/16、64/64 全部一致，NaN/Inf 位置也一致**。

### torch 有没有更新的浮点类型？

没有。当前（`torch` 2.13 与 main 文档）全部浮点 dtype 是：`float64/32/16`、`bfloat16`、
`float8_e4m3fn`、`float8_e5m2`、`float8_e4m3fnuz`、`float8_e5m2fnuz`、`float8_e8m0fnu`、
`float4_e2m1fn_x2` —— 表里都已覆盖。**没有 `float6_*`**（只有 `ml_dtypes` 有 `float6_e2m3fn` /
`float6_e3m2fn`），也没有裸的 `float8_e4m3`。另外本机 build 里有 `int1..int7` / `uint1..uint7`
这些 shell dtype，以及一个新的复数类型 `bcomplex32`（bfloat16 复数），与本表无关。

## `99-Todo/candidates-eco-tensorrt-llm.md` 复查（2026-08-28，结论：关档）

按「TensorRT-LLM 里有哪些**跟 TensorRT 相关的工具**可以搬进 cookbook」这个标准复查，**原文 16 条候选
全部作废**，且没有新的可搬项，文件改写为关档说明。

- 原文 16 条（quickstart / async / streaming / sampling / 投机解码 / KV-cache 配置 / multi-LoRA /
  guided decoding / 多模态 / trtllm-serve / chat REPL / eval）教的都是 `tensorrt_llm.LLM` 这套
  **PyTorch 后端**的 API，没有一条讲 TensorRT。属于「在 cookbook 里把 TRT-LLM 跑起来」，标准不符。
- 更关键的是**这个仓库的 TensorRT 面已经几乎没了**（实测，非推断）：
  - 全仓非 3rdparty 的 Python 文件里，**只有 9 个还 `import tensorrt`**（注意：直接 grep
    `^import tensorrt` 会把 `import tensorrt_llm` 一起匹配进来，得到 299 个的假象）。
  - 建 engine 那一层整体删除：`builder.py`、`network.py`、`module.py`、`python_plugin.py`、
    `tools/plugin_gen/{core,plugin_gen,shape_infer}.py`、`tools/onnx_utils.py` 都还列在
    `legacy-files.txt` 里但磁盘上不存在（该文件 1301 条里有 451 条已失效）。
  - `_deprecation.py::emit_engine_arch_deprecation` 专门用来在残留的 engine 路径上打
    「legacy TensorRT engine-build workflow，改用 PyTorch backend」。
- 因此三个本来最值得看的插件例子**全部 import 不起来**：`examples/python_plugin/`（依赖已删的
  `tensorrt_llm.python_plugin`，那个 `@trtllm_plugin` + `PluginBase` 的纯 Python 插件写法是这里唯一
  真正 cookbook 形状的点子）、`examples/openai_triton/manual_plugin/`（依赖已删的 `module/builder/
  network`，C++ 侧还是 TRT 11.1 里标了 `TRT_DEPRECATED` 的 `IPluginV2DynamicExt`）、
  `plugin_autogen/`（依赖已删的 `tools/plugin_gen`）。
- 仍能 import 的 5 个只是普通的 torch → ONNX → TRT 建引擎/跑引擎（qwenvl 的 ViT、qwen2audio），
  `03-Workflow/pyTorch-ONNX-TensorRT` 已覆盖且更干净。`scripts/` 全是仓库自身的构建/lint 管道。
- **想要「用 Python 写 TRT 插件」的例子，去 TensorRT OSS 的 `samples/python/python_plugin` 拿**
  （已记在 `candidates-github.md`），不要从这里拿。
- 顺带更新：`candidates-ecosystem.md` 的跨仓表里删掉 6 条 TRT-LLM 行（原 #1/#2/#8/#11/#12/#14）并改写
  主题 1；`99-Todo/README.md` 的生态表把 TRT-LLM 改成 No action。

## `05-Plugin/TritonAOTPlugin`（2026-08-28）

把 TensorRT-LLM 那两个已经跑不起来的例子（Triton kernel 手工包成 C++ 插件 / 自动生成插件）**从零重写**，
不依赖它的任何实现。用的是 Triton 自带的 AOT 工具链（`triton.tools.compile` + `link`，本机 3.7.1
可用），插件按 cookbook 的 `IPluginV3` 写法。

四个 case：

1. 手写插件。AOT 把 kernel 编成**内嵌 cubin 的 C 源码 + `cuLaunchKernel` 包装**，`link` 再给它一个
   不带 hash 的稳定符号名（每个 variant 的文件名里带 content hash，改 kernel 就变，所以必须调
   linker 出来的 `add_scalar_default`）。
2. **实测出 Triton AOT 的一个静默算错 bug**：`fp32` 的 kernel 标量参数，生成的 C 原型写成
   `double`，却把 `&scalar` 直接塞进 `cuLaunchKernel` 的参数数组——kernel 那一格只有 4 字节，于是读到
   double 的低半部分。`1.0` 是 `0x3FF0000000000000`，低半部 = 0，**结果变成 `x + 0`，launch 还返回
   `CUDA_SUCCESS`**。`0.1` 会变成 `-1.59e-23`。修法是 `sed 's/double scalar/float scalar/'`，同时
   调用方声明也要跟着从 `c_double` 改 `c_float`（只改一半仍然错，我第一次就踩了）。case 2 用纯
   ctypes 复现，并对「坏」和「好」两种都下断言，将来 Triton 修了这个 bug 例子会直接失败而不是留着
   一个没用的 sed。
3. 证明「AOT = 不再依赖 Python/Triton」：子进程里把 `import triton` 屏蔽掉重新建 engine 并跑，
   数值正确；`readelf -d` 只有 libcudart/libcuda/libstdc++/libgcc/libc（顺带发现 **libnvinfer 也不在
   NEEDED 里**，插件只实现接口，TRT 加载时才解析）。
4. `plugin_gen.py`：从一个 9 键的 spec 生成 339 行 C++ + Makefile 并编译。故意换成 **GELU**（不同
   kernel、不同参数表、没有 attribute）来证明生成器不是把隔壁那个文件抄一遍。更重要的是把**多个 AOT
   variant 映射成 TensorRT tactic**：3 个 `BLOCK_SIZE`/`num_warps` 组合 → 3 个 tactic，builder 计时后
   选中 `0x3`（BLOCK=4096/warps=8），**比固定用第一个 variant 快 1.83x**（0.0134 vs 0.0245 ms，两次
   运行都是 1.83~1.84x）。等于把 Triton 的 autotune 挪到 build 期，部署进程里没有 autotuner。

踩到并记录的坑：

- **插件库必须同时导出 `setLoggerFinder` 和 `getCreators`**。只导出后者不报链接错误、`loadLibrary`
  也正常返回 handle，但 creator 根本不会注册，错误延迟到 `Cannot find plugin: ...` 才爆出来。
- 带 `:16` 对齐提示时，linker 生成的 dispatcher 会检查 `ptr % 16 == 0`，不满足就**不发射 kernel**
  直接返回 `CUDA_ERROR_INVALID_VALUE`——不查返回值的话输出 buffer 保持原样。
- `TacticValue` 只有在 `profiling_verbosity = DETAILED` 时才出现在 engine information 里。
- Makefile 里 `$(wildcard)` 对「recipe 运行时才生成的文件」无效（目录列表在 parse 期就缓存了），
  要用 shell 的 `$$(ls ...)`。
- 另外看到但没动的两处 Triton 生成代码缺陷：`assert(algo_id < sizeof(kernels))` 用的是字节数不是元素
  个数；launcher 在 `gX*gY*gZ == 0` 的路径上没有 `return`。

定位：最初按要求放在 `08-Advance/`，发现 `05-Plugin/PythonPlugin/add_scalar_triton.py` 已经有一个
**JIT** 版的 Triton 插件之后，改放 `05-Plugin/`（同一主题的 JIT / AOT 两面，一个用于开发、一个用于
部署，README 互相链接）。tag 也从 `advance,plugin,compile` 改成 05-Plugin 惯例的 `plugin,compile`。

## `07-Tool/nvtriPy` + 两个 candidates md 关档（2026-08-28）

### 先说一个我自己造成的事故

为了看 Tripy，我直接 `pip install nvtripy` 装进了 cookbook 的环境，**把环境搞坏了**：

```
Successfully installed ... mlir-tensorrt-*-0.1.43+cuda12.trt109 numpy-1.26.0
                          nvtripy-0.1.7 tensorrt-cu12-10.16.1.11 ...
>>> tensorrt.__version__  →  '10.16.1.11'   （原本 11.1.0.106）
>>> numpy.__version__     →  '1.26.0'       （原本 2.1.0）
```

nvtripy 不是纯 Python 包，它依赖 `tensorrt-cu12 10.x` + `mlir-tensorrt ... cuda12.trt109`，装进来
就把系统的 TensorRT 11 顶掉了。已全部卸载回滚（`tensorrt` 恢复 11.1.0.106、`numpy` 恢复 2.1.0、
`colored` 恢复 2.3.2），并用 `05-Plugin/BasicExample` + `02-API/Layer/Cast` 跑通验证环境完好。
**教训：任何 `pip install` 之前先确认它会不会动 tensorrt/numpy，装第三方 TRT 前端一律进 venv。**

### 例子本身

这个坑反而成了例子的主线。`07-Tool/nvtriPy/` 由 `main.py`（驱动，建私有 `.venv` 并在里边跑）+
`tripy_cases.py`（真正的 Tripy 代码，只能在 venv 里跑）组成，开头就打印两边的 TRT 版本证明隔离：
cookbook 解释器 TensorRT 11.1.0.106，`.venv` 里 TensorRT 10.16.1.11 + nvtripy 0.1.7。

四个 case（对应原 md 的 #1/#7、#3、#2）：

1. **eager → compile**：同一个 `tp.Module` 先即时跑再 `tp.compile`。**发现两者并不一致**：
   相对差 9.8e-05。定位后是 **matmul**：纯 elementwise 图（gelu + 乘法）逐位相同，单个 `tp.Linear`
   则不同，而且**偏的是 eager 那边**（精确值 0.28，eager 给 0.28000974655151367，compiled 给
   0.2800000011920929，TF32 量级）。所以 eager 适合查形状和逻辑，不适合验证部署前的最后几位数。
2. **惰性求值**：定义张量 2.7 ms，第一次 `.eval()` 几百 ms（编译发生在这里），第二次 0.017 ms。
   直接给定义计时会严重低估。同源的表现：`Executable` **拒绝**未求值的输入而不是替你求值。
3. **动态形状**：`InputInfo(shape=((1,4,8), 4))` 就是 min/opt/max，一个 Executable 服务 batch 1/4/8；
   batch 9 报错，而且底下透出来的是 TensorRT 自己的 `satisfyProfile` 消息（`Valid range for
   profile 0: [1,4]..[8,4]`）——出范围是报错，不是偷偷重建。
4. **save/load Executable**：编译 176.5 ms vs 加载 2.4 ms（54 KiB），**74x**，输出一致。

原目录其实已有一个 stub（`enabled: false` + `.skip_unit_test`，而且它 README 里的安装命令正是会
炸环境的那条），已整体替换并删掉 `.skip_unit_test`，现在 runner 能发现并通过（tags `tool,slow`）。
#4~#10（ResNet50 / NanoGPT / SD / SAM2 / ModelOpt 量化）没做，理由写进了 README：都要 gated 或
数 GB 的 HF 下载 + 在 venv 里再装 torch/transformers，而 API 还没到 1.0。

### 两个 md 的处置

+ `candidates-eco-tripy.md`：自包含的几条已全部落地，**删除**。
+ `candidates-eco-torch-tensorrt.md`：**删除**，但里边还开着 5 条（#5 Refit 标着 HIGH 且确实没做、
  #7 FP8 ViT、#8 VGG16 PTQ、#14 weight streaming、#15 多卡），已整表迁进 `99-Todo/README.md`
  的「Still open from the ecosystem repos」一节，没有丢。顺带纠正：原 md 里 #5 底下那段
  "**As implemented**: 4 cases..." 讲的其实是 CUDA graph，是 **#6 的笔记错位**贴到了 #5 下面。
+ **#10 multi-profile 做不了**（本轮唯一没完成的委托）：装的 `torch_tensorrt` 2.14.0a0 完全没有这套
  API —— `Input(profiles=...)` 抛 `ValueError`、`runtime.optimization_profile` 不存在、
  `torch.classes.tensorrt.Engine` 上没有任何 profile 方法、`_TRTInterpreter.py` 里写死单个
  `create_optimization_profile()`。源码 clone（同为 2.14.0a0 但更新的 commit）有这功能，但
  `set_active_profile` 在 `core/runtime/TRTEngine.cpp` 里，光覆盖 py 文件没用，得整套源码重建。
  已记进 README 待办。TRT API 层的等价物 `08-Advance/MultiOptimizationProfile` 本来就有。

## `99-Todo/` 合并 + ChatGLM-6B 流水线取经（2026-08-28）

### 合并

8 个 md（README + 5 个 candidates + trex 2 个）**合成一个 `README.md`**，1288 行压到约 250 行。
去重方式：原 README 里的「Top candidates」两张表其实是 `candidates-github.md` /
`candidates-gitlab.md` 摘要表的副本，只留一份并把详情合进行内；已完成项的叙述（trex 全部 16 项、
链接体检表、已落地的生态例子）压成一句话。结构改成便于「晒需」：

+ **§1 = 挑选清单**，每条一行、带稳定编号（S1-6 常驻 / G1-12 OSS / L1-11 GitLab / M1-12 ModelOpt /
  R1-3 RTX + T1-4 tensorrtx / P1-6 Torch-TRT+Tripy 遗留 / 社区仓库），划掉不要的即可。
+ §2 = **已决定不做**（DLA、demoDiffusion、TRT_* 算子、sampleDevice RAII、TRT-LLM 关档、两个假缺口），
  防止重新提案。
+ §3 = GitLab **EXCLUDE 名单**（客户名目录、泄露内网 IP/客户模型路径的工具、Myelin/fusion 内部测试等），
  这份必须原样保留。
+ §4 = 参考（生态仓库、11.0→11.2 头文件 530 vs 530 空 diff、TREx 迁移始末）。

### ChatGLM-6B（`/work/chatglm-6b`，TRT 8.6 时代的手写 LLM 流水线）

读完写进 `99-Todo/chatglm-6b.md`，并在 README 里立了 **S6**。这套东西的价值在于**它把 TRT-LLM
后来产品化的那些招数摊开在 850 行里**，而且多数招数跟 LLM 无关：

值得搬的（C1-C12，前四条最硬）：

1. **把整个 KV cache 打成一个 I/O 张量**——28 层 ×(K,V) = 56 进 56 出，用一个 `Split(num_outputs=56)`
   和一个 `Concat` 变成 1 进 1 出，115 个 I/O 张量降到 6 个。
2. **KV 的输入和输出绑同一个地址**，cache 原地增长；`backupFile/main-twoBuffer.py` 保留了更早的
   乒乓版本，正好做 before/after。
3. **把采样搬进 engine**：surgery 阶段接上 Slice(最后一个位置)→MatMul(lm_head 常量)→Softmax→ArgMax，
   engine 直接返回 token id。词表 130528，等于每个 token 少搬 26 万个数。
4. **靠改模型自己的 `forward` 来导出**：`exportONNX.py` 只有 14 行且不含 export 调用，真正的
   `torch.onnx.export` 被塞进 patch 过的 `modeling_chatglm.py`，并且**卡在第二次调用（`past_key_values`
   非空）才触发**——导出的是带真实 cache 的 decode 图，不是拼出来的假图。
5. `markGraphOutput`：把任意节点的输出/输入标成图输出并截断图，用来二分精度 bug。
6. 旋转位置编码的 cos/sin 表在 surgery 时算成常量；改 `Transpose.perm` 原地重排 attention 布局；
   把第二个 ONNX（lm_head）的权重提出来当 Constant 并进主图；
   `gs.Constant` 必须 `np.ascontiguousarray`，否则 **TRT 把形状读成 `(0)`**（作者原注三个感叹号）。

不要搬的：**它的 build 配方在 TRT 11 下基本是非法的**——`EXPLICIT_BATCH` flag、`BuilderFlag.FP16`
+ `OBEY_PRECISION_CONSTRAINTS`、逐层 `set_output_type`、`FASTER_DYNAMIC_SHAPES_0805` 全部已移除，
build 那一段是重写不是移植（反过来说，它是 strong-typing 迁移的好素材）；
`builder_optimization_level=5` 绕开 myelin 是时代特定的 workaround，别照抄结论；
ChatGLM 特有的 token id / 两行 position_ids / 中文标点正则 / `.i(2).i().i().i().i()` 链也别搬。

建议形态是**两块而不是一个大例子**：`07-Tool/OnnxGraphSurgeon` 补工具箱（C5/C6/C8/C9），
外加一个新的 `03-Workflow` 叶子——用**玩具级 2 层 decoder** 把 C1/C2/C3/C4/C10 串起来。
**cookbook 目前完全没有自回归工作流**，这是真缺口。原 6B 模型不搬，只作为全尺寸参照引用。

## `03-Workflow/pyTorch-KVCache-ONNX-TensorRT`（2026-08-28）

把 ChatGLM-6B 那套东西用 **gpt2-small** 重写成 cookbook 例子。选 small 不选 medium 的依据是实测对比
（词表都是 50257，架构、cache 布局、导出路径全同；差别只是 51→5 与 99→5 的标题数字，以及下载/体积
2.5 倍），已记在 `99-Todo/chatglm-6b.md`。

五个 case，全部实跑：

1. **I/O 爆炸**：直接读 `00-Data` 里现成的 `model-large.onnx`（就是 gpt2-medium 的 prefill 图），
   2 进 **49 出**；对应的 decode 图是 **99 个 I/O 张量**——每个 token 都要 set 一遍。零下载。
2. **打包导出**：wrapper `Module` 把整个 cache 收成一个张量，导出后 **5 个 I/O**（2538 节点）。
   在 `L_past=3` 上 trace，在 1/7/16 上与 eager PyTorch **token 完全一致**——`transformers` 刷的那堆
   `TracerWarning` 看着像把图冻死在 trace 长度上，实测没有（但 `is_causal` 确实被烤进去了，所以这是
   decode 图，prefill 要单独导）。engine 476 MiB，**build 11.6 s**。
3. **原地 cache**：`past_kv` 和 `present_kv` 绑**同一块 18 MiB 显存**，跑完 17 步，
   输出 "TensorRT is a new type of neural network that can be used to train a neural"。
4. **布局决定能不能 alias**（这条是新发现，比原版更有价值）：
   `[2N,B,H,L,D]`（transformers 的存法）前 92160 个值 **≠** 输入；`[L,2N,B,H,D]`（本例导出的）**=**。
   序列轴不在最外层时新数据是插进去的，共享 buffer **静默算错**。ChatGLM 的 cache 恰好是 `[L,B,32,128]`，
   所以它那个"原地增长"不是巧思，是布局的必然结果。
5. **采样进图**：`next_token (1,1) int32` 0.782 ms/token vs `logit (1,50257) fp32` 0.866 ms/token
   ——每 token 少搬 **196 KiB**，整步 **1.11x**（复跑 1.09x）。两个 engine 由同一个 wrapper 的
   `b_return_logit` 开关产出，是同类对比。

顺带修的两处：

+ `requirements.txt` 里的 **`tripy` 删掉**——包名就是错的（真名 `nvtripy`），而且真装进来会把
  TensorRT 11 换成 10.16。已在文件里写明原因防止再加回来。新增 `transformers`（`--dry-run` 查过，
  只多拉 typer/shellingham/annotated-doc，不碰 numpy 和 tensorrt）。
+ `.pre-commit-config.yaml` 的 codespell `--skip=".git,3rdparty,*.ipynb,*.txt"` **引号是字面量的一部分**，
  导致 `*.txt` 这条 skip 从来没生效过（`requirements.txt` 里的 `lief` 被判成拼写错误才暴露出来）。
  去掉引号。

runner 通过（clean 状态 61 s，HF 权重已缓存的前提下）。生成物 ~2.2 GB（两个 ONNX + 两个 engine）
全部 gitignore 且 clean 清掉。

## `07-Tool/OnnxGraphSurgeon` 补 ChatGLM 工具箱（S6a，2026-08-28）

先纠正我上一轮的判断：**S6a 比我说的小**——`mark_graph_output`（C5 的那个 helper）和 `add_node`
（C-addNode）**早就已经移植进 `tensorrt_cookbook/utils_onnx.py` 并导出了**，`np.ascontiguousarray`
的坑也已经写在 `10-advanceAPI.py` 的注释里。真正缺的是：`mark_graph_output` **在整个 cookbook 里
零使用**，没人演示那个「二分精度 bug」的工作流；以及 C6 常量表、C8 合并两个 ONNX。

新增三个文件（`01`~`10` 是 API 巡礼，`11`~`13` 是真模型上实际会用到的三件事）：

+ **`11-mark_output_to_bisect.py`**：把「FP16 结果不对」变成一个节点名。构造
  `((x*300)*300)/90000 == x`，FP16 下 90000 和中间值都溢出成 `inf`。**关键观察：最终输出既不是 inf
  也不是 nan，而是静默变成 0**（ORT 的 fp16 Div 行为），所以没有任何告警。逐节点截图后
  `node_scale_up_1` 第一个不一致（FP32 90000 vs FP16 inf）——错的是它，不是产生错误最终值的那个节点。
+ **`12-constant_table.py`**：常量折叠管不到「依赖运行时张量、但取值范围有界」的子图。旋转位置编码是
  典型：`cos(position*inv_freq)` 在主机上按所有可达 position 算一遍就变成一个 `Gather`，4 节点 → 1，
  误差 5.96e-08。**代价写清楚了**：表的大小从此是硬上限，原图没有这个限制。
+ **`13-merge_two_models.py`**：把单独导出的 head 接到 body 上，两种做法都演示（把权重当
  `gs.Constant` 提出来 / 整批搬节点），再接一个 `ArgMax` 尾巴，让图直接返回类别号而不是分数向量。

顺带修了个**移植 bug**：`mark_graph_output` 的 `b_mark_input=True` 分支对每个输入无条件写
`.dtype`，碰到 `gs.Constant` 会抛 `property 'dtype' of 'Constant' object has no setter`。
ChatGLM 原版在输入分支**没有**这行，是移植时加进去的。已加 `isinstance(..., gs.Variable)` 判断。
（这条路径此前全 cookbook 无人调用，所以一直没暴露。）

runner 通过（13.0 s）。

## 进度流水

- 2026-08-27：完成 Polygraphy 调研，产出 A/B/C 三类共 19 个候选；建 `More/` 与本文件。
- 2026-08-27：**A1 完成**。
- 2026-08-27：**A2 完成**（合并了原 A3）。
- 2026-08-27：**A4 完成**。
- 2026-08-27：**A5 完成**。
- 2026-08-27：**A7 完成**。
- 2026-08-27：**A8 完成**。
- 2026-08-27：**A9 完成**。
- 2026-08-27：**A10 完成，A 组全部结束**。
- 2026-08-27：**A6 完成**（发现 INT8 校准 API 在 TRT 11 已整体移除，改写为迁移指南）。
- 2026-08-27：**B1 完成**（发现 initializer=0 但权重存在 attribute 里，导出比源模型还大；控制流网络 Polygraphy 导不出、cookbook 的能）。
- 2026-08-27：**B2 完成**（tactic replay API 在 TRT 11 整体移除；实测 `save_timing_cache=` 只写不读、加速 1.00x）。
- 2026-08-27：**B3 完成**（弱类型网络已消失、`strongly_typed=False` 被静默忽略；四个 loader 三种结局）。
- 2026-08-27：**B4 完成**（`set_tensor_debug_state(True)` 是空操作；标记本身就有 1.56x 代价）。
- 2026-08-27：**B5 完成**（零拷贝在 25 MB 上 32.1x、MNIST 上只有 1.46x；view 悬空后 free 路径静默读错数据、resize 路径 SIGSEGV；`stream=` 在 pageable 内存上不异步，掩盖了漏掉的 synchronize）。
- 2026-08-27：容器重启导致 `onnxruntime`/`onnx_graphsurgeon`/`onnxslim`/`tensorrt_cookbook` 全丢，已重装（见「环境」一节）。
- 2026-08-27：**B6 完成**（`LoadPlugins`/CLI `--plugins` 走 ctypes，注册不了 V3 插件 creator，必须用 `registry.load_library()`；`register` 未导出；registry 只有 3 个 op、跑整张图）。
- 2026-08-27：**B7 完成**（VC 把 105 MB 的 lean runtime 塞进每个 plan，一层网络也是 9221x；代价在反序列化 34.4x、推理为 0；裸 runtime 反序列化 VC engine 返回 None）。
- 2026-08-27：**C1 完成**（订阅 arg group 白拿 95 个 CLI 选项；依赖只写在 docstring 里，漏订阅是运行期裸 KeyError；不能给 polygraphy 加子命令）。
- 2026-08-27：**C2 完成，A/B/C 全部结束**（entry point 必须真安装，PYTHONPATH 不够；import 即注册补上了 B6 缺的那条路；float64 参考揭出两个后端一致但都错 100%）。
- 2026-08-27：**CLI 侧复查完成**（10 个目录逐个实跑；修掉 Inspect 的假 FP16 步骤、Convert 的孤儿 data_loader、Plugin-TODO 的空结果与 config.yaml 污染、误提交的 p3~p7.py；补上 weight-strip/reconstruct、--save-visual、debug build/repeat、data concat、--data-loader-script、--convert-to onnx；multi-device 只出方案，单卡环境无法验证）。
- 2026-08-27：**`MultiDevice/` 建好**（CP 切序列插 6 个 collective、TP 切权重每 rank 一个文件；模型里没有 attention/SwiGLU 时整条流程静默变成复制一份；rank 数来自 `--nb-rank` 不是 `--gpus`。ONNX 改写部分单卡跑通并进 CI，多卡实跑留 case 06 待办）。
- 2026-08-27：**`07-Tool/trtexec` 复查完成**（发现步骤 11 因 `--refPair` 只给一对而整段中断、步骤 12 的 tuner knob 名根本不存在，两处都改对并跑通；补 `--stronglyTyped`（TRT 11 上是空操作）、多流吞吐、权重剥离 84x 且 CLI refit 静默失效；trtexec 的 `--fp16` 等选项是直接 Unknown option，与 polygraphy 的「help 里还在、build 才炸」形成对照）。
- 2026-08-27：**`07-Tool/TritonServerDeploy` 建好**（空壳 → 四阶段部署范例；本容器无 tritonserver 亦无 docker，阶段 3 跳过，但用 stub server 把客户端半边测了 16 项；记下 config.pbtxt 的 batch 维/标量 reshape/profile 三个坑）。
- 2026-08-27：**tritonserver 免 docker 跑通**（从 nvcr.io 匿名扒镜像层，只取 53 MB + 两个 .deb 里的 so；26.07-py3 的 TRT 恰为 11.1.0.106 与本机一致；TritonServerDeploy 四阶段全部实跑，输出与本地参考逐位相同）。
- 2026-08-27：**`08-Advance/EmptyTensor` 建好**（检测器无框 / 空 batch 的真实场景；发现并修掉 `utils_class.py` 四处 `cudaMalloc(0)` 返回 NULL 导致 `enqueueV3` 静默拒跑的共享代码 bug —— 这个 bug 一开始让例子自己得出了「MAX over nothing = 0」的错误结论，实为 `-inf`/`NaN`）。
- 2026-08-27：**`08-Advance/MIG-TODO` → `MIG`，决定只留 README 不写例子**（MIG 是宿主机配置、TRT 侧无差异；唯一值得记的是「engine 必须在部署用的 profile 上 build」，本机 114 SM/79 GiB vs 14 SM/9.75 GiB；对应测量因本机未开 MIG 列为待办）。
- 2026-08-27：**`08-Advance/GreenContext` 建好**（MIG 的进程内替代：切 SM 分区、免 root 免重启；吵闹邻居 p95 8.23x → 3.05x；**发现 TRT 的 aux stream 会逃出分区且默认就会发生**；**在分区内 build 白捡 19%**，用对照组推翻了我最初「无差异」的错误结论）。
- 2026-08-27：**`08-Advance/TensorRTGraphSurgeon` 建好**（INetwork 层面动刀：走图/加层/靠改线删层/换 plugin；换 plugin 输出逐位相同但**丢了融合、代价 2.13x**；发现「插件库已加载 + 释放过 engine → 查注册表必崩」的触发条件）。
- 2026-08-27：**FP16 ONNX 用量统计完成**（00-Data 里没有任何 FP16 ONNX；5 个 03-Workflow 例子用着已删除的 `BuilderFlag.FP16`、trex 的 model.fp16 其实是 FP32；建议默认用 ModelOpt AutoCast 生成，另备一个 torch.half() 的纯 FP16 版）。
- 2026-08-27：**`90-Misc/Number` 数据类型表复查完成**（对照 torch/ml_dtypes 逐 bit 实测：E8M0 的 `0x80` 是 2.0 不是 1.0、`0xFF` 是 NaN 不是最大值；FP4E2M1 的「最大的小于 1 的数」印了个它根本没有的 0.75；`Integer.md` 不是生成的且 UINT8/INT8 填的是 16 位范围；torch 目前没有比表里更新的浮点类型，补上了 FP6/MXFP6，删掉了并不存在的 NVFP8 与 FP7E4M3）。
- 2026-08-28：**TensorRT-LLM 候选清单关档**（按「只要跟 TensorRT 相关的工具」的标准，16 条候选全废；实测该仓非 3rdparty 里只剩 9 个文件 `import tensorrt`，建 engine 那层已整体删除，三个插件例子全部 import 不起来；纯 Python 写 TRT 插件的例子改从 TensorRT OSS 拿）。
- 2026-08-28：**`05-Plugin/TritonAOTPlugin` 建好**（从零重写 TRT-LLM 那两个已失效的 Triton 插件例子：AOT 内嵌 cubin 的 C++ 插件 + 从 spec 生成插件；**实测出 Triton AOT 把 `fp32` 标量参数声明成 `double` 导致静默算成 `x+0` 且 launch 仍返回成功**；多个 AOT variant 映射成 TRT tactic，builder 选出的比固定第一个快 1.83x；另记录「插件库不导出 `setLoggerFinder` 就静默注册失败」）。
- 2026-08-28：**`07-Tool/nvtriPy` 建好，Tripy / Torch-TensorRT 两个 candidates md 关档**（先踩了个大坑：`pip install nvtripy` 把 cookbook 环境的 TensorRT 11 换成了 10.16、numpy 降到 1.26，已回滚验证；例子改成在私有 venv 里跑，并把这件事写成主线。发现 **Tripy 的 eager 与 compiled 在 matmul 上不一致且偏的是 eager**；Torch-TRT 的 multi-profile 因装的版本根本没有该 API 而未完成，连同另外 4 条未做项迁进 `99-Todo/README.md`）。
- 2026-08-28：**`99-Todo/` 8 个 md 合并成一个 README（1288 → ~250 行）**，按 §1 挑选清单 / §2 不做 / §3 EXCLUDE / §4 参考 重组，每条候选给了稳定编号便于晒需；**读完 `/work/chatglm-6b` 写出 `99-Todo/chatglm-6b.md`**（TRT 8.6 时代手写 LLM 流水线：KV cache 打包成单张量、输入输出同地址原地增长、采样搬进 engine、靠 patch `forward` 在第二次调用时导出 decode 图；其 build 配方在 TRT 11 下已非法）。
- 2026-08-28：**`03-Workflow/pyTorch-KVCache-ONNX-TensorRT` 建好**（用 gpt2-small 重写 ChatGLM-6B 的那套招数：KV cache 打包成单张量 99→5、输入输出绑同一块显存原地增长、采样进图省 196 KiB/token、导出图与 PyTorch token 逐个一致；**新发现：能不能 alias 完全由 cache 布局决定**，transformers 的 `[2N,B,H,L,D]` 共享 buffer 会静默算错。顺带删掉 requirements 里危险且名字就错的 `tripy`，修好 codespell 那条因引号而从未生效的 `--skip`）。
- 2026-08-28：**S6a 完成**（`07-Tool/OnnxGraphSurgeon` 新增 `11`~`13`：二分精度 bug / 主机端常量表 / 合并两个 ONNX + ArgMax 尾巴。先纠正判断：`mark_graph_output` 与 `add_node` 早已移植进 utils，只是**零使用**，缺的是演示；顺带修好 `mark_graph_output` 的 `b_mark_input` 对 `gs.Constant` 写 dtype 崩溃的移植 bug）。

## `tensorrt_cookbook` 工具函数归位（99-Todo 第 1 条，2026-08-31）

### 环境（本轮开工时又坏了一次）

容器重启后 `onnxruntime` / `onnx_graphsurgeon` / `onnxslim` **以及 `tensorrt_cookbook` 的
editable 安装** 全部丢失（比 2026-08-27 那次多丢了后者，症状是所有 main.py 报
`ModuleNotFoundError: No module named 'tensorrt_cookbook'`）。恢复：

```bash
pip install onnxruntime-gpu onnx_graphsurgeon onnxslim
pip install -e .          # 这一条 2026-08-27 的记录里漏了
```

TensorRT 11.1.0.106 / numpy 2.1.0 / Python 3.12.3，与之前一致。

### 动机

`utils_function.py` 1183 行里装了四类完全无关的东西，最后一节「Tool functions for TensorRT」
其实是 engine / layer / onnx-parser 三个主题混在一起；而 `utils_network.py` 里又躺着三个
只吃 engine-information JSON 的函数。判断依据不是主观分类，是**已有的 import 边**：
`utils_network.py` 从 `utils_function` 里捞 `parse_onnx` 和 `layer_*`，
`utils_network_serialization.py` 也从 `utils_function` 捞 `layer_*` —— 被谁 import
就说明它属于谁。

### 落地的布局

| 模块 | 职责 | 本轮变化 |
| --- | --- | --- |
| `utils_cookbook.py` | cookbook 自身设施：路径、日志、API 覆盖率、README/版权生成 | **+`case_mark`、+`text_to_logger_level`** |
| `utils_function.py` | **与框架无关**的数据helper：数学、数组打印/比对、dtype 转换 | 1183 → 397 行 |
| `utils_onnx.py` | 纯 ONNX / onnx-graphsurgeon | 不变（**保持不 import tensorrt** 这条不变量） |
| `utils_network.py` | 搭建与检视 `INetworkDefinition` | **+5 个 `layer_*`、+`parse_onnx`**；移出 3 个 engine-JSON 函数 |
| `utils_engine.py` | **新建**：plan 文件头 / engine / context / engine-JSON 的检视 | `Pointer`、`print_engine_information`、`read_host_array_from_pointer`、`print_engine_io_information`、`print_context_io_information`、`_numel`、`get_engine_tensor_info`、`is_tensor_used_later`、`export_engine_as_onnx` |
| `utils_workflow.py` | **新建**：Torch→ONNX→ORT→Polygraphy→TRT 的逐段支持度检查 | `check_torch_operator`、`get_profile_shapes_from_dynamic` |
| `utils_class.py` / `utils_plugin.py` / `utils_mpi.py` / `utils_network_serialization.py` / `utils_engine_explorer.py` | — | 只改 import 行 |

**对外零影响**：`__init__.py` 是 `from .utils_* import *`，而全仓库只有一处
（`07-Tool/OnnxVisualization/04-LoopBackend/main.py` 的 `onnx_outliner.preprocess`）
按子模块 import，其余 700+ 处全走 `from tensorrt_cookbook import X`。所以跨模块搬函数
不会破坏任何例子 —— 这也是敢动的前提，先查清楚才动的。

**顺带的收益**：`utils_function.py` 的 import 从 21 个降到 10 个，
`ctypes`/`subprocess`/`sys`/`tempfile`/`traceback`/`OrderedDict`/`Path`/`onnx`/`onnxruntime`/
`cudart`/`fold_constants` 全部随对应函数走了。现在它只依赖 numpy/torch/tensorrt。
`onnxruntime` 这个重依赖收敛到 `utils_workflow.py` 一处（`check_torch_operator` 是唯一用户，
外部也只有 `07-Tool/CheckTorchOperator` 一个例子在用）。

### 记录一处**没有**合并的重复

`utils_network.py::export_engine_as_onnx`（现已移入 `utils_engine.py`）与
`utils_engine_explorer.py::export_engine_to_onnx` 是同一个想法的两份实现（engine 信息 → 可用
Netron 打开的 ONNX），但**输入不同**：前者吃 engine-information JSON 文件路径、走 `gs`；
后者吃 TREx 的 `EnginePlan` 对象、走 `onnx.helper`。两边各有例子在用，强行合并要改调用方，
收益不抵风险，**故意留着**并在此记一笔，免得下次又被当成新发现。

### 验证

+ `pyflakes` 对 10 个 `utils_*.py` 干净（只剩一条与本次无关的 `utils_class.py:1045 n_byte` 旧告警）。
+ `pre-commit run --files tensorrt_cookbook/*.py` 全过（yapf 重排了一轮，已收进改动）。
+ 三个直接用到被搬函数的例子实跑 rc=0：`07-Tool/ContextPrinter`、`07-Tool/EnginePrinter`
  （`print_engine_information` / `print_engine_io_information` / `export_engine_as_onnx`）、
  `07-Tool/CheckTorchOperator`（五段全 SUPPORTED）。
+ `pytest tests/NetworkSerialization/`（`layer_*` 的重度用户）：**8 failed / 114 passed / 16 skipped**。
  **这 8 个是既有失败，不是本次引入的** —— 把改动 `git stash` 掉、把两个新模块移走之后跑同一条
  命令，结果一模一样（8 failed / 114 passed / 16 skipped）。而且 `pytest tests/NetworkSerialization/pluginv3`
  单独跑是 **8 passed**，即它只在整套一起跑时挂，属于 2026-08-27 记过的那类
  「插件库已加载 + 释放过 engine → 查注册表必崩」的交互问题，另案。

## `NetworkSerialization` 的 80/81 问题（99-Todo 第 2 条，2026-08-31）

### 先弄清 80/81 到底是什么

线索是 `utils_network_serialization.py` 里那个默认开着的开关 `self.use_patch_80`（5 处使用）。
实测（TensorRT **11.1.0.106**，probe 脚本见下文结论）：

**从未被赋值过的 `trt.Dims` 型层属性，TensorRT 把它留在内部哨兵 `nbDims == -1` 上。**
从 Python 读它的行为是**不对称**的：

| 表达式 | 结果 |
| --- | --- |
| `len(dims)` | **抛** `ValueError: __len__() should return >= 0` |
| `list()` / `bool()` / 迭代 / `in` | 同样抛（都建立在 `len()` 上） |
| `dims[0]` | 抛 `IndexError: Out of bounds` |
| `repr(dims)` / `str(dims)` | **不抛**，打印一个垃圾 rank：`(80)` 或 `(81)` |
| `dims.__len__()` | **不抛**，返回 `-1` |

抛异常的是 `len()` 内建函数（它拒绝负返回值），底下的 `__len__` 槽位老老实实返回 `-1`。
**这就是「只有 vscode debug 能读出来、正常脚本一读就报错」的全部原因** —— 调试器 watch 窗口调的是
`__repr__`。名字里的 80/81 就是那个垃圾 rank。

受影响属性（11.1.0.106 全部实测确认）：`IShuffleLayer.reshape_dims`、
`ISliceLayer.axes`/`.start`/`.shape`/`.stride`、`I(De)QuantizeLayer.block_shape`，
外加**此前没被 patch 覆盖的 `IResizeLayer.shape`**（scales 模式下）。
垃圾 rank **与输入 rank 无关、与读哪个属性无关**，本机一律是 `81`
（rank 1~6 的 6 组输入逐个试过）。

### 不打补丁会怎样（实测，不是推测）

`use_patch_80 = False`，序列化一个只做 transpose、从不设 `reshape_dims` 的 Shuffle：
**哨兵原样落进 JSON**，得到 `"reshape_dims": [81]`。序列化和反序列化**都报成功**，
重建的网络在 build 时才失败（被要求 reshape 成形状 `(81)`），
表现为 `engine_bytes is None` → 上层 `deserialize_cuda_engine(None)` 抛 TypeError。
又一个「看着对、实际没生效」的静默失败。

### 更优美的 work around（已落地）

新增 `tensorrt_cookbook.is_dims_unset(dims)`，放在 `utils_network.py`（层属性检视，与 `layer_*` 同区），
实现就一行：**`return dims.__len__() < 0`**。5 处调用点全部改用它。它替掉的是两套更脆的判据：

1. `try: len(x) / except ValueError:` + `re.fullmatch(r"\(\d+\)", repr(x))`。
   今天是对的，但它依赖 TensorRT 对一个**根本没承诺过要格式化**的值的 `repr` 拼法；
   而且它的 `else` 分支（「显式设成 `[]`」）是**死代码** —— 实测显式 `[]` 读回来是 `()`、
   `len() == 0`、根本不进 `except`。
2. `ast.literal_eval(str(layer.axes))` 然后 `axes_dump > 8`，即「rank 超过 `trt.Dims.MAX_DIMS`
   就当垃圾」。**这条有真实失效模式**：垃圾值今天恰好是 80/81，但没有任何东西保证它一直大于 8，
   一旦垃圾值落在 0~8，就会被当成真的 `axes` 静默接受。

`__len__() < 0` 只依赖 `-1` 这个哨兵，无需 try/except，也不会被换一个垃圾值骗到。
顺带删掉了 `ast` import（全文件仅此一处用）。

### 看门狗测试（这正是「检测当前版本 TRT 是否已修复」的那件事）

新增 `tests/NetworkSerialization/test_unset_dims_patch.py`，7 个用例全过：

+ `TestUnsetDimsSentinel` 把 bug 本身钉死：`len()` 必抛、`__len__()` 必须是 `-1`、
  `repr()` 必须打出一个 > `MAX_DIMS` 的垃圾 rank、而**已赋值**的 dims（含显式 `[]`）绝不能被误判。
  **TRT 哪天修好了，这几个用例就会开始失败 —— 那就是删掉 `use_patch_80` 和 `is_dims_unset` 的信号。**
+ `TestPatchIsLoadBearing` 证明补丁还在真干活：同一个 round trip，patch 开着 `reshape_dims`
  被归一成 `[]` 且重建成功；patch 关掉则 JSON 里出现垃圾 rank 且 `engine_bytes is None`。
+ 另加一个 Resize 用例，断言 scale 模式下 `"shape"` 不得把哨兵漏进 JSON。

### 顺手修掉的一处（此前没人发现）

`IResizeLayer.shape` 在 scales 模式下同样是哨兵。它**从不破坏 round trip**
（反序列化侧靠结构化标志 `is_static_scale_mode` 直接跳过 `shape`），
但序列化侧一直在往 JSON 里写 `"shape": [81]` —— 一个可被读者当成真数据的垃圾值。
现已一并归一成 `[]`。这条对应 README「Issues and suggestions」的第 9 条，
和第 8、10 条是同一个根因，已在 README 里标注。

### 验证

+ `pytest tests/NetworkSerialization/` → **8 failed / 120 passed / 16 skipped**。
  120 = 改动前的 114 + 本次新增的 6（Resize 那个用例后加，最终 7 个）。
  **8 个失败仍是既有的那 8 个 pluginv3**（本轮开头已用 `git stash` 对照确认），与本次无关。
+ `pytest test_shuffle/test_slice/test_qdqstructure/test_dynamicquantize/test_resize/test_fill/test_nonzero`
  → 25 passed / 8 skipped，即 5 个改动过的 patch 调用点全部覆盖到。
+ `pre-commit` 全过。

### 还没做的

README 里 `use_patch_80` 之外的 TODO（INT8-PTQ / timing cache / refit / skipped cases 等）和
「改进建议」几节没动，那些不属于第 2 条问的范围。

## `07-Tool/OnnxGraphSurgeon` 去 `main.sh` + 全面改下划线（99-Todo 第 6 条，2026-08-31）

三件事一起做完，runner 通过（13.6 s，13 个脚本全跑）：

1. **删掉 `main.sh`**，把有序步骤搬进 `unit_test.yaml` 的 `run:`（连同每步的注释一起搬，没有丢信息），
   `rm -rf *.onnx *.weight` 变成 `pre:`。这样「runner 看到的顺序」和「人读目录看到的顺序」是同一份，
   不再有两处需要同步。`run_tests.py` 是 `shell=True` 跑的，所以 `05_print_graph.py > result-05.log`
   这种重定向照常работа。
2. **13 个 py 脚本 + `graph-api.py` 全部 `-` 改 `_`**（`git mv`，git 全部识别为 rename 而非删+增）：
   `01-create_model.py` → `01_create_model.py` …… `graph-api.py` → `graph_api.py`。
3. **脚本内生成的 onnx 名字同步改**：`f"model-{stem}"` → `f"model_{stem}"`，
   后缀 `"-01.onnx"` → `"_01.onnx"`、`-body/-head/-merged` 同理。
   由于文件名本来就是 `Path(__file__).name` 推出来的，改完脚本名后模型名自动跟着变，
   实测产物已全部是 `model_10_advanceAPI_00.onnx` 这种形状，目录里再没有带 `-` 的产物。
   **注意没动的一处**：`05_print_graph.py` 里的 `cookbook_path("00-Data","model","model-trained.onnx")`
   ——那是 `00-Data` 下的共享模型，不属于本目录生成物，改了会找不到文件。

连带改的引用：本目录 README（`./main.sh` 那段改成 runner + 单跑两种用法，并说明「没有 main.sh，
顺序在 unit_test.yaml 里」）、README 里 `11`~`13` 的文件名、`99-Todo/chatglm-6b.md` 里
`08-isolate_subgraph.py` / `06-fold.py` 两处。`progress.md` 的历史记录保持原样（那是当时的事实）。

**没改 `API/gsAPI-TODO.py`**：`-TODO` 是 cookbook 用来标「这一项还没做完」的既有约定
（99-Todo 第 5 条要做的 `07-Tool/ONNX-TODO` 就是同一个约定），把它改成下划线会把标记吃掉。
如果希望连这个也统一，告诉我，改起来是一行。

## 99-Todo 顶部 8 条清单的进度（2026-08-31）

`99-Todo/README.md` 开头有 8 条待办，按顺序做。当前：

| # | 任务 | 状态 |
| --- | --- | --- |
| 1 | 整理 `tensorrt_cookbook` 中各工具函数的位置 | ✅ 完成（见上文） |
| 2 | `NetworkSerialization` 的 todo/issue，尤其 80/81 | ✅ 完成（见上文） |
| 3 | `07-Tool/NsightSystems` 更多例子（如用 sqlite 爬信息） | ⏳ 已调研，未动手 |
| 4 | `07-Tool/nvtriPy` 的版本兼容性问题 | ⬜ 未开始 |
| 5 | `07-Tool/ONNX-TODO` | ⬜ 未开始 |
| 6 | `OnnxGraphSurgeon` 去 main.sh + 改下划线 | ✅ 完成（见上文，**提前做的**：纯文件改名不占 GPU，当时全量 runner 正在跑） |
| 7 | `07-Tool/Onnxruntime` 更丰富的用法 | ⬜ 未开始 |
| 8 | 例子是否临时从 HF 等下载模型 → 缓存到本地 | ⬜ 未开始 |

### 第 3 条的调研笔记（还没写代码）

现状：目录里只有 `main.sh`（两次 `nsys profile` 包 trtexec：建 engine / 载 engine）+ 12 个
`nsys xxx --help` 抓取。**产出的 `.nsys-rep` 有 54 MB，但没有任何一行代码去读它** ——
只能靠 `nsys-ui` 人眼看，CI 里等于只验证了「nsys 没崩」。

装的是 **Nsight Systems 2026.3.1.117**。可做的方向：
+ `nsys stats --report <name>` 内置报告 40 个（`cuda_gpu_kern_sum`、`cuda_api_sum`、
  `nvtx_gpu_proj_sum`、`nvtx_kern_sum`、`cuda_kern_exec_sum` 等），CSV 输出可直接断言。
+ `nsys export --type sqlite` → 用 Python `sqlite3` 自己查，这才是委托里点名的「用 sqlite 爬取信息」。
  与 TensorRT 结合的重点是把 **NVTX 层名（需 `--profilingVerbosity=detailed`）与 CUDA kernel 关联**，
  算出「每个 TRT 层花了多少 GPU 时间」——这是 `04-Feature/ProfilingVerbosity` 和
  `07-Tool/trtexec` 都拿不到的角度。
+ 目录里的 `.nsys-rep` / `.trt` 是 gitignore 之外的大文件（54 MB × 2 + 13 MB），要确认清理规则。

### 全量 runner（2026-08-31，本轮重构后的回归）

跑的是 `tests/run_tests.py --include "02-API/**"`，但**该 glob 没有起到限制作用**，实际从 00 一路跑到底
（这本身是个待查的小问题：`--include` 似乎没生效）。截至记录时仍在 06-DLFrameworkTRT，已出 2 个失败，
**两个都与本轮重构无关，但都是真问题，待确认后处理**：

+ `03-Workflow/pyTorch-KVCache-ONNX-TensorRT`：**缓存的 engine 是别的架构上建的** ——
  `Error Code 6: The engine plan file is generated on an incompatible device, expecting compute 10.0
  got compute 9.0`。本机是 H100（compute 9.0）。例子会复用 476 MiB 的缓存 engine 而**不校验设备**，
  换机器就炸。修法应该是缓存命中前先比对 compute capability，或者干脆把 engine 从缓存里排除。
+ `05-Plugin/TritonAOTPlugin`：`main.py:222` 在 engine information 里找不到
  `PluginType == "GeluTriton"` 的层，`[...][0]` 抛 `IndexError`。
  疑与 `TacticValue` 只在 `profiling_verbosity = DETAILED` 下出现有关（本目录 README 自己记过这条），
  也可能是插件层被融合/改名。**需要单独复现确认是否为既有失败。**

## 全量 runner 暴露的 4 个失败，逐个处理（2026-08-31）

### 1. `07-Tool/OnnxGraphSurgeon` —— 假失败，已消

`cmd=chmod +x main.sh` 失败，因为 sweep 启动时读的是我改之前的 `unit_test.yaml` 快照，
跑到这个目录时 `main.sh` 已被删。单独 `--case 07-Tool/OnnxGraphSurgeon` 用新 yaml 通过（13.6 s）。
**教训：改 unit_test.yaml 的同时不要有全量 runner 在跑**，它是启动时一次性发现的。

### 2. `07-Tool/ListAPIs` —— 缺 `tensorrt_rtx`

`ModuleNotFoundError: No module named 'tensorrt_rtx'`。
**先 dry-run 查过再装**（这是 nvtripy 那次的教训）：
`pip install --dry-run tensorrt_rtx` → 只会装 4 个包
`tensorrt_rtx / tensorrt_rtx_cu13 / _cu13_bindings / _cu13_libs`，全是 **cu13**，
**不碰 `tensorrt`、不碰 `numpy`**，无降级风险。等 sweep 跑完再实装（装包会影响在跑的用例）。

### 3. `03-Workflow/pyTorch-KVCache-ONNX-TensorRT` —— 改成不缓存 engine

原来 `build_engine()` 开头 `if trt_file.exists(): return`，于是复用了一个**别的算力上建的** 476 MiB plan，
报 `Error Code 6: The engine plan file is generated on an incompatible device, expecting compute 10.0
got compute 9.0`（本机 H100 = 9.0）。

按委托改成**每次运行都重建 engine**：删掉早退，函数 docstring 里写清为什么不缓存
（plan 与算力绑定，省 12 s 换来的是换机器时一条难懂的报错）。
**ONNX 仍然缓存**：它与设备无关，而且重新导出才是慢的那一步（650 MB × 2）。
连带更新了 `.gitignore`、`unit_test.yaml` 注释、README 的「both are caches」那段
（原文说 ONNX 和 engine 都是 make 式缓存，现在只有 ONNX 是），
以及那行已成死代码的 `" (cached)"` 打印。删掉磁盘上两个过期 plan 后实跑 **rc=0**。

### 4. `05-Plugin/TritonAOTPlugin` —— TRT 把插件吞进 Myelin，tactic 不再可观测

`main.py:222` 的
`[layer["TacticValue"] for layer in ... if layer.get("PluginType") == "GeluTriton"][0]`
抛 `IndexError`。

**先确认不是我这轮重构引入的**：把 HEAD 的 `tensorrt_cookbook` 用 `git archive` 解到 /tmp，
`PYTHONPATH` 指过去跑（这样不用 `git stash`，不打扰正在跑的 sweep）——**原版包同样失败**，
既有问题。

**根因**（实测）：插件层被 **Myelin 吸收**，engine information 里回来的是一个叫
`GeluTritonLayer_myl0_1`、`LayerType: custom_layer` 的层，**根本没有 `PluginType` 键、
没有 `TacticValue` 键**，`TacticName` 是空串。`profiling_verbosity = DETAILED` 下如此，
**改成静态 shape 建也一样**（所以不是动态 profile 的副作用），
`get_layer_information` 逐层查、`ONELINE` 格式查，结论相同。
（progress.md 记的 2026-08-28 那次是能读到 `0x3` 的，同一个 TRT 版本 —— 中间环境变了什么没查出来，
但现在的行为是稳定可复现的。）

**改法**：不重写整个例子（另外 3 个 case 都好的），只重写这一个 case 的「读 tactic」那一步 ——
把证据从「读一个 JSON 字段」换成「量」：另建一个只放第一个 variant 的 engine 对比耗时，
builder 要是没真在计时，两个 engine 就该一样快。并把丢失的可观测性本身写进 README
（这正是 cookbook 该记的那类发现）。同时给加速比加了 `assert speedup > 1.2`，
免得将来 3 个 variant 塌成 1 个又变成静默通过。实跑 **rc=0**。

注意：本次量到 2.72x（README 里记的是 1.83x），但**这次是在全量 sweep 占着 GPU 时量的，数字不可信**,
等 GPU 空了要重量一遍再决定 README 里写哪个数。

## 全量 sweep 收尾 + 又发现的 5 个失败（2026-08-31）

**先更正一件事**：那次 `run_tests.py` **不是跑完的，是被我给的 `timeout 5400` 砍掉的** ——
日志里没有 `=== Summary ===`，只跑到 `07-Tool/Polygraphy/More/07`，共发现/尝试 169 个用例。
所以「全量通过」这句话现在还不能说，需要重跑一次不设 timeout 的。

9 个失败，**没有一个是本轮 `tensorrt_cookbook` 重构引入的**。前 4 个见上一节，新的 5 个：

| 用例 | 原因 | 处理 |
| --- | --- | --- |
| `07-Tool/nvtx` | `'DummyDomain' object has no attribute 'get_counter'`。没有 profiler 挂上时 nvtx 给的是 DummyDomain，而例子假设它有 `get_counter` | 未处理，既有问题 |
| `07-Tool/OnnxVisualization/standalone` | `main.py` 要一个位置参数 `input`，而 `unit_test.yaml` 里就是裸 `python3 main.py` → argparse 直接 rc=2 | 未处理，**是 unit_test.yaml 的配置错**，一直没被发现是因为它从来没通过过 |
| `07-Tool/trex/11-ProcessEnginePipeline` | **硬编码了一个不存在的绝对路径** `/work/trt-samples-for-hackathon-cn/...`（少了 `-wili`，是别人另一个 checkout） | ✅ 改成 `cookbook_path("00-Data","model","model-trained.onnx")` |
| `07-Tool/trex/get_data.py` | 同一个硬编码路径问题（`model_dir`） | ✅ 改成 `cookbook_path("00-Data","model")` |
| `07-Tool/trex/02-DrawEngineGraph`、`11-ProcessEnginePipeline` | `graphviz.backend.execute.ExecutableNotFound: failed to execute PosixPath('dot')` —— **pip 的 `graphviz` 只是绑定，系统的 graphviz 没装** | ⛔ 需要 root，`apt-get` 在本会话被拒 |
| `07-Tool/trex/14-EngineArchive` | `AssertionError: Failed to deserialize the engine plan`（`utils_engine_explorer.py:1508`） | 未处理，怀疑与 KVCache 同类（plan 与设备/版本不匹配），待查 |

改完路径后 `11-ProcessEnginePipeline` 已经能跑过 trtexec 那一步，现在**卡在同一个缺 `dot` 上** ——
说明路径修对了。装上 graphviz 之后这两个 trex 用例应该能过。

### 已完成的三条委托

+ **ListAPIs**：`pip install tensorrt_rtx` 装好了（先 dry-run 确认过只装 4 个 cu13 包）。
  实测 **tensorrt 仍 11.1.0.106、numpy 仍 2.1.0**，没被动过。`tensorrt_rtx 1.6.1.120`。
  例子 **rc=0**，现在会同时导出 `result-tensorrt-11.1.0.106.log` 和 `result-tensorrt_rtx-1.6.1.120.log`。
  **要把 `tensorrt_rtx` 加进 `requirements.txt` 吗？** 它是 ListAPIs 的硬依赖，但会给所有人多装 4 个包。
+ **KVCache 不缓存 engine**：已改，rc=0。
+ **TritonAOTPlugin**：已改，rc=0。**空闲 GPU 上重量了**：0.0067 / 0.0186 = **2.78x**，
  连跑两次数字完全一致。README 里原来写的 1.83x（0.0134/0.0245）已过期，已更新，
  并注明「比值会变，别当常数」——所以断言用的是 `> 1.2` 而不是等于某个数。

**决定（2026-08-31，用户）**：`tensorrt_rtx` **不加进 `requirements.txt`**，
它只是 `07-Tool/ListAPIs` 的可选依赖，需要时手动 `pip install tensorrt_rtx`（已验证不动 tensorrt/numpy）。
graphviz 由用户自行安装。

## `07-Tool/NsightSystems` 加 sqlite 分析（99-Todo 第 3 条，2026-08-31）

新增 `main.py`（7 个 case，runner 通过 75 s），接在 `main.sh` 之后跑；`unit_test.yaml` 的 clean 加了 `*.sqlite`。
原来这个目录**两个 54 MB 的 `.nsys-rep` 没有任何一行代码去读**，CI 里只证明了「nsys 没崩」。

### 先解决「太大跑不动」

`main.sh` 的报告各约 52 MB，因为 CPU 采样和上下文切换默认开着。
`main.py` 用 `--trace=cuda,nvtx --sample=none --cpuctxsw=none` profile，报告只有 **136 KiB**，
export 不到 1 秒。`nsys export` 要遍历报告里每一个 event，所以这一步是「例子能用」和「例子要跑两分钟」的分界。

### 坑 1：NVTX 文本存在两个地方

一行要么有 inline 的 `text`，要么有指向 `StringIds` 的 `textId`，**从不同时有**（实测 21 / 30 / 0）。
TensorRT 的层名走的是 interned 那条路，于是最自然的写法
`WHERE text LIKE '%myl%'` **返回 0 行且不报错**，正确写法是 `COALESCE(n.text, s.value)` → 13 个层。

### 坑 2（本条的主菜）：kernel 表几乎是空的，总时间也是错的

TensorRT 把网络作为 **CUDA graph** 执行，而 `nsys` 默认 `--cuda-graph-trace=graph`，
**只把每次 graph launch 记成一个不透明活动，不记录图内的节点**：

| | `graph`（默认） | `node` |
| --- | --- | --- |
| `CUPTI_ACTIVITY_KIND_KERNEL` | **11 行，0.032 ms** | **572 行，1.397 ms** |
| `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` | 51 行，1.358 ms | **表根本不存在** |
| `cudaGraphLaunch` | 51 | 51 |

即：直接 `SUM(end-start)` kernel 表 —— 最自然的做法 —— 在默认粒度下**把 GPU 时间少算 43x**，
而且 50 次迭代只看到 11 个 kernel（那 11 个是 capture 那一趟，之后每次 replay 都是隐形的）。**全程无任何告警。**

默认值不是在撒谎，是在回答另一个问题：graph 模式的 `GRAPH_TRACE` 合计 **1.358 ms**
对 node 模式的 kernel 合计 **1.397 ms**，差 2.8%。而且数目严丝合缝对得上：
**51 次 replay × 11 个 kernel + 11 个 captured = 572**（代码里下了断言）。

**注意这不是例子自己选的**：`trtexec` 跑的时候**没有** `--useCudaGraph`，
是 Myelin 自己把融合区域做成了 CUDA graph —— 也就是说，**任何人 profile 一个 TensorRT engine 都会默认撞上这个**。
解法是 `--cuda-graph-trace=node`，代价是运行时开销更高。

### 产出：每个 TensorRT 层的 GPU 时间

NVTX range 在 CPU 时间线、kernel 在 GPU 时间线，所以要三跳 join：
NVTX range(CPU) 包住 launch API(CPU) ——`correlationId`—— kernel(GPU)。

```
node_conv2d_1_myl0_4    8096 ns  25.2%
node_linear_myl0_7      5248 ns  16.3%
node_conv2d_myl0_2      3648 ns  11.4%
...                     total 32128
```

`node_*` 是 ONNX 节点名一路带下来的，`__myl_*` 是 Myelin 融合区域。
**逐层时间精确等于外层 `ExecutionContext::enqueueV3` 的总和**（32128 == 32128，已下断言）——
这是「join 写对了」而不是「看着差不多」的判据。

**caveat 和结果一样重要**：只有 capture 那一趟带 NVTX，所以这是**一次迭代**的逐层归属，不是稳态；
50 次 replay 完全没有 NVTX。稳态的逐 kernel 数据要用 `nsys stats --report cuda_gpu_kern_sum`
（最后一个 case 做了交叉验证：同样 1.396 ms，每个 kernel 52 次）。

### 什么时候才该写 SQL

`nsys stats --help-reports` 有约 40 个内置报告，常见问题都不用写 SQL。
SQL 的价值在于没有内置报告的场合 —— 比如上面那个 NVTX 层名 ↔ kernel 的 join。
一个小坑：`nsys stats` 会自己推导 `<report>.sqlite`，与 case 1 导出的文件同名，
且它认为过期时会**拒绝复用并直接报 usage 错误**，所以要加 `--force-export=true`。

## 自查上一轮遗留的 4 个失败（2026-08-31）

7 个相关用例一起跑：**7 passed / 0 failed（205 s）**。

### `07-Tool/trex/14-EngineArchive` —— 与 KVCache 同一类 bug，已修

`AssertionError: Failed to deserialize the engine plan`。根因：`07-Tool/trex/data/` 里的
`model.engine` 是 **7 月 21 日**用旧 TensorRT 建的，而 `get_data.py` 的跳过条件只看
**三个 JSON 在不在**，既不看 engine 在不在、也不看是哪个 TRT 版本建的 ——
于是过期 plan 一直被复用，直到 `14-EngineArchive` 去反序列化它才炸，且报错离病因很远。

修法：跳过条件加上 `engine_file.exists()`，并写一个版本戳
`data/trt-version.txt`；`_stamped_version() != trt.__version__` 就整体重建，
并打印「原来是 X 建的、现在跑的是 Y，重建」。重建后用例通过。
（注意这个 `get_data.py` 就是上一轮修硬编码路径的那个文件 —— 它之前**根本跑不起来**，
所以过期产物才一直没被刷新。两个 bug 叠在一起互相掩护。）

### `07-Tool/OnnxVisualization/standalone` —— 测试配置错，已修

这个目录**没有 `unit_test.yaml`**，于是 runner 用默认规则 `python3 main.py`；
但它的 `main.py` 是个 CLI，`input` 是必填位置参数 → argparse 直接 rc=2。
**它大概从来没通过过。**

补了 `unit_test.yaml`：`pre` 先建 model zoo，`run` 用
`serial_chain.onnx`（12 节点 → 4 节点 + 1 个 local function，覆盖率 100%，ORT 逐位一致）跑一遍真实 CLI，
再抓一份 `--help`。README 里写的用法本来就是这样，只是没人把它写进 runner。
注意算法本身**早就有测试**（`01-BasicUsage/test_standalone.py` 查它与
`tensorrt_cookbook/onnx_outliner` 的漂移，并在禁止 `import tensorrt_cookbook` 的解释器里跑），
这里补的是 CLI 入口的冒烟测试，不重复。

### `07-Tool/nvtx` —— 环境里的 nvtx 太老，已升级 + 加保护

`'DummyDomain' object has no attribute 'get_counter'`。装的是 **nvtx 0.2.15**，
而 `case_counter` 的 docstring 本来就写着「Needs nvtx >= 0.2.16」——
0.2.15 里 `get_counter` / `get_timestamp` 在 `Domain` 和 `DummyDomain` **两个类上都不存在**。

+ `pip install -U nvtx` → 0.2.16（先 dry-run 确认，装完 tensorrt 仍 11.1.0.106、numpy 仍 2.1.0）。
+ `requirements.txt` 把 `nvtx` 改成 **`nvtx>=0.2.16`** 并注明原因 ——
  这是代码里已经写明的硬依赖，和 `tensorrt_rtx`（可选、只给 ListAPIs 用）不是一回事。
+ 代码里加了版本保护：缺 `get_counter` 就打印「nvtx 太老，`pip install -U nvtx`」后跳过。
  **理由**：裸 `AttributeError` 提到 `DummyDomain`，看起来像上一个 case 说的「没挂 profiler 所以是 no-op」，
  其实是版本问题 —— 两个原因指向同一个类名，最容易误诊。

### `07-Tool/trex/02-DrawEngineGraph` —— 仍待用户装 graphviz

`ExecutableNotFound: dot`。pip 的 `graphviz` 只是绑定，系统包没装，`apt-get` 本会话无权限。
用户已表示自行安装。装完这个和 `11-ProcessEnginePipeline` 应该都能过
（后者的硬编码路径已修，现在卡的就是同一个 `dot`）。

## `07-Tool/nvtriPy` 版本兼容性（99-Todo 第 4 条，2026-09-01）

**先甄别上次会话的半成品**：`main.py` 的 mtime（18:40）晚于 `log-main.py.log`（17:52），
日志里**没有 `[version_matrix]` 这一节** —— 说明上次加了 `report_version_matrix()` 但从没跑过、
README 也没写。这轮把它跑通、补完、写进文档。

### 结论：这个 venv 不是「以后可以去掉的临时绕路」

原来 README 只说了「装进来会把 TensorRT 11 换成 10.16、NumPy 2.1 换成 1.26」，
读起来像是 resolver 挑错了包、pin 死就行。不是。读 nvtripy **自己的** metadata：

```txt
tensorrt-cu12<11,>=10.15
mlir-tensorrt-{compiler,runtime}==0.1.43+cuda12.trt109
```

+ `<11` 是**上游主动声明拒绝 TensorRT 11**，不存在能让它和 cookbook 解释器共存的 pin。
+ `cu12` 是第二重：包索引只发 `cuda12` 构建、最高到 `trt109`，在 CUDA 13 容器里
  **即使放宽 TensorRT 约束也没有可装的 artifact**。

CUDA 12 的栈在 CUDA 13 容器里能跑，是因为 wheel 自带 CUDA runtime（`nvidia-cuda-runtime-cu12` 12.9.79），
驱动向后兼容，venv 全程不碰容器的 CUDA 13 toolkit，只共享 GPU 驱动。

### 加的两道防锈

难点在于「一个过期的 pin」和「一个仍然正确的 pin」**看起来一模一样**（都能装上、都能跑过）。

+ `tensorrt-cu12<11` 这条要求被 **assert** 住：哪天上游去掉了，例子会直接失败并提示
  「重新确认这个 venv 是不是还有必要」，而不是继续安静地多维护一个 venv。
+ 解析出的栈与 `TESTED_STACK` 比对，报 drift。

**第三道是这轮补的**：原来 `NVTRIPY_VERSION` 的注释写着「`case_version_matrix` reports when a
newer one exists」，**但那个函数根本没做这件事** —— 注释在承诺一个不存在的检查。
补了 `_latest_available_version()`（`pip index versions` 打索引，best-effort、超时/无网返回 None
不影响已装好的 venv）。实测 NVIDIA 索引和 PyPI **都是 0.1.7 最新**，assert 也仍然成立。

### 顺手

README 里 case 2 / case 4 的实测数字是旧机器状态的（first `.eval()` 759 ms、compile 176.5 ms /
54 KiB / 74x），与当前实跑（107.9 ms、423.5 ms / 89 KiB / 168x）差很远，已更新，
并注明比值逐次会变、看数量级即可。`rc=0`。

## `07-Tool/ONNX-TODO` → `07-Tool/ONNX`（99-Todo 第 5 条，2026-09-01）

原来这个目录是**纯占位符**：`api.py` 和 `my_workflow.py` 各 16 行、**只有 license 头没有代码**，
README 自己写着「notes only, no standalone runnable sample」，`unit_test.yaml` 是 `enabled: false`。
现在 `git mv ONNX-TODO ONNX`（去掉 `-TODO` 标记）、删掉两个空壳、补 `main.py` + README，
`unit_test.yaml` 打开。runner **1 passed（11.2 s）**，9 个 case 全过。

### 先摸清「已经覆盖了什么」，避免重复

派了一个 Explore agent 通盘查 raw-onnx 的覆盖情况。结论：**现有覆盖是顺带的，不是设计出来的** ——
`onnx.helper` 建模型（`OnnxVisualization/00-ModelZoo/model_zoo.py` 一个人就建了 28 个）、
`onnx.checker`、external data（`OnnxWeightProcess/`）都很充分，因为那些目录**需要**它们；
而这个库的**另一半——模型手术与互操作**：`shape_inference`、`version_converter`、
`utils.extract_model`、`compose`、`defs`、`parser`、`inliner`、`reference`、metadata 字段，
**全树 0 处实际调用**（只在 `90-Research/02-PackageSurvey/check_existing_tools.py` 的模块清单里
被列过名字，从没对模型跑过）。所以这个目录就写这一半，每个 case 都挂到「TensorRT 会怎样」上。

### 9 个 case，全部先实测再落笔

| case | 结论 |
| --- | --- |
| `checker` | **`check_model()` 放行了 TensorRT 拒绝的模型**（`[N,4] * [3]`，无法广播）。默认只查 proto 结构、不跑 shape inference；`full_check=True` 能提前抓到，且报的是算子名，而不是 `shapeContext.cpp:2700`。另附 `check_model(<path>)`——超过 2 GiB 的模型根本 `onnx.load` 不进 ModelProto |
| `shape_inference` | **`unk__0` 是从哪来的**：`nBS` 一路活过 conv/pool，到 `Reshape` 的 `-1` 就死了，之后 `linear`/`relu_2` 全是 `unk__0`。`data_prop=True` 也救不回来（实测过）。`softmax` 又变回 `nBS`，**只是因为导出器把输出 `y` 声明成了 `[nBS,10]`**——推理把声明的图 I/O 当事实，并不是它自己推回来的 |
| `defs` | op schema 注册表，回答「这个算子要哪个 opset」。`Reshape` 被改过 8 次（1,5,13,14,19,21,23,24,25），`Relu` 4 次 |
| `version_converter` | **升得上去，降不下来**。18→21/26 成功；18→13/11 是一句 C++ `assert`：`No Adapter From Version $14 for Relu`——**有用的信息在消息末尾**，开头是 `BaseConverter.h` 的路径。且 18/21/26 **在 TensorRT 里都是同样的 27 层**，所以「为了 TensorRT 转 opset」几乎从来不是解法 |
| `extract_model` | 报两个张量名就切出子图，**自动带上需要的 initializer、并给新输入填推理出的形状（连符号一起）**。给 TensorRT 报 bug 造最小复现最快的路子 |
| `compose` | `merge_models` **对歧义直接报错而不是猜**：名字撞了（任意两个独立导出的模型都叫 `a`/`b`）→ 让你用 `add_prefix`；opset 不一致 → 硬错。手工拼接会产出一个「能加载、能过默认 checker、但是错的」文件 |
| `parser` + `reference` | ONNX 自己的文本 IR（8 行写完一个模型）+ 纯 Python 的 `ReferenceEvaluator`。价值在于它是**规范的答案而不是某个实现的答案**——onnxruntime 和 TensorRT 吵架时的第三方意见，且不需要装 onnxruntime |
| `inliner` | local function **不需要为 TensorRT 展开**：带函数和展开后**都是 6 层**，说明 parser 自己就 inline 了。展开是给 Netron / diff / 老工具看的 |
| `metadata` | `producer_name` / `model_version` / `domain` / `doc_string` / `metadata_props` 都能往返，**对引擎零影响**（前后都是 27 层），也从不被校验。本区真正影响行为的只有 `ir_version` 和 `opset_import` |

### 两个写代码时踩到的坑

+ `full_check=True` 抛的是 `onnx.shape_inference.InferenceError`，**不是** `checker.ValidationError`
  的子类，只 catch 后者会漏掉一半。代码里两个都 catch 了并注明原因。
+ TensorRT parser 的诊断直接写 stderr 且不缓冲，而脚本 stdout 一旦重定向到日志就变成块缓冲 ——
  结果**所有 TRT 报错都跑到日志最顶上**，和触发它的 case 脱节。`trt_parse()` 开头加了
  `sys.stdout.flush()`，现在报错就在对应 case 里。

### 顺带发现，**没有动**：`07-Tool/README.md` 与子目录已经不同步

`build-Copyright-and-README.py` 是从每个子 README 的**前 3 行**生成父 README 的。重新生成一遍，
diff 里除了我这个目录（前 3 行没变，所以父 README **不需要改**）之外还冒出一堆无关改动：
`OnnxVisualization` **在已提交的父 README 里根本没有**（漏了一个目录），
而 `DebugUtils` / `FP16Tuning` / `OnnxWeightProcess` 的文字与子 README 对不上（说明父 README 被手工编辑过），
且 `nvtriPy` / `TritonServerDeploy` **重新生成后反而更差**——它们的子 README 前 3 行是半句话，
截出来就是断句。这是既有问题、且不属于本条任务，所以**把父 README 恢复了原样**，记在这里待办。

## 环境：这个容器缺包（2026-09-01）

新会话的容器**比上次会话少装了东西**，`import tensorrt_cookbook` 直接失败：
`onnx_graphsurgeon`、`onnxruntime`、`colored` 都没有，`tensorrt_cookbook` 本身也没 `pip install -e`。
按 CLAUDE.md 记的开发装法修复，每步先 `--dry-run` 确认：

+ `pip install onnx_graphsurgeon colored` —— 纯 Python，只新增这两个包。
+ `pip install --no-deps --no-build-isolation -e .` —— cookbook 自身。
+ `pip install onnxruntime-gpu` —— 只新增 `flatbuffers` + `onnxruntime-gpu 1.29.0`。

**装完 `tensorrt` 仍 11.1.0.106、`numpy` 仍 2.1.0**（已验证）。

## `07-Tool/Onnxruntime` 变厚（99-Todo 第 7 条，2026-09-01）

原来只有一个 `case_normal`：建 session、打印 I/O、跑一次。**而且里面有个真 bug**：

```python
for name, tensor in output_name_list, output_list:   # 迭代的是一个 2 元组，不是 zip
```

这行把两个**输出名**当成一对 name/value 打了出来（旧日志里那行 `y \n z` 就是它）。已改成 `zip`。
README 还写着「compare latency with CUDA EP」，但代码里既没有 TensorRT EP 也没有任何 latency 对比。
现在 7 个 case，runner **1 passed（13.9 s）**。

### 头号发现：这台机器上 **ORT 的 TensorRT EP 根本加载不了，而且是静默降级到 CPU**

```txt
TensorrtExecutionProvider    -> silently fell back: got ['CPUExecutionProvider']
CUDAExecutionProvider        -> run failed: cudaErrorNoKernelImageForDevice
CPUExecutionProvider         -> ok
```

+ **TensorRT EP**：`ldd libonnxruntime_providers_tensorrt.so` → `libnvinfer.so.10 => not found`。
  **ORT 的 TRT EP 被钉死在 TensorRT 大版本上**，容器里是 `libnvinfer.so.11`，soname 跨大版本不兼容。
  ORT 自己的报错却在怪 PATH / LD_LIBRARY_PATH / GPU 不受支持 —— **指错了方向**，
  再怎么改 PATH 也变不出一个不存在的 `.so.10`。**PyPI 上没有针对 TensorRT 11 编的 onnxruntime-gpu**
  （查到最新的 1.29.0 为止），TRT 11 机器上要用这个 EP 只能自己源码编 ORT。
  注意这只影响 **ORT 内嵌的** TensorRT，TensorRT 本身和 cookbook 其他例子都不受影响
  （本例里 `case_reference_for_tensorrt` 就在同一个进程里建了真引擎）。
+ **CUDA EP**：PyPI 的 wheel 是 CUDA 12 构建，没有本机算力的 kernel。

最要命的是**降级不抛异常**：`InferenceSession(..., providers=["TensorrtExecutionProvider"])`
只打一条 warning 就换成 CPU，结果**照样算对，只是慢**。
唯一的判据是 `session.get_providers()`。而且两种失败**形态完全不同**：库加载不了的在**建 session 时**被换掉，
库能加载但没 kernel 的在 **run 时**才抛 —— 所以必须「建 + 跑 + 再查 get_providers()」三步才能都覆盖。

### 第二个发现：默认的 TensorRT 引擎不是 FP32

同一个 ONNX 分别喂 ORT 和 TensorRT：

```txt
default (TF32 allowed)   max|ORT - TRT| = 6.146e-03  (3.39e-04 relative)
TF32 cleared             max|ORT - TRT| = 9.537e-06  (5.26e-07 relative)
```

**清掉一个 flag，差距变了 644 倍。** `BuilderFlag.TF32` **默认开着**，所以「不加任何精度 flag」建出来的引擎
做的不是 FP32 矩阵乘，是 tensor core 上 10 位尾数的 TF32。
所以在把 1e-3 的差异归咎于转换 bug 之前，先清 TF32 重测：塌下去就没 bug，剩下的 ~1e-5 是正常的重结合误差。

### 第三个发现：ORT 优化后的图**不能喂给 TensorRT**

`optimized_model_filepath` 能把 ORT 真正要跑的图写出来（12 节点 → 10 节点，三个 `Relu` 被折进生产者，
冒出 `FusedGemm` 和 `ReorderOutput`）。但它带上了 `com.microsoft` / `com.microsoft.nchwc` 域，
**TensorRT 解析直接失败**，报的是 `Plugin not found` —— 因为那是 ORT 私有的 layout-aware kernel。
导出来是给人**读**的，建 TensorRT 永远要用原始文件。

### 其余 4 个 case

profiling（ORT 自带 profiler → Chrome-trace JSON，逐算子耗时，**算子名是优化后的**，正好交叉印证上一条）、
dynamic shape（一个 session 吃 batch 1/4/8/37，不用 profile 不用重建，正是 TensorRT 用灵活性换 kernel 选择的对照）、
IOBinding + OrtValue（对应 TensorRT 的 `setTensorAddress`）、基础推理。

### 一个小而实的收拾

ORT 在设不了 CPU 亲和性的机器上会**每个线程打一行** `pthread_setaffinity_np failed`，输出前先刷 100+ 行红字。
它自己的报错里就写了解法：显式设线程数。所以所有 SessionOptions 都设了 `intra_op_num_threads`。

另外两个新例子都加了 `sys.stdout.reconfigure(line_buffering=True)`：ORT 和 TensorRT 的诊断直写 stderr 不缓冲，
而脚本 stdout 一重定向到日志就变块缓冲，不处理的话**日志开头是一大坨它们的消息**，和触发的 case 完全脱节。

## 例子不再临时下载模型（99-Todo 第 8 条，2026-09-01）

### 先做全树盘点，结论是「面比想象的小得多」

派 Explore agent 把 `from_pretrained` / `hf_hub_download` / `snapshot_download` / `load_dataset` /
`torch.hub` / `load_state_dict_from_url` / `wget` / `curl` / `urlretrieve` / `requests.get` /
`onnx.hub` / `git clone` / 脚本内 `pip install` 全查了一遍。大量看似命中的其实是假阳性
（本地 `.npz`、`add_constant(weights=...)`、`resnet18(weights=None)`）。

**真正在运行时联网、且 unit test 开着的，只有两个**：

| 例子 | 情况 |
| --- | --- |
| `03-Workflow/pyTorch-KVCache-ONNX-TensorRT` | **唯一一个离线硬失败的**。`GPT2LMHeadModel.from_pretrained("gpt2")` 和 `GPT2Tokenizer.from_pretrained("gpt2")` **每次都无条件调用**，没有 `local_files_only`、没有 `cache_dir=`、没有存在性判断。目录里缓存好的 `model-gpt2-step*.onnx` **救不了它** —— 生成那个 case 仍要调 `load_torch_model()` 去拿 `n_layer/n_head/head_dimension`，tokenizer 也要单独下 |
| `07-Tool/nvtriPy` | 首次跑 `pip install nvtripy` 建 venv，但**有 `.venv` 就复用，装不上就打 `Skipped` 退 0** —— 已经是优雅降级，不用改 |

其余全部离线干净：`00-Data` 的下载本来就是 README 里的手工步骤且 `enabled: false`，
`model-large.onnx`（1.6 GB）等产物早已落地在树里，Torch-TensorRT 那一族**故意用 `weights=None`**
（`EngineCaching/main.py` 里还写明了「就是为了避开下载」），Triton 那个扒容器层的脚本**不在 `run:` 里**。

### 改法：沿用 `00-Data` 既有约定，下载是独立的一次性步骤

+ 新增 **`00-Data/get-model-gpt2.py`**，把 `gpt2` 抓到 `00-Data/model/gpt2/`（**528 MB**）。
  只取 `config.json` / `model.safetensors` / tokenizer 那几个文件 —— 上游仓库还带着
  TensorFlow / Flax / Rust / ONNX 四份同样的权重，全下会**翻两番**。已有则直接退出。
+ `main.py`：`MODEL_ID = "gpt2"` → `MODEL_PATH = cookbook_path("00-Data","model","gpt2")`。
  `from_pretrained` **给路径就完全不碰网**，连 `local_files_only` 都不用加。
  加 `require_local_model()`：没有就把那条命令打出来、`Skipped` 退 0，不去够网络。
+ README（`00-Data` 和例子各一处）写明命令；`unit_test.yaml` 的注释同步改
  （原来写的是「首次运行下载 ~550 MB 然后缓存」）。
+ **`.gitignore`**：`00-Data/model/gpt2/` 整个忽略，并补了 `*.safetensors`。
  原来只有 `*.json` 被忽略，**528 MB 的 `model.safetensors` 和 `merges.txt` 是可以被 commit 进去的**。

### 一个下载时踩到的坑

`snapshot_download` 抓大文件时走 HuggingFace 的 **Xet** 协议，在这里直接
`404 .../xet-read-token/...`（小文件全都成功，只有权重失败）。`HF_HUB_DISABLE_XET=1` 回退到普通 HTTPS 就好。
已写进 `00-Data/README.md`。

## 环境续：容器缺的包比一开始发现的还多（2026-09-01）

除了前面修的 `onnx_graphsurgeon` / `onnxruntime` / `colored` / `tensorrt_cookbook`，
做第 8 条时又发现 **`transformers` 也没装**（所以 KVCache 那个例子在这容器里本来就跑不了，
和联网与否无关）。继续按 requirements.txt 补，每步 dry-run：
`transformers`、`pybind11-stubgen`、`torchinfo`、`onnxslim`、`opencv-python-headless`、`openpyxl`，
以及 **`nvtx` 又退回到了 0.2.15**（上次会话升到 0.2.16 的成果没了，再次确认这是**另一个容器**），
已重新升到 0.2.16 并实测 `Domain.get_counter` 存在。

`pip check` 显示**仍缺** `build` / `graphviz` / `lief` / `mpi4py` / `nccl4py`（还有 `nvidia-modelopt`、
`pyarrow` 等）。这些影响的是别的例子（trex 要 graphviz、MPIUtils 要 mpi4py、ModelOptimizer 要 modelopt），
不在本轮任务范围内，**没有装** —— 其中 mpi4py/nccl4py 要编译、modelopt 很重，建议由用户决定。
全程 `tensorrt` 保持 11.1.0.106、`numpy` 保持 2.1.0。

**另外：这台机器是 B200（sm100），不是之前进度里记的 H100（sm90）。** 这解释了
PyPI 的 `onnxruntime-gpu`（CUDA 12 构建）为什么报 `cudaErrorNoKernelImageForDevice`。
之前 KVCache 那个「expecting compute 10.0 got compute 9.0」的报错是反过来的方向，也对得上。

### 顺手修掉审计带出来的一个坏配置

`00-Data/unit_test.yaml` 的 `run:` 第一条是 **`python3 extract-mnist.py`，而这个文件不存在**
（实际叫 `extract-data-gz.py` / `-hf` / `-kg`），而且**漏了 `get-data.py` 这一步**。
因为 `enabled: false`，它从来没被执行过，所以这个错一直没暴露。
已改成真实的四步，并写清楚**为什么必须保持 disabled**：每一步都依赖人先手工下载的压缩包，
而它们的产物早就落在 `data/` 和 `model/` 里了。`get-model-gpt2.py` 故意**不**列进 `run:` ——
它是这里唯一真正联网的脚本，按 README 手工跑一次即可。

## 本轮收尾验证（2026-09-01）

| 用例 | 结果 |
| --- | --- |
| `07-Tool/ONNX`（新建） | ✅ 9 个 case |
| `07-Tool/Onnxruntime`（重写） | ✅ 7 个 case |
| `07-Tool/nvtriPy` | ✅ |
| `07-Tool/nvtx` | ✅（nvtx 升回 0.2.16 后，py + C++ 两条都过） |
| `03-Workflow/pyTorch-KVCache-ONNX-TensorRT` | ✅ 53.8 s；另外单独验证过 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 全程离线 rc=0，以及把 `00-Data/model/gpt2` 藏起来后打印下载命令并 `Skipped` 退 0 |

**99-Todo 顶部 8 条清单全部完成。** pre-commit 对所有改动文件全绿。

### 没有跑全量 sweep，原因

这个容器缺的包太多（见上一节），全量跑出来的失败会**大部分是环境问题而不是真问题**，
噪声盖过信号。建议先定夺 `graphviz` / `mpi4py` / `nccl4py` / `nvidia-modelopt` 装不装，再跑一次全量。

### 留给下一轮的两条（都已记在 `99-Todo`）

+ **S7**：`07-Tool/README.md` 与子目录不同步（`OnnxVisualization` 整个漏掉；直接重新生成会让
  `nvtriPy` / `TritonServerDeploy` 变差，因为它们子 README 的前 3 行是半句话）。
+ 清单以外的 §1.1~§1.7 候选（S1~S5、G*、L*、M*、R*、T*、P*）尚未开始。

## 用户澄清后的收尾（2026-09-01，第二轮）

用户指出四件事，逐条处理：

### 1. 装包正确姿势是 `cd cookbook && pip install -e .`

我前面用的是 `--no-deps`（顶层 README 里那条是 `pip install -r requirements.txt` **之后**再
`pip install -e . --no-deps`，两条路都对，只是我漏了前半步）。改用完整的 `pip install -e .` 后一次装齐：
`build`、`graphviz`、`lief`、`mpi4py`、`nccl4py`、`cupy-cuda12x`、`cppimport`、`onnxconverter-common` 等。
`pip check` 现在只剩一条与 cookbook 无关的（`triton-kernels` 要 pytest）。
**`tensorrt` 仍 11.1.0.106、`numpy` 仍 2.1.0、`nvtx` 仍 0.2.16**（`requirements.txt` 里的
`nvtx>=0.2.16` 约束起了作用，没被降回去）。

**但它把 `onnxruntime-gpu` 从 1.29.0 降到了 1.24.4。** 查了原因：
**`nvidia-modelopt[onnx]` 要求 `onnxruntime-gpu~=1.24.2`**。也就是说 1.24.4 不是可选项，
是执行 `pip install -e .` 的人必然得到的版本 —— 所以 `07-Tool/Onnxruntime` 的文档基准
已全部改成 1.24.4。

**降级后现象变了，例子的自检逻辑扛住了（依旧 pass），但结论更强了**：
1.29.0 时 CUDA EP 是**跑到 run 才抛** `cudaErrorNoKernelImageForDevice`；
1.24.4 时它**连库都加载不了**（`libcublasLt.so.12` 缺失），于是**两个 GPU EP 都静默降级到 CPU**。
而且缺的库变成 `libcublas.so.12` / `libcudart.so.12` **加上** `libnvinfer.so.10` ——
**是两重互相独立的大版本错配**（wheel 是 cu12 + TRT10，容器是 CUDA 13 + TRT 11），
任何一重单独存在都足以致命。这两点都写进了 README 和 `_explain_tensorrt_ep_mismatch()`。
「升级 ORT 就能解决」也被否掉了：版本被 modelopt 钉着，且更新的 wheel 仍是 cu12 + TRT10。

### 2. 测试平台确定为 B200

+ `07-Tool/nvtriPy/README.md` 的「Measured on H100 PCIe」改成 B200（这个例子本轮实测过）。
+ `03-Workflow/pyTorch-KVCache-ONNX-TensorRT/README.md` 的实测数字全部换成 B200 实测值：
  引擎构建 11.7 s → **16.4 s**；引擎内采样 0.782/0.865 ms → **0.597/0.684 ms**，1.11x → **1.15x**。
+ **其余带 H100 字样的 README 没有动**（TritonAOTPlugin、EngineCaching、NsightSystems、
  TritonServerDeploy、trtexec、OnnxVisualization×4、GreenContext、TensorRTGraphSurgeon、MIG）。
  理由：那些数字是**当时在 H100 上真测的**，改标签而不重测等于造假；它们需要的是**在 B200 上重新测量**，
  这件事应该并进后面那轮全面测试里做。**GreenContext 尤其要注意**：它的正文写死了
  「H100 PCIe 80 GB（114 SM）」和「`minSmPartitionSize` / `smCoscheduledAlignment` 都是 8」，
  B200 上 SM 数和这两个对齐值都会不同，是**必然要改**的一个。

### 3. nvtx 仍需手动装 0.2.16

已确认现状就是 0.2.16，且 `pip install -e .` 不会把它降回去（`requirements.txt` 有 `>=0.2.16` 约束，
是上一轮会话加的）。`07-Tool/nvtx` 实测通过（py + C++ 两条）。

### 4. README.md 用 `build-Copyright-and-README.py` 重新生成即可

照做了，但**光重新生成并不够** —— 生成器是取每个子 README 的**前 3 行**，
所以子 README 的第 3 行必须是一句**完整的 `+ ` 摘要**，否则截出来就是断句。
先把 12 个不合规的子 README 的开头改成合规摘要，**然后**再生成：

+ **两个空标题**（父 README 里会出现光秃秃的 `##`）：`04-Feature/TacticSource`（`#`）、
  `08-Advance/Safety`（`#`）。已补成 `# Tactic Source` / `# Safety`。
+ **第 3 行是断句**（正文直接换行，没有 `+ ` 摘要）：`07-Tool/{TritonServerDeploy,nvtriPy,trex}`、
  `03-Workflow/pyTorch-KVCache-ONNX-TensorRT`、`05-Plugin/TritonAOTPlugin`、
  `08-Advance/{EmptyTensor,GreenContext,MIG,TensorRTGraphSurgeon}`。
  统一加一行完整摘要，原正文原样保留在下面。
+ **`07-Tool/DebugUtils`** 的摘要以冒号结尾（`... in \`tensorrt_cookbook\`:`），是个悬空引导句，改写。

生成后的收益：`07-Tool/README.md` 里**原本整个漏掉的 `OnnxVisualization` 回来了**，
`FP16Tuning` 从「Thanks Xuewei Li for providing the solution」变回真正的功能描述，
所有条目都是干净的单行 `+`。全树再无 `^##$` 空标题。
`01-SimpleDemo` 在父 README 里少了两条 bullet，但那两条是**父里手写的旧文案**
（写着「Now only newest TensorRT-10 is recommended」），子 README 的版本更准
（「4 equivalent implementations, 3 in Python and 1 in C++」「TensorRT-10 / 11」），所以是净改善。

**顺带**：我给 `99-Todo/README.md` 加的清单表格把原来的摘要行挤掉了，导致父 README 里
99-Todo 那条变成了中文小节标题，已补回 `+ Todo list and research notes for the cookbook.`。

**这条对应之前记的 S7，现已解决，可以从 `99-Todo` 划掉。**

## `tests/run_tests.py` 的 `--include` 一直是个哑弹（2026-09-01，本轮修复）

上一轮记过一句「`--include "02-API/**"` 似乎没起限制作用，实际从 00 跑到底」，当时归为待查。
这轮再次撞上（`--include "07-Tool/trex/**"` 又在跑 `02-API/Layer/Activation`），查清了，**是两个叠加的 bug**：

### bug 1：`action="append"` 会**追加到** default，而不是替换它

```python
parser.add_argument("--include", action="append", default=["**"], ...)
```

传 `--include "07-Tool/trex/**"` 得到的是 `["**", "07-Tool/trex/**"]`，
而 `_path_match` 是 `any(...)` —— `"**"` 匹配一切，**于是这个开关从来没有起过作用，永远是全量跑**。
改成 `default=None`（`_path_match` 本来就把空 patterns 当「全匹配」）。

### bug 2：`PurePath.match` 的 `**` 不跨 `/`，而且是**右锚定**的

即使修好 bug 1，`--include "02-API/**"` 仍然**一个都选不中** ——
Python 3.13 之前 `PurePath.match` 里的 `**` 等价于 `*`，不跨路径分隔符，
所以它匹配不到 `02-API/Layer/Activation` 这种两层的。**那会比全量跑更糟：静默跑 0 个用例。**
另外 `match` 右锚定，`Cast` 会匹配上 `02-API/Layer/Cast`，也不是这里想要的语义。

改用 `fnmatch.fnmatchcase`（`*` 跨 `/`，且全串匹配），并在 docstring 里写明为什么不用 `PurePath.match`。

### 验证

| 命令 | 修复前 | 修复后 |
| --- | --- | --- |
| `--include "07-Tool/trex/**"` | 232（全量） | **16** |
| `--include "02-API/**"` | 232（全量） | **62** |
| 不带 `--include` | 232 | 232 |
| `--include "07-Tool/**" --exclude "07-Tool/trex/**"` | — | 58 |
| `--case 07-Tool/ONNX` | 1 | 1 |

**这条值得单独说**：上一轮那次「全量 sweep」是**误触发**的，而且因为被 `timeout 5400` 砍掉，
才留下「跑到 07-Tool/Polygraphy 就没了」的记录。以后想跑子集，`--include` 现在是真的能用了。

## `07-Tool/trex` 全绿（2026-09-01）

用户手动装上 graphviz 后，配合修好的 `--include`：**16 selected / 16 passed / 0 failed（171.8 s）**。

上一轮遗留的两个卡点都消了：

+ **`02-DrawEngineGraph`** —— 四个 case 全过，`engine_graph{,_simple,_highlight}.svg` 都是新渲染出来的。
  之前报的 `ExecutableNotFound: dot` 确实只是系统 graphviz 没装（pip 的 `graphviz` 只是绑定）。
+ **`11-ProcessEnginePipeline`** —— 过。它当初有两个 bug 叠在一起：硬编码的绝对路径（上轮已修）
  和缺 `dot`（本轮由用户装上）。
+ **`14-EngineArchive`** —— 过（上轮修的版本戳重建逻辑现在在 B200 上也成立：
  容器换机器后 `data/` 里的旧 engine 会被正确识别为过期并重建）。

## 本轮最终验证（2026-09-01）

装齐依赖（`pip install -e .`）+ 用户装上 graphviz + 修好 `--include` 之后：

| 批次 | 结果 |
| --- | --- |
| `--include "07-Tool/trex/**"` | **16 / 16 passed**（171.8 s） |
| 本轮改动过的 8 个用例（`07-Tool/{ONNX,Onnxruntime,nvtriPy,nvtx,NsightSystems,OnnxGraphSurgeon,OnnxWeightProcess}` + `03-Workflow/pyTorch-KVCache-ONNX-TensorRT`） | **8 / 8 passed**（204.5 s） |

pre-commit 对全部改动文件全绿。**未做任何 commit。**

### 本轮改动清单

**新增**：`07-Tool/ONNX/{main.py,README.md,unit_test.yaml}`（由 `ONNX-TODO` 改名而来，9 个 case）、
`00-Data/get-model-gpt2.py`。
**重写**：`07-Tool/Onnxruntime/{main.py,README.md}`（1 个 case → 7 个）。
**修 bug**：`tests/run_tests.py`（`--include` 从来无效）、
`00-Data/unit_test.yaml`（引用了不存在的 `extract-mnist.py`）、
`.gitignore`（528 MB 的 `.safetensors` 本来可以被 commit）、
`03-Workflow/pyTorch-KVCache-ONNX-TensorRT/main.py`（运行时下载 → 读本地 + 缺失则跳过）、
`04-Feature/TacticSource` 和 `08-Advance/Safety` 的**空标题**。
**文档**：12 个子 README 的前 3 行改成合规摘要 + 重新生成全部父 README；
`00-Data/README.md` 增加下载章节；`nvtriPy` / KVCache 的 H100 数字换成 B200 实测值。

### 下一轮的入口

+ **S8**（新记）：其余 10 处 H100 实测数字要在 B200 上重测，`08-Advance/GreenContext` 必改。
+ 全量 sweep 现在可以放心跑了（依赖齐了、`--include` 也能用了）。
+ `99-Todo` §1.1~§1.7 的候选（S1~S5、G*、L*、M*、R*、T*、P*）尚未开始。

## S2：C++ 例子的 TensorRT 对象释放（2026-09-01）

### 先说结论：**S2 的前提已经过时了**

S2 点名的三个例子是 `01-SimpleDemo/TensorRT-8.0`、`-8.6`、`08-Advance/Safety`，理由是
「它们在 TensorRT 11 下编不过（`kEXPLICIT_BATCH` 没了、`nvinfer1::safe::ICudaEngine` 没了），
所以修了也没法验证」。实际情况：

+ **`01-SimpleDemo/TensorRT-8.0` 和 `-8.6` 这两个目录已经不存在了** ——
  在 `9bbf1807 v2.2.2-trt-11.0` 里被删掉了。`01-SimpleDemo` 现在是扁平的，只有一个 `main.cpp`。
+ **`08-Advance/Safety` 现在只有 `main.py`，没有 C++。**

所以「三个漏对象的 C++ 例子」一个都不在了。S2 变成：**把现存的 C++ 例子审一遍并留下证据**。

### 静态审计：18 个 `.cpp` 全部平衡

写了个按**所有权规则**判断的脚本，而不是数 `delete` 的个数 —— 后者会误判。踩到的点：

+ **`IOptimizationProfile` 不归调用方所有**。`NvInfer.h` 写得很明确：
  「The builder retains ownership of the created optimization profile and returns a raw pointer,
  i.e. **the users must not attempt to delete the returned pointer**」。
  我第一版粗暴计数把 `07-Tool/NetworkPrinter`（4 个 create / 3 个 delete）报成了漏，**是误报**，
  它不删 `profile` 恰恰是对的。
+ `buildSerializedNetwork` 返回的 `IHostMemory` 归调用方所有，虽然它看起来不像工厂函数。

结果：**18 个文件，0 个问题**。

### 动态验证：`compute-sanitizer --leak-check full`

逐个 build 出 `.exe` 再跑 memcheck + leak-check：
`01-SimpleDemo`、`04-Feature/DebugTensor/C++`、`07-Tool/{NetworkPrinter,nvtx}`、
`08-Advance/{CudaGraph,PinnedMemory,C++StaticCompilation}`、
`05-Plugin/{BasicExample,Resource,PluginInsideEngine-C++}` ——
**全部 `0 bytes leaked in 0 allocations` / `0 errors`。**

### 落地物：`tests/check_cpp_ownership.py`

只验证一次是不够的 —— 少一个 `delete` 在跑通的用例里**完全看不出来**。所以把审计固化成脚本，
它知道上面两条 grep 不知道的规则，并且**两个方向都查**：
该删的没删、以及不该删的删了。

自测（改完即刻还原，`git diff` 干净）：

| 注入的回归 | 是否抓到 |
| --- | --- |
| 删掉 `delete config;` | ✅ `` `config` (IBuilderConfig) is never deleted `` |
| 加上 `delete profile;` | ✅ `` `profile` (IOptimizationProfile) is deleted, but the builder owns it `` |

`compute-sanitizer` 是动态的补充：它抓的是**设备**内存，且只覆盖实际跑到的路径；
这个脚本是静态的，覆盖所有分支但只认识 TensorRT 对象。两者互补。

## S1：`trt.BuilderFlag` 每个 flag 的用法（2026-09-01）

### 先摸底：20 个 flag 里 7 个全树零覆盖

派 agent 把 20 个 flag 逐个查了一遍。现状分三类：

+ **有专门目录、覆盖充分**（11 个）：`REFIT`、`EDITABLE_TIMING_CACHE`、`SPARSE_WEIGHTS`、
  `VERSION_COMPATIBLE`、`STRIP_PLAN`、`REFIT_IDENTICAL`、`WEIGHT_STREAMING`、`SAFETY_SCOPE`（QNX 才生效）、
  `ERROR_ON_TIMING_CACHE_MISS`（在 Polygraphy 里）、`EXCLUDE_LEAN_RUNTIME`（Polygraphy，走的是 kwarg 不是裸 flag）、
  `TF32`（就是本轮 `07-Tool/Onnxruntime` 那个 644 倍）。
+ **提到但没演示**（2 个）：`DIRECT_IO`（`04-Feature/DataFormat` 里当使能开关用，本身从不是主角）、
  `REFIT_INDIVIDUAL`（`04-Feature/Refit` 里跟另外三个 refit flag 一起设上，旁边写着 `# [TODO]: add a example`）。
+ **全树零覆盖**（7 个）：`DEBUG`、`GPU_FALLBACK`、`DISABLE_TIMING_CACHE`、`DISABLE_COMPILATION_CACHE`、
  `STRICT_NANS`、`MONITOR_MEMORY`、`DISTRIBUTIVE_INDEPENDENCE`。后两个在
  `02-API/BuilderConfig/main.py` 里是**被注释掉的**，只剩一行描述。

而 `02-API/BuilderConfig` 只演示了 flag API 的**形状**（`set_flag`/`get_flag`/`clear_flag`/`flags`），
拿 `DEBUG` 当占位符，**从不用任何 flag 真的建一次引擎**，所以它说不出任何一个 flag 干了什么。

### 新增 `02-API/BuilderFlag/`

不把 20 个 flag 硬塞进已有的扁平 API 巡览，而是单开一个 leaf，主线是「**测**，不是「描述」」。

| case | 结论 |
| --- | --- |
| `case_set_get_clear` | **TF32 默认就是开的**（`flags == 64`）。而 `flags` 是个裸位掩码，**整体赋值会连默认值一起replace 掉** —— 一行 `config.flags = ...` 就把 TF32 静默关了，这是没人要求过的精度变更，唯一症状是引擎变慢 |
| `case_flag_matrix` | 20 个 flag 逐个单独设置、同一个网络、量 plan 大小。**20 个全部被接受，没有一个建不出来**；**只有 5 个改变了 plan**。「被接受」不等于「有效果」——另外 14 个不是 no-op，它们作用在运行期、build log 或本机没有的硬件上，**plan 大小根本就是错的探针** |
| `case_version_compatible_and_lean_runtime` | `VERSION_COMPATIBLE` 在一个 **54 KiB** 的引擎上加了 **100 MiB**（内嵌 lean runtime，是固定开销，模型越小越离谱）。`EXCLUDE_LEAN_RUNTIME` 能**逐字节**还原到基线（55,692 == 55,692，下了断言）；**而单独用它是静默 no-op** —— 不报错不警告，plan 一模一样 |
| `case_strip_plan` | 大家都拿它缩引擎，但在小网络上 **55,692 → 78,972，反而变大**（加的 refit 元数据比去掉的权重还多）；权重占主导时 **9,508,100 → 109,532，小 87 倍**。用之前先看比例 |
| `case_cache_flags` | `DISABLE_TIMING_CACHE` / `DISABLE_COMPILATION_CACHE`：**plan 一个字节不变**。见下面「自我纠正 2」——最后量的是 timing cache 自身的大小，并意外量出了它的构成 |
| `case_monitor_memory` | plan 完全相同，纯诊断，只在 INFO/VERBOSE 下、只在构建期可见 |
| `case_coverage_map` | flag → 真正演示它的目录，**并断言这张表与 `trt.BuilderFlag.__members__` 完全一致** |

### 最后那个断言是重点

「每个 flag 的用法」这种任务，写完当天是真的，之后就慢慢变假。所以覆盖表不是一段文字，
是一个 `dict`，`case_coverage_map` 拿它和 `trt.BuilderFlag.__members__` 对拍，
**将来 TensorRT 加一个 flag，这个例子会直接失败并点名**（Missing / Stale 都报）。

### 诚实标注

覆盖表里 `DEBUG`、`STRICT_NANS`、`DISTRIBUTIVE_INDEPENDENCE` 一开始被我写成「here」，
但本文件对它们其实只有 matrix 里那行 `+0`，**这属于夸大**。已改成 `listed only - <为什么没法演示>`：
`DEBUG` 是构建期同步、没有可展示的产物；`STRICT_NANS` 需要一个会产生 NaN 的图；
`DISTRIBUTIVE_INDEPENDENCE` 需要一个张量并行组。同时把 `DISABLE_TIMING_CACHE` /
`DISABLE_COMPILATION_CACHE` 真正补成了 `case_cache_flags`，它们才配写「here」。

`02-API/BuilderConfig/main.py` 里那段注释掉的 flag 列表也换成了指向新目录的说明 + 上面两条要点。

### 自我纠正 1：`case_cache_flags` 第一版的测法是错的

第一版量的是「第二次构建快多少」：默认省 6.7%、`DISABLE_TIMING_CACHE` 省 0.5%、
`DISABLE_COMPILATION_CACHE` 省 7.6%。**第三行直接把叙事推翻了**，而且当时构建耗时 24 s
（平时 3.3 s）—— 因为 compute-sanitizer 正在同一块卡上跑，**7 倍的争用噪声把信号全淹了**。
这种数据不能拿来讲故事。

改成量**确定性的东西**：构建结束后 timing cache 自身的字节数（`get_timing_cache().serialize().nbytes`）。
不受争用影响，两次跑出来一致（26,350 / 26,409）。

### 自我纠正 2：`DISABLE_COMPILATION_CACHE` **也**会清空 timing cache

我原本写的是「`DISABLE_COMPILATION_CACHE` 不动 timing cache，它俩是两个不同的缓存」。**实测打脸**：

```txt
(none)                     26,409 B
DISABLE_TIMING_CACHE          198 B
DISABLE_COMPILATION_CACHE     376 B
```

**两个 flag 都把它清空了**，而只有其中一个名字里带 timing。把三个数按
header + 时序 + 编译产物拆开：

| 成分 | 大小 |
| --- | --- |
| 纯 header（timing cache 关掉） | ~198 B |
| + tactic 时序 | ~178 B |
| + JIT 编译出来的 kernel | **~26,033 B（98.6%）** |

也就是说：**大家嘴里的「timing cache」绝大部分是 compilation cache**，
它得名的那个 tactic 时序只占千分之七。这也解释了为什么
`DISABLE_COMPILATION_CACHE` 会让一个名字里带 `timing` 的文件变小 ——
不知道这个布局的话，这看着像个 bug。

这条是本轮 S1 里最意外的发现，而且**是因为第一版结论被实测否掉才挖出来的**。

## S2 最终统计（2026-09-01）

13 个带 Makefile 的 C++ 目录：

| 结果 | 数量 | 说明 |
| --- | --- | --- |
| compute-sanitizer 干净 | **10** | `0 bytes leaked in 0 allocations` / `0 errors` |
| memcheck 下超时 | 2 | `03-Workflow/{pyTorch-ONNX-TensorRT,pyTorch-TensorRT}/C++`，memcheck 下建引擎慢 10~50 倍，1200 s 没跑完。**不是漏，是太慢**；静态检查已覆盖 |
| 不适用 | 1 | `90-Misc/NpyAndNpz` 压根不碰 CUDA（纯 npy/npz 读写），sanitizer 报 `Target application terminated before first instrumented API call`，且它一个 TensorRT 对象都没有 |

静态所有权检查：**18 个 `.cpp` 全过**。
