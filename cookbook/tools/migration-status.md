# unit_test.sh 迁移进度盘点（2026-02-24）

## 当前状态

- `unit_test.sh` 总数：**175**
- `unit_test.yaml` 总数：**135**
- 其中 `enabled: false`：**22**
- 已复用公共脚本 `tools/unit_test_common.sh` 的 `unit_test.sh`：**134**
- 仍为旧模式（未复用公共脚本）的 `unit_test.sh`：**41**
- 已改为叶子 runner 转调（`trt_run_case "$SCRIPT_DIR"`）的目录：**121**

## 仍未改造目录清单（legacy，41 个）

按一级目录统计：

- `00-Data`：1
- `01-SimpleDemo`：1
- `02-API`：14
- `04-Feature`：2
- `07-Tool`：9
- `08-Advance`：10
- `98-Uncategorized`：4

完整目录列表：

1. `00-Data`
2. `01-SimpleDemo/TensorRT-10`
3. `02-API/Layer/AttentionInput`
4. `02-API/Layer/AttentionOutput`
5. `02-API/Layer/Condition`
6. `02-API/Layer/ConditionalInput`
7. `02-API/Layer/ConditionalOutput`
8. `02-API/Layer/Dequantize`
9. `02-API/Layer/Iterator`
10. `02-API/Layer/LoopOutput`
11. `02-API/Layer/Plugin`
12. `02-API/Layer/PluginV2`
13. `02-API/Layer/PluginV3`
14. `02-API/Layer/Quantize`
15. `02-API/Layer/Recurrence`
16. `02-API/Layer/TripLimit`
17. `04-Feature/CaptureReplay`
18. `04-Feature/ProgressMonitor`
19. `07-Tool/NetworkPrinter`
20. `07-Tool/NetworkSerialization`
21. `07-Tool/NsightSystems`
22. `07-Tool/OnnxGraphSurgeon`
23. `07-Tool/OnnxWeightProcess`
24. `07-Tool/Onnxruntime`
25. `07-Tool/TriPy`
26. `07-Tool/nvtx`
27. `07-Tool/trtexec`
28. `08-Advance/C++StaticCompilation`
29. `08-Advance/CudaGraph`
30. `08-Advance/DataFormat`
31. `08-Advance/EmptyTensor-Unfini`
32. `08-Advance/MultiContext`
33. `08-Advance/MultiDevice`
34. `08-Advance/MultiOptimizationProfile`
35. `08-Advance/MultiStream`
36. `08-Advance/PinnedMemory`
37. `08-Advance/Subgraph-Unfini`
38. `98-Uncategorized/DeviceInfo`
39. `98-Uncategorized/LibraryInfo`
40. `98-Uncategorized/NpyAndNpz`
41. `98-Uncategorized/Number`

> 注：以上统计规则为“目录中 `unit_test.sh` 尚未接入 `tools/unit_test_common.sh`”。

---

## 后续除了改 `unit_test.yaml` / `unit_test.sh` 之外还要做什么

1. **补齐聚合层回归验证**
   每批迁移后至少执行：
   - `python3 tools/run_examples.py --list --include '<batch-glob>'`
   - 聚合入口抽样执行（如 `07-Tool/unit_test.sh`, `08-Advance/unit_test.sh`）

2. **完善 CI 门禁（推荐）**
   在 CI 增加两类检查：
   - 语法检查：`bash -n` 覆盖所有 `unit_test.sh`
   - 结构检查：扫描 legacy 数量是否下降，防止新目录回退到旧模式

3. **补充 README 迁移说明**
   在根 README / cookbook README 增加“新示例接入规范”：
   - 必须提供 `unit_test.yaml`
   - `unit_test.sh` 仅允许薄壳模板
   - `enabled: false` 的适用条件（占位/无可运行代码）

4. **统一 clean 语义审计**
   检查是否存在目录仍手写清理逻辑不一致（日志、临时模型、缓存文件），
   统一到 `unit_test.yaml.clean`，并验证 `--clean` 行为一致。

5. **Runner 能力补强（按需）**
   对复杂目录（多脚本循环、条件执行）评估是否要在 `run_examples.py` 增加：
   - 更细粒度失败分类
   - 更清晰的 summary（按目录/标签聚合）

6. **收口计划与冻结策略**
   当 legacy 清零后，建议：
   - 把 legacy 检查设为强制门禁
   - 对旧模板脚本约定“只允许修 bug，不再新增业务逻辑”

当前 `unit_test.yaml`：

1. `02-API/Layer/Cast/unit_test.yaml`
2. `02-API/Builder/unit_test.yaml`
3. `02-API/BuilderConfig/unit_test.yaml`
4. `02-API/CudaEngine/unit_test.yaml`
5. `02-API/ExecutionContext/unit_test.yaml`
6. `02-API/HostMemory/unit_test.yaml`
7. `02-API/Network/unit_test.yaml`
8. `02-API/ONNXParser/unit_test.yaml`
9. `02-API/OptimizationProfile/unit_test.yaml`
10. `02-API/Runtime/unit_test.yaml`
11. `02-API/Tensor/unit_test.yaml`
12. `05-Plugin/BasicExample/unit_test.yaml`
13. `07-Tool/Polygraphy/Run/unit_test.yaml`

第二批新增（13 个 Layer 目录）：

14. `02-API/Layer/Activation/unit_test.yaml`
15. `02-API/Layer/Assertion/unit_test.yaml`
16. `02-API/Layer/Concatenation/unit_test.yaml`
17. `02-API/Layer/Constant/unit_test.yaml`
18. `02-API/Layer/Convolution/unit_test.yaml`
19. `02-API/Layer/Deconvolution/unit_test.yaml`
20. `02-API/Layer/Elementwise/unit_test.yaml`
21. `02-API/Layer/Gather/unit_test.yaml`
22. `02-API/Layer/MatrixMultiply/unit_test.yaml`
23. `02-API/Layer/Pooling/unit_test.yaml`
24. `02-API/Layer/Reduce/unit_test.yaml`
25. `02-API/Layer/Resize/unit_test.yaml`
26. `02-API/Layer/Slice/unit_test.yaml`

第三批新增（15 个 Layer 目录）：

27. `02-API/Layer/AttentionStructure/unit_test.yaml`
28. `02-API/Layer/Cumulative/unit_test.yaml`
29. `02-API/Layer/DynamicQuantize/unit_test.yaml`
30. `02-API/Layer/Einsum/unit_test.yaml`
31. `02-API/Layer/Fill/unit_test.yaml`
32. `02-API/Layer/GridSample/unit_test.yaml`
33. `02-API/Layer/Identity/unit_test.yaml`
34. `02-API/Layer/IfConditionStructure/unit_test.yaml`
35. `02-API/Layer/LRN/unit_test.yaml`
36. `02-API/Layer/LoopStructure/unit_test.yaml`
37. `02-API/Layer/NMS/unit_test.yaml`
38. `02-API/Layer/NonZero/unit_test.yaml`
39. `02-API/Layer/Normalization/unit_test.yaml`
40. `02-API/Layer/OneHot/unit_test.yaml`
41. `02-API/Layer/Padding/unit_test.yaml`

第四批新增（14 个 Layer 目录）：

42. `02-API/Layer/ParametricReLU/unit_test.yaml`
43. `02-API/Layer/RaggedSoftMax/unit_test.yaml`
44. `02-API/Layer/ReverseSequence/unit_test.yaml`
45. `02-API/Layer/Scale/unit_test.yaml`
46. `02-API/Layer/Scatter/unit_test.yaml`
47. `02-API/Layer/Select/unit_test.yaml`
48. `02-API/Layer/Shape/unit_test.yaml`
49. `02-API/Layer/Shuffle/unit_test.yaml`
50. `02-API/Layer/Softmax/unit_test.yaml`
51. `02-API/Layer/Squeeze/unit_test.yaml`
52. `02-API/Layer/TopK/unit_test.yaml`
53. `02-API/Layer/Unary/unit_test.yaml`
54. `02-API/Layer/Unsqueeze/unit_test.yaml`
55. `02-API/Layer/template/unit_test.yaml`

第五批补充：特殊占位目录（`enabled: false`）

- `02-API/Layer/AttentionInput`
- `02-API/Layer/AttentionOutput`
- `02-API/Layer/Condition`
- `02-API/Layer/ConditionalInput`
- `02-API/Layer/ConditionalOutput`
- `02-API/Layer/Dequantize`
- `02-API/Layer/Iterator`
- `02-API/Layer/LoopOutput`
- `02-API/Layer/Plugin`
- `02-API/Layer/PluginV2`
- `02-API/Layer/PluginV3`
- `02-API/Layer/Quantize`
- `02-API/Layer/Recurrence`
- `02-API/Layer/TripLimit`
- `02-API/Layer/QDQStructure`

同批继续迁移（示例）：03-Workflow 和 04-Feature 共 12 个旧模式目录已转为 `unit_test.yaml` + 叶子薄壳。

第六批继续迁移（18 个目录）：

- 04-Feature 下 10 个目录（Event-unfinish, GPUAllocator, HardwareCompatibility, LabeledDimension, LeanAndDispatchRuntime, ProfilingVerbosity, Refit, Sparsity, TimingCache, VersionCompatibility）
- 04-Feature/WeightStreaming
- 04-Feature/deprecated/AlgorithmSelector
- 06-DLFrameworkTRT/Torch-TensorRT
- 07-Tool 下 5 个目录（CheckTorchOperator, ContextPrinter, EnginePrinter, FP16Tuning, ListAPIs）

第七批补充：特殊占位目录（`enabled: false`）

- 04-Feature/OutputAllocator
- 04-Feature/Safety
- 04-Feature/SerializationConfig
- 07-Tool/Netron
- 07-Tool/ONNX-TODO
- 07-Tool/PolygraphyExtensionTrtexec
- 07-Tool/TRTEngineExplorer

第八批继续迁移（05-Plugin 同构目录，10 个）

- APIs
- BasicExample-V2-deprecated
- DataDependentShape
- IdentityPlugin
- InPlacePlugin
- MultiVersion
- ONNXParserWithPlugin
- PassHostData
- PluginInsideEngine-Python
- QuickDeployablePlugin

第九批继续迁移（05-Plugin 收尾目录，9 个）

- PluginInsideEngine-C++
- PythonPlugin
- ShapeInputTensor
- Tactic+TimingCache
- UseCuBLAS
- UseCuBLAS-V2-deprecated
- UseFP16
- UseINT8-PTQ-V2-deprecated
- UseINT8-PTQ-error

说明：`05-Plugin` 下所有 `unit_test.sh` 已全部完成统一化改造（legacy=0）。

第十批继续迁移（07-Tool/Polygraphy，9 个）

- API
- Check
- Convert
- Data
- Debug
- Inspect
- Plugin-Unfini
- Surgeon
- Template

说明：`07-Tool/Polygraphy` 下所有 `unit_test.sh` 已全部完成统一化改造（legacy=0，Run 目录已在更早批次迁移）。

---

## 已完成的关键改造

- 统一执行器：`tools/run_examples.py`
- 规范文档：`tools/example-spec.md`
- 公共 shell 复用：`tools/unit_test_common.sh`
- 根聚合和多层聚合脚本已接入“优先 runner、回退旧入口”模式

---

## 迁移优先级建议

### P0（高收益，低风险）

优先把“纯 `python3 main.py > log-main.py.log`”的叶子目录补 `unit_test.yaml`，并把对应 `unit_test.sh` 改为：

- `source tools/unit_test_common.sh`
- `trt_bootstrap_runner "$SCRIPT_DIR"`
- `trt_run_case "$SCRIPT_DIR"`

建议第一批（可批量模板化）：

- `02-API/Builder`
- `02-API/BuilderConfig`
- `02-API/CudaEngine`
- `02-API/ExecutionContext`
- `02-API/HostMemory`
- `02-API/Network`
- `02-API/ONNXParser`
- `02-API/OptimizationProfile`
- `02-API/Runtime`
- `02-API/Tensor`

### P1（中收益）

迁移 02-API/Layer 下大量同构目录（多数同样是 `main.py` + 可选 clean）。

### P2（有差异逻辑）

迁移包含 `make test` / `main.sh` / 额外命令的目录（如部分 05-Plugin、07-Tool 子目录），在 `unit_test.yaml` 中用 `pre/run/post/clean` 显式表达。

---

## 推荐节奏

1. 每批 10~20 个目录
2. 每批只做两件事：
   - 新增 `unit_test.yaml`
   - 叶子 `unit_test.sh` 改为 `trt_run_case` 薄包装
3. 每批结束执行：
   - `bash -n` 语法检查
   - `python3 tools/run_examples.py --list --include '<batch-glob>'`

---

## 验收标准（阶段性）

- 聚合层 `unit_test.sh` 不再包含业务逻辑（仅调公共 helper）
- 叶子目录逻辑以 `unit_test.yaml` 为主
- 旧 `unit_test.sh` 仅保留兼容壳，逐步收敛到最小模板
