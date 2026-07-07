# trex → cookbook 迁移进度

> 本文件记录 `trt-engine-explorer` (TREx) 迁移进 cookbook 的实时进度，防止会话中断丢失。
> 总体计划、功能映射表见 `99-Todo/trex.md`。

## 关键决策（已确认）

1. **基准分支**：以 `origin/main` 为基准，合并 `origin/dev-trt-10.9-update`。
2. **绘图**：统一用 matplotlib，保存图片/文字到文件；不要交互/浏览器/hover；子图用 GridSpec 或拆分；配色用 matplotlib 默认；Range Slider 砍掉，改为脚本内可手调的画图参数；动画→gif 或拆分 png；3D→2D。
3. **工具函数**：全部放入单文件 `tensorrt_cookbook/utils_engine_explorer.py`。
4. **例子位置**：`07-Tool/trex/`（`07-Tool/trex-TODO` 保持不动）。
5. **去 pandas**：用 numpy + list/dict 替代。
6. **删除策略**：仅删“已成功迁移”的 trex 代码；弃用/未迁移的暂时保留；`trt-engine-explorer-backup` 永不删。

## 进度总览

| # | 功能 | 状态 | cookbook 位置 | 备注 |
|---|------|------|--------------|------|
| 0 | 分支合并 (main + 10.9) | ✅ 完成 | subrepo 分支 `wili/base-main` | 冲突全部以 main 侧解决（main 为超集） |
| 1 | 基础数据层 + LoadEnginePlan | ✅ 完成 | `utils_engine_explorer.py` + `07-Tool/trex/00-LoadEnginePlan` | 见下 |
| 6 | ReportCard-Latency | ✅ 完成 | `07-Tool/trex/01-ReportCardLatency` | 见下 |
| 4 | DrawEngineGraph (SVG/dot) | ✅ 完成 | `07-Tool/trex/02-DrawEngineGraph` | 见下，需 graphviz dot |
| 7 | ReportCard-Convolution | ✅ 完成 | `07-Tool/trex/03-ReportCardConvolution` | 见下 |
| 8 | ReportCard-GEMM | ✅ 完成 | `07-Tool/trex/04-ReportCardGEMM` | 见下 |
| 9 | ReportCard-Memory | ✅ 完成 | `07-Tool/trex/05-ReportCardMemory` | 见下 |
| 10 | ReportCard-Precision | ✅ 完成 | `07-Tool/trex/06-ReportCardPrecision` | 见下 |
| 11 | CompareEngines | ✅ 完成 | `07-Tool/trex/07-CompareEngines` | 见下 |
| 3 | ParseTrtexecLog | ✅ 完成 | `07-Tool/trex/09-ParseTrtexecLog` | 见下 |
| 5 | ExportToONNX | ✅ 完成 | `07-Tool/trex/10-ExportToONNX` | 见下 |
| 2 | ProcessEnginePipeline | ✅ 完成 | `07-Tool/trex/11-ProcessEnginePipeline` | 见下，需 GPU |
| 12 | LayerLint | ✅ 完成 | `07-Tool/trex/08-LayerLint` | 见下 |
| 13 | ExcelSummary | ✅ 完成 | `07-Tool/trex/12-ExcelSummary` | 见下，需 openpyxl |
| 14 | EngineArchive | ✅ 完成 | `07-Tool/trex/14-EngineArchive` | 见下，需 tensorrt |
| 15 | DeviceInfo / ConfigGpu | ✅ 完成 | `07-Tool/trex/13-DeviceInfo` | 见下，需 GPU/pynvml |
| + | EngineCard / Summarize | ✅ 完成 | `07-Tool/trex/15-EngineSummarize` | 见下 |

## 已完成明细

### #0 分支合并
- 在 subrepo `cookbook/92-LocalFile/trt-engine-explorer` 新建分支 `wili/base-main`（基于 `origin/main`），
  合并 `origin/dev-trt-10.9-update`。
- 冲突文件（CHANGELOG.md, bin/trex.py, install.sh, setup.py, trex/colors.py, trex/graphing.py,
  utils/draw_engine.py, utils/summarize_engine.py）全部取 main 侧（main 为超集，含 INT4/FP4、
  ego 渲染、precision stats、engine card、TRT 10.10）。手工清理了 bin/trex.py 的嵌套冲突标记。
- 合并后新增：`utils/gen_engine_card.py`, `utils/summarize_engine.py`, 及 engine_plan 的 precision-stats。
- 已提交（merge commit）。版本 `0.2.1`。

### #1 基础数据层 + LoadEnginePlan
**工具函数** `tensorrt_cookbook/utils_engine_explorer.py`（已注册进 `tensorrt_cookbook/__init__.py`），去 pandas：
- `Activation`（张量区域抽象，含格式字典 + 精度/字节表）
- `Layer` + `fold_no_ops`（层抽象，folds NoOp）
- JSON 读取：`read_graph_file` / `read_profiling_file` / `read_timing_file` /
  `get_device_properties` / `get_performance_summary` / `get_builder_config` / `import_graph_file`
- **`EnginePlan`**：核心类。用 `self.records`（list[dict]）替代 pandas DataFrame；
  `plan.col(name)` 取 numpy 数组；`get_layers_by_type` / `find` / `get_bindings` / `summary`。
  профил数据用逐条或按名合并（自动处理层数不一致的“partial profiling”情况）。
- 分组聚合（替代 pandas groupby）：`group_count` / `group_sum` / `group_mean`
- 汇总：`summary_dict` / `print_summary` / `json_summary`
- 精度统计：`compute_precision_stats` / `print_precision_stats` / `json_precision_stats`

**验证**：用 trex 自带 `tests/inputs/mobilenet.qat...` JSON 加载，pct_time 合计=100.000，
精度分布合计=100%，total_runtime 与独立计算的 avgMs 之和一致。
（原始 trex 在当前新版 pandas 下反而因 `fillna(0)` 报错——正说明去 pandas 的价值。）

**范例** `07-Tool/trex/`：
- `get_data.py`：用 trtexec 从 `00-Data/model/model-trained-int8-qat.onnx` 构建 INT8 引擎，
  导出 graph/profile/timing JSON 到 `data/`（固定 shape，保证字节统计为正）。
- `main.py`：3 个 case——加载+summary+precision stats；按类型/单层延迟报告；
  matplotlib 双面板图（GridSpec：延迟条形 + 权重精度饼图）存 `data/overview.png`。
- `README.md` / `unit_test.yaml`（pre: get_data.py, run: main.py）/ `.gitignore`（忽略 data/）。
- **已通过 `run_tests.py --case 07-Tool/trex`（passed，18.5s）。**

### 目录结构调整（重要）
`07-Tool/trex/` 改为**容器 + 编号子范例**结构，多个范例共享一份引擎 JSON：
- `07-Tool/trex/get_data.py`：共享数据生成器。用 trtexec 构建 INT8 引擎并导出
  graph/profile/timing JSON 到 `07-Tool/trex/data/`。**已存在则跳过重建**（各子范例的
  `pre:` 都调用它，因 run_tests 是**顺序执行**，共享安全）。
  - 关键修正：profile 步骤加 `--separateProfileRun`，否则 `--dumpProfile` 会导致
    trtexec 跳过 e2e timing 导出（timing.json 缺失）。
- 每个子范例：`main.py` 从 `../data/` 读 JSON，图片存到自己目录（本地 `*.png`）。
- `.gitignore` 忽略 `data/` 与 `*.png`（生成物）；log 提交以展示输出。
- `00-LoadEnginePlan/`：原 #1 范例迁入此处。

### #6 ReportCard-Latency
`07-Tool/trex/01-ReportCardLatency/main.py`——把 trex 的 `report_card_perf_overview` /
`layer_latency_sunburst` / `plot_engine_timings`（原为 plotly 交互下拉 ~10 视图）改为
matplotlib 存图，6 个 case / 6 张 PNG：
- `latency_by_type.png`：按类型的延迟(ms/%)+层数（三面板 GridSpec，layer_colormap 上色）
- `latency_per_layer.png`：最慢 N 层横条，按类型上色 + 手工图例（N=n_top_layers 可调）
- `latency_distribution.png`：每层延迟%直方图（hist_bins 可调）
- `precision_rollup.png`：按精度的层数/延迟占比双饼（precision_colormap）
- `latency_by_type_precision.png`：按类型堆叠(按精度)条形——**替代原 type/precision sunburst（3D→2D）**
- `engine_timings.png`：timing.json 逐迭代延迟散点 + 均值线
工具模块新增：`precision_colormap` / `layer_colormap` / `colors_for()`（移植自 trex/colors.py，matplotlib 兼容）。
Range-Slider 砍掉，改为脚本顶部 `n_top_layers`/`hist_bins` 参数。**已通过 run_tests。**

### #4 DrawEngineGraph
`07-Tool/trex/02-DrawEngineGraph/main.py`——把 trex 的 `graphing.DotGraph`/`to_dot`/`render_dot`
及 `utils/draw_engine.py`（原 975 行、依赖 PlanGraph/regions/memory-node 的复杂实现）改写为
**精简 graphviz 渲染器**（按用户"不必一模一样、找等价实现"的指示）：
- 工具模块新增（graphviz 惰性 import，不影响其他功能的轻依赖）：
  - `build_engine_graph(plan, ...)`：一层一节点（按类型 layer_colormap 上色，标注 名/类型/延迟），
    边=生产者→消费者共享张量（按精度 precision_colormap 上色，标注 shape/dtype）；
    图输入/输出 binding 画成灰色端节点。可选参数 display_layer_names/display_latency/
    display_edge_details/display_bindings/display_constants/highlight_layers/max_name_len。
  - `render_engine_graph(plan, path, format)`：渲染 svg/png/dot 等。
  - `clean_layer_name()`：转义 graphviz 特殊字符。
- 4 个 case：默认全图(png) / SVG(矢量) / 简化图(无边标签无 binding) / 高亮最慢层(红框)。
- **需系统 graphviz `dot` 二进制**（已装 2.43.0）。已通过 run_tests（3/3, 33.8s）。

### #7 ReportCard-Convolution
`07-Tool/trex/03-ReportCardConvolution/main.py`——移植 `report_card_convolutions_overview` +
`df_preprocessing.annotate_convolutions`。
- 工具模块新增 `annotate_convolutions(plan)`：对每个 Convolution 层计算 MACs、
  arithmetic intensity(MACs/byte)、compute/memory efficiency(每 ms)、等效 GEMM 的 M/N/K，
  返回 list[dict]（去 pandas 的 annotate_convolutions 版本）。
- 3 个 case：per-conv 指标文字表 / 2×2 bar 面板(延迟按精度上色、MACs、算术强度、footprint) /
  **roofline 散点(算术强度 vs 计算效率，点大小/颜色=延迟)——2D 替代原 plotly 3D M-N-K scatter**。
- 已通过 run_tests（4/4, 37.2s）。为 #8 GEMM 打好基础（M/N/K 已就绪）。

### #8 ReportCard-GEMM
`07-Tool/trex/04-ReportCardGEMM/main.py`——移植 `report_card_gemm_MNK` /
`report_card_gemm_MNK_scatter` / `report_card_perf_scatter`（原 plotly 3D scatter）。
复用 `annotate_convolutions`（卷积当 implicit GEMM，M=N*P*Q, N=K_out, K=C*R*S）。3 个 case：
- `gemm_mnk_bars.png`：每卷积 M/N/K 分组条形(log 轴)
- `gemm_mnk_vs_latency.png`：M/N/K 各 vs 延迟 三面板散点(色=footprint)——2D 端口
- `gemm_mnk_projection.png`：M×N 散点(点大小=K, 色=延迟)——**3D M-N-K scatter 的 2D 投影**
已通过 run_tests（5/5, 44.3s）。

### #9 ReportCard-Memory
`07-Tool/trex/05-ReportCardMemory/main.py`——移植 `report_card_memory_footprint`。
用 records 里的 weights_size / total_io_size_bytes / total_footprint_bytes。4 个 case：
- 文字表：总权重/激活/footprint + footprint 最大的层
- `footprint_per_layer.png`：最大 N 层 weights+activations 堆叠横条
- `footprint_by_type.png`：按类型 weights+activations 堆叠条
- `footprint_distribution.png`：weights/activations/total 三直方图
已通过 run_tests（6/6, 48.5s）。

### #10 ReportCard-Precision
`07-Tool/trex/06-ReportCardPrecision/main.py`——移植 report_card_perf_overview 里的精度相关视图
(precision_per_layer / precision_per_type sunburst / precision statistics) + `report_card_reformat_overview`。
5 个 case：
- 文字精度字节统计（print_precision_stats）
- `precision_per_layer.png`：每层延迟按精度上色
- `precision_stats.png`：输入/输出激活/权重 三面板按精度字节条形
- `precision_per_type.png`：按类型层数堆叠(按精度)——**2D 替代 precision sunburst**
- `reformat_overview.png`：Reformat 层按 Origin(如 QDQ) 计数/延迟占比——port of reformat_overview
配色沿用 trex 方案（INT8 绿/FP32 红/INT32 灰）。已通过 run_tests（7/7, 53.6s）。

### #11 CompareEngines
`07-Tool/trex/07-CompareEngines/main.py`——移植 compare_engines.py
(compare_engines_overview / compare_engines_summaries_tbl)。
- **get_data.py 改为构建两个引擎**：`model.*`(INT8) + `model.fp16.*`(FP16)，同一 MNIST 网络不同精度。
  其余范例仍只用 INT8。skip-if-exists 覆盖两者。
- 4 个 case：并排摘要表+总体 speedup(INT8 比 FP16 快 1.11x) / 按类型堆叠延迟(每引擎一列) /
  按类型分组横条比较+每类型 speedup 表 / 按精度堆叠延迟。
- engines 列表可加更多引擎做多路比较。已通过 run_tests（8/8, 81.8s；首个范例现构建 2 引擎）。

### #12 LayerLint
`07-Tool/trex/08-LayerLint/main.py`——移植 lint.py 四个 linter(Conv/Reformat/Slice/QDQ)。
工具模块新增（去 pandas，基于 records + create_activations）：
`lint_convolutions`(TensorCore 未加速/量化卷积浮点输出/通道对齐) / `lint_reformats` /
`lint_slices` / `lint_qdq`(未融合 Q/DQ) / `lint_engine`(全跑，返回 {类别:[hazards]})；
另导出 `create_activations`。2 个 case：文字隐患报告 / 各类别隐患数条形图。
INT8 MNIST 上检出 3 处隐患(conv1 对齐/conv2 浮点输出/reformat Float→Int8)。已通过 run_tests（9/9, 86.6s）。

### #3 ParseTrtexecLog
`07-Tool/trex/09-ParseTrtexecLog/main.py`——移植 `utils/parse_trtexec_log.py`（去掉 archiving 依赖，直接读普通日志文件）。
工具模块新增 `parse_build_log` / `parse_profiling_log` / `write_build_metadata` / `write_profiling_metadata`。
解析 trtexec build/profile 日志的 `=== Section ===` 块 → metadata JSON（device_information/build_options/performance_summary 等）。
喂回 EnginePlan(profiling_metadata_file/build_metadata_file) 后，print_summary 的 Device/Builder/Performance
三段被填满（#00 里原本是空的）。实测 H100/CC9.0/throughput 14033.8。已通过 run_tests（10/10, 88.3s）。

### #5 ExportToONNX
`07-Tool/trex/10-ExportToONNX/main.py`——移植 `graphing.OnnxGraph`/`make_onnx_tensor`（不依赖 PlanGraph，
直接从 EnginePlan 生成）。工具模块新增 `export_engine_to_onnx(plan, path)`（onnx 惰性 import）：
每层一个 onnx node（Convolution→Conv, Pooling→Max/AveragePool），按张量名连边，binding 作图 I/O，
属性尽力附加(跳过嵌套结构)。注意：导出的 ONNX 仅供 Netron 可视化，用 TRT 层名当 op type，
onnx.checker/onnxruntime 会拒绝——这是预期的（原 trex 也如此）。已通过 run_tests（11/11, 91.3s）。

### #2 ProcessEnginePipeline
`07-Tool/trex/11-ProcessEnginePipeline/main.py`——移植 `utils/process_engine.py` 的端到端流程。
**自己构建引擎（需 GPU+trtexec），不用共享 get_data**，输出到 `pipeline_out/`。5 步串起全部环节：
build(graph.json) → profile(profile/timing.json) → 解析日志(metadata, #09) → 渲染 SVG(#02) →
EnginePlan 加载+summary+precision(#00/#09)。已通过 run_tests（12/12, 117s）。

### #13 ExcelSummary
`07-Tool/trex/12-ExcelSummary/main.py`——移植 `excel_summary.ExcelSummary`（改用 openpyxl 替代
pandas.ExcelWriter+xlsxwriter，嵌入 matplotlib PNG 替代 plotly）。工具模块新增
`write_engine_excel(plan, path, image_files=...)`（惰性 openpyxl）：写 Summary/Layers/Precision
三张表 + 可选图片工作表。范例生成一张延迟条形图嵌入，产出 4 工作表 xlsx。需 `pip install openpyxl`。
已通过 run_tests（13/13）。

### #15 DeviceInfo / ConfigGpu
`07-Tool/trex/13-DeviceInfo/main.py`——移植 `utils/device_info.py` + `utils/config_gpu.py` 只读部分
（改用 pynvml 替代 pycuda）。工具模块新增 `query_device_info` / `sample_gpu_state` / `get_max_clocks`。
2 个 case：各 GPU 元信息(名/显存/最大时钟)转 JSON / 当前温度功耗时钟 + 与最大时钟比较提示是否降频。
**锁频需 root**，故只读不锁，仅提示 nvidia-smi 锁频命令。需 GPU+pynvml。已通过（H100 实测）。

### #14 EngineArchive
`07-Tool/trex/14-EngineArchive/main.py`——移植 `archiving.EngineArchive`（TEA = 引擎归档 zip）。
工具模块新增 `EngineArchive(path, mode='w'|'r')`：writef_txt/writef_bin/add_file/archive_plan_info
(反序列化引擎导出属性+IO张量 JSON) / namelist/readf。改进：把原 override 语义改为显式 w/r 模式以支持读回；
反序列化时屏蔽 weight-streaming 属性避免噪声日志。范例把 engine+graph/profile JSON+plan_cfg 打包成
.tea 再读回。需 tensorrt。已通过。

### +EngineCard / Summarize
`07-Tool/trex/15-EngineSummarize/main.py`——移植 main/10.9 分支的 `utils/summarize_engine.py` +
`utils/gen_engine_card.py`。工具模块新增 `summarize_engine_tactics(plan, group_tactics, sort_key)`
(按 tactic 分组，去 hash 后合并，count + latency%，排序)。2 个 case：tabulate 打印 tactic 延迟表 /
生成**Markdown 引擎卡片** `engine_card.md`（原 HTML 卡片会弹浏览器，按 cookbook 风格改 Markdown：
汇总+按类型延迟+top tactics+精度字节+lint 隐患）。已通过。

## 迁移完成情况
**映射表 15 项 + EngineCard/Summarize 全部完成**（外加 #0 分支合并）。`07-Tool/trex/` 共 16 个子范例
(00-15)，全部通过 `run_tests.py --tags trex`。工具全部集中在 `tensorrt_cookbook/utils_engine_explorer.py`
(无 pandas/plotly；graphviz/onnx/openpyxl/pynvml/tensorrt 均惰性 import)。

## 删除记录（已执行）
所有功能迁移完成后，按用户确认的"只删已迁移的代码、留文档"方案，在 `trt-engine-explorer`
（**非 backup**）执行 `git rm -r`：删除 `trex/ utils/ bin/ notebooks/ examples/ tests/`
（共 118 个文件）。保留：README/CHANGELOG/KNOWN_ISSUES/RESOURCES/LICENSE、requirements*、
setup.py、install.sh、images/、.vscode/、__init__.py、.git。
`trt-engine-explorer-backup` 完整保留（17 项）作对比。删除后 cookbook 范例全部照常运行
（只依赖 `tensorrt_cookbook`，不依赖 trex）。删除已 `git rm` 暂存，**尚未 commit**，可随时从
git/backup 恢复。

## 任务完成
- 16 个功能全部迁移到 `07-Tool/trex/`（00-15）+ 工具集中在 `utils_engine_explorer.py`。
- 分支已合并（main + 10.9）。
- trt-engine-explorer 已按约定"清空代码、留文档"。
- 剩余可选：如需彻底清空文档/.git 或提交 subrepo 删除，等用户指示。
