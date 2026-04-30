# 把 trt-engine-explorer (TREx) 拆解进 cookbook

> 本文档包含两部分：
> 1. **分析报告**——对 `92-LocalFile/trt-engine-explorer` (TREx) 的结构、接口、可行性分析。
> 2. **迁移草案**——功能→cookbook 目录映射表、迁移顺序、待决策项。
>
> 分析基准：当前签出的是 TREx `dev` 分支（2024-04）。**注意：迁移前应先确认基准分支（见 §分析-4b）。**

---

# 一、分析报告

## 1. 主要结构与主要功能

TREx（TensorRT Engine Explorer）是 NVIDIA/TensorRT OSS 里 `tools/experimental/` 下的实验性工具，用于**探究和可视化 TensorRT engine 的结构与性能**。当前签出的 `dev` 分支代码约 **4756 行**（`trex/` 包）。

### 目录结构

| 目录 | 作用 |
|------|------|
| `trex/` | **核心 Python 包**（17 个模块，~4.7k 行）——数据加载、DataFrame 预处理、绘图、graph 渲染、报告、对比、lint |
| `utils/` | **命令行/脚本工具**：`process_engine.py`(端到端流水线)、`draw_engine.py`(画图)、`parse_trtexec_log.py`(解析日志)、`config_gpu.py`(锁频/功耗)、`device_info.py` |
| `bin/trex` | 统一 CLI 入口（封装 `draw` / `process` 子命令）|
| `notebooks/` | 5 个 Jupyter notebook：tutorial、api-examples、engine_report_card、compare_engines、q_dq_placement |
| `examples/` | pytorch/tensorflow/tensorrt 的 resnet 示例（生成输入用）|
| `tests/` | pytest 单元测试（10 个 test 文件）+ 测试用 JSON 资源 |
| `images/` | README 配图 |

### 核心模块分工（`trex/`）

- **数据层**：`parser.py`(读各种 JSON)、`layer.py`(Layer 抽象)、`engine_plan.py`(核心类 `EnginePlan`)、`df_preprocessing.py`(清洗成 Pandas DataFrame)、`activations.py`(张量/激活)、`archiving.py`(`EngineArchive`——读写 `.tea` 归档)
- **可视化层**：`graphing.py`(975 行，最大——生成 dot/SVG/ONNX graph)、`plotting.py`(plotly 柱状/饼/直方图)、`colors.py`(配色)
- **分析层**：`report_card.py`(711 行，各种性能报告视图)、`compare_engines.py`(619 行，多 engine 对比)、`lint.py`(Conv/Reformat/Slice/QDQ 四类性能隐患检查器)、`excel_summary.py`(导出 Excel)
- **交互层（notebook 专用）**：`interactive.py`、`notebook.py`、`misc.py`

### 核心功能
1. 把 engine graph JSON 加载成 **Pandas DataFrame**，可切片/查询/过滤
2. 把 engine plan **渲染成 SVG/PNG/ONNX**（用 Graphviz）
3. **性能报告**（report card）：latency 分布、卷积/GEMM 统计、内存占用、精度分布、sunburst/scatter/3D 图
4. **多 engine 对比**（layer 对齐、加速比高亮）
5. **Layer linter**：标记潜在性能隐患
6. 端到端流水线（`process_engine.py`）：ONNX → trtexec 建 engine → profile → 生成 JSON → 画 SVG

## 2. 对外暴露的接口

有三层接口：

**(a) 命令行接口**
- `bin/trex draw ...`（画 engine 图）
- `bin/trex process ...`（`process_engine.py`：build/profile/draw 全流程）
- `utils/*.py` 每个都可独立当脚本跑（都有 `make_subcmd_parser` / `__main__`）

**(b) Python 核心 API**（`trex/__init__.py` 默认导出，即 "core"）
只导出 6 个模块：`df_preprocessing`、`misc`、`lint`、`activations`、`engine_plan`、`colors`。核心入口是 **`EnginePlan` 类**（构造参数：graph/profiling/metadata JSON 文件），加上 `summary_dict`、`print_summary`。

**(c) Notebook/绘图 API**（非 core，需显式 import）
`graphing.to_dot / render_dot / DotGraph / OnnxGraph`、`report_card.report_card_*`(约 20 个报告函数)、`compare_engines.compare_engines_*`、`plotting.plotly_*`、`excel_summary.ExcelSummary`、`interactive.InteractiveDiagram`。

## 3. "读 JSON 做统计和可视化" 的理解是否正确？

**基本正确，但有一处需要补充。** TREx 确实**主要**是"读取 TRT build/runtime 产生的 JSON，做统计和可视化"，且标榜**不需要 GPU** 即可分析（只吃 JSON）。

但有两点例外：
- **生成 JSON 那一步需要 GPU 和 trtexec**（`process_engine.py`、`config_gpu.py` 会调用 `trtexec`、`pynvml` 锁频）。这部分是"产生输入"，不是"分析输入"。
- `archiving.py` 依赖 `import tensorrt`（`StreamTrtLogger` 继承 `trt.ILogger`），并非纯 JSON 处理。

更精确的说法：**TREx = 一个 GPU-free 的 JSON 分析/可视化库 + 一套需要 GPU 的 JSON 生成脚本**。搬进 cookbook 时这两部分最好分开对待。

## 4. 拆解进 cookbook 的可行性评估

### (a) 可行性 & 规模估计
**可行，且与 cookbook 定位契合度高**——它天然是"输入 → 处理 → 输出图片/文字"模式。粗略估算：

- **工具函数**：核心 ~30-40 个函数/类值得沉淀进 `tensorrt_cookbook`（parser 读 JSON、DataFrame 预处理、`EnginePlan`、graphing 的 to_dot、report_card 的绘图函数）。可能新增 1 个 `utils_engine_explorer.py` 模块（或拆成 2-3 个）。
- **范例代码**：约 **10-15 个 leaf 目录**的 `main.py`（见迁移草案映射表）。
- **依赖**：新增 `graphviz`、`plotly`(或改 matplotlib)、`pandas`、`openpyxl`(excel)、可能 `kaleido`(plotly 导出静态图)。需评估是否进 `requirements.txt`。
- **工作量**：中大型工程，逐个功能迁移。

### (b) 未合并分支的必要性分析

**重要发现**：签出的 `dev` 是 2024-04 的旧代码。多个分支更新、更值得参考：

| 分支 | 领先 dev | 时间 | 是否引入 |
|------|---------|------|--------|
| **`origin/main`** | 23 commits | **2025-05** | **强烈建议以 main 为基准**——含 TRT 10.10、engine card、`trex card` 子命令、precision stats、安装修复 |
| **`origin/dev-trt-10.9-update`** | 33 commits | 2025-03 | **建议合并**——FP4/INT4 支持、`trex summarize`、从字符串列表建 plan、`shape_call` 算子过滤、name-leak 修复 |
| `origin/dev-engine-card` | 22 commits | 2025-04 | engine-card 功能，多半已并入 main，可跳过 |
| `origin/dev-trt-10.0-update` / `dev-nzmora-python-update` | 106-107 | 2024 | 大概率已并入后续分支，历史参考即可 |
| `origin/tensorboard-integration` | 34 | **2022** | 太老（TensorBoard 集成实验），**不建议** |
| `dev_nzmora_json_validation` / `dev-leod-fix_name_leak` | 1-7 | 小修 | 若已进 main 则跳过 |

**结论**：不要以当前 `dev` 为迁移源。**应以 `origin/main`（最新，含到 TRT 10.10 的更新和 engine card）为基准**，再从 `dev-trt-10.9-update` 挑 FP4/INT4 和 summarize 功能。动手前先做一次分支对比确认。

### (c) 弃用 Jupyter notebook 改成脚本模式
**完全可以，且更适合 cookbook。** 理由：
- cookbook 现有范例全是 `python3 main.py > log-main.py.log` 模式，`run_tests.py` 也依赖这个约定，notebook 无法纳入统一测试。
- notebook 的"交互"主要靠 `interactive.py`/`ipywidgets`（下拉切换视图）——脚本模式下把每个视图输出成独立 PNG/SVG + 文本即可，信息不丢失。
- 绘图用 **plotly**（交互式 HTML）。脚本模式下要么导出静态图片（需 `kaleido`），要么改 matplotlib。这是迁移时的决策点。
- `notebook.py`/`interactive.py`/`display_df` 可直接**弃用**，不必迁移。

### (d) "边搬边删、清空即完成"的策略
**策略很好**，进度可衡量，保留 `trt-engine-explorer-backup` 做对照也是对的。建议：
- **先定基准分支**（见 4b），否则会基于旧代码搬运、白做功。
- 先建**功能→目录映射表**（见迁移草案），同时驱动"搬运"和"删除"，避免遗漏/重复。
- 胶水/notebook 专用代码（`interactive.py`、`notebook.py`、`bin/trex`）可直接删不必搬。
- `tests/` 里的 JSON 资源和 pytest 用例很有价值——可改造成 cookbook 的输入数据和验证。

---

# 二、迁移草案

## A. 迁移前置决策（动手前先敲定）

1. **基准分支**：推荐 `origin/main`（最新）+ 从 `dev-trt-10.9-update` 摘 FP4/INT4/summarize。**待确认。**
2. **绘图方案**：保留 plotly + `kaleido` 导出 PNG，还是统一改 matplotlib？**待确认。**
3. **落地位置**：建议放 `07-Tool/EngineExplorer/`（TREx 本质是分析工具），或单独新编号一节。**待确认。**
4. **工具包组织**：新增单个 `utils_engine_explorer.py`，还是按数据层/绘图层拆多个模块？**待确认。**

## B. 功能 → cookbook 目录映射表

| # | cookbook 目标目录（暂定） | 来源（TREx） | 功能 | GPU? | 依赖 |
|---|--------------------------|-------------|------|------|------|
| 1 | `LoadEnginePlan` | `parser.py` / `engine_plan.py` / `df_preprocessing.py` / `layer.py` | 读 graph/profiling JSON → `EnginePlan` → DataFrame，`print_summary` | 否 | pandas |
| 2 | `ProcessEnginePipeline` | `utils/process_engine.py` | ONNX→trtexec build→profile→生成全部 JSON | **是** | trtexec |
| 3 | `ParseTrtexecLog` | `utils/parse_trtexec_log.py` | 解析 trtexec build/profile 日志 → metadata JSON | 否 | - |
| 4 | `DrawEngineGraph` | `graphing.py` / `utils/draw_engine.py` | engine plan → dot/SVG/PNG | 否 | graphviz |
| 5 | `ExportToONNX` | `graphing.py` (`OnnxGraph`, `make_onnx_tensor`) | engine plan → ONNX（Netron 查看） | 否 | onnx |
| 6 | `ReportCard-Latency` | `report_card.py` (`perf_overview`, `layer_latency_sunburst`, `plot_engine_timings`, `perf_scatter`) | latency 分布 / sunburst / timing | 否 | plotly |
| 7 | `ReportCard-Convolution` | `report_card.py` (`convolutions_overview`) / `df_preprocessing.annotate_convolutions` | 卷积统计 | 否 | plotly |
| 8 | `ReportCard-GEMM` | `report_card.py` (`gemm_MNK`, `gemm_MNK_scatter`, `efficiency_vs_latency_3d`) | GEMM 的 M/N/K 统计与效率 | 否 | plotly |
| 9 | `ReportCard-Memory` | `report_card.py` (`memory_footprint`) | 内存占用 | 否 | plotly |
| 10 | `ReportCard-Precision` | `report_card.py` (`table_view`, `reformat_overview`) | 精度/reformat 分布 | 否 | plotly |
| 11 | `CompareEngines` | `compare_engines.py` | 多 engine 对齐、加速比、对比图 | 否 | plotly |
| 12 | `LayerLint` | `lint.py` (Conv/Reformat/Slice/QDQ Linter) + `report_card.report_card_pointwise_lint` | 性能隐患检查 | 否 | pandas |
| 13 | `ExcelSummary` | `excel_summary.py` | 导出 Excel 汇总 | 否 | openpyxl |
| 14 | `EngineArchive` | `archiving.py` | 读写 `.tea` engine 归档 | 部分 | tensorrt |
| 15 | `DeviceInfo` / `ConfigGpu` | `utils/device_info.py` / `utils/config_gpu.py` | 打印设备信息 / 锁频功耗管理 | **是** | pynvml |
| —(可选) | `EngineCard` / `Summarize` | main / 10.9 分支新增 | engine card、`trex summarize` | 否 | —（需先合分支） |

**不迁移（直接弃用）**：`interactive.py`、`notebook.py`、`misc.display_df`、`bin/trex`（CLI 胶水）、`notebooks/*.ipynb`、`install.sh`、`setup.py`。

**沉淀为工具包**（`tensorrt_cookbook`，非独立范例）：`parser.py`、`layer.py`、`df_preprocessing.py`、`activations.py`、`colors.py`、`misc.py`(部分)、`engine_plan.EnginePlan`、`graphing.py` 核心渲染函数、`plotting.py` 绘图原语。

## C. 建议迁移顺序（依赖从底到上）

1. **数据基座**（先搬）：#1 `LoadEnginePlan` + 底层工具包（parser/layer/df_preprocessing/activations）。这是所有后续功能的前提。
2. **输入生成**：#2 `ProcessEnginePipeline`、#3 `ParseTrtexecLog`（产出可复用的示例 JSON，供后续范例当固定输入）。
3. **可视化基础**：#4 `DrawEngineGraph`、#5 `ExportToONNX`。
4. **报告类**（并行）：#6–#10 各 ReportCard。
5. **高级分析**：#11 `CompareEngines`、#12 `LayerLint`。
6. **辅助**：#13 `ExcelSummary`、#14 `EngineArchive`、#15 `DeviceInfo/ConfigGpu`。
7. **可选新功能**：合入 main/10.9 分支后再搬 EngineCard / Summarize。

每完成一项：在 cookbook 落地范例 + 说明文档，并从 `trt-engine-explorer`（**非 backup**）删除对应代码/文档。`trt-engine-explorer` 清空即任务完成。

## D. 待办进度追踪

- [ ] 确认基准分支（§A.1）
- [ ] 确认绘图方案（§A.2）
- [ ] 确认落地位置与工具包组织（§A.3, §A.4）
- [ ] #1 数据基座
- [ ] #2 #3 输入生成
- [ ] #4 #5 可视化基础
- [ ] #6–#10 报告类
- [ ] #11 #12 高级分析
- [ ] #13 #14 #15 辅助
- [ ] （可选）EngineCard / Summarize
- [ ] `trt-engine-explorer` 清空
