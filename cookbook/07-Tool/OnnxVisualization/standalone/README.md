# ONNX Outliner —— 独立版

把**自己的** ONNX 里重复出现的子图折叠成共享的 local function，让它在 Netron 里能一眼看懂。
不需要 `pip install tensorrt_cookbook`，也不需要 `TRT_COOKBOOK_PATH`：
把**本目录整个拷到任何地方**就能跑。

```bash
pip install -r requirements.txt                    # 只有 3 个必需依赖，见下表

python3 main.py my_model.onnx                       # 输出 my_model-outlined.onnx
python3 main.py my_model.onnx -o out.onnx --report report.json
python3 main.py my_model.onnx --max-level 2 --strictness L2 --strict
python3 main.py --help                              # 全部开关
```

+ 目录内容

```txt
    main.py             CLI 入口，唯一需要直接运行的文件
    requirements.txt    依赖清单
    onnx_outliner/      算法本体，`tensorrt_cookbook/onnx_outliner` 的独立副本
```

+ 依赖（`requirements.txt`）

| 包                   | 必需 | 缺席时                         |
| -------------------- | ---- | ------------------------------ |
| `onnx`               | 是   | 报错并提示安装                 |
| `onnx_graphsurgeon`  | 是   | 报错并提示安装                 |
| `numpy`              | 是   | 报错并提示安装                 |
| `onnxslim`           | 否   | 跳过常量折叠（P0）             |
| `onnxruntime`        | 否   | 跳过数值交叉校验，报告里注明   |
| `tensorrt`           | 否   | 跳过 parse 校验，报告里注明    |

+ 与 cookbook 版的关系

`../01-BasicUsage/main.py` 是 cookbook demo（`import tensorrt_cookbook`，模型来自 `../00-ModelZoo`），
本目录则是给「只想处理自己模型的人」用的。
`onnx_outliner/` 与上游 [`tensorrt_cookbook/onnx_outliner`](../../../tensorrt_cookbook/onnx_outliner)
的唯一差异是 `outliner.py` 里的 `_tensorrt_check` 自己创建 builder / network / logger，
而不是 `from ..utils_class import TRTWrapperV1`。
上游改动后本副本需要手动同步，[`../01-BasicUsage/test_standalone.py`](../01-BasicUsage/test_standalone.py) 会检查是否已经漂移。

+ 算法本身（流水线、严格度 L0~L3、嵌套层次、Loop 后端、实测结果）见 [`../01-BasicUsage/README.md`](../01-BasicUsage/README.md)。
