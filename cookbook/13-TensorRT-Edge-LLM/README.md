# 13-TensorRT-Edge-LLM

+ 从 [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) 吸纳的内容。

上游是 NVIDIA 面向 Jetson / DRIVE / DGX Spark 的 C++ LLM 推理运行时。两条前端路径共用同一套 C++
运行时：受支持的 **HuggingFace → ONNX → C++ builder**，以及实验性的 **ONNX-less 直接 engine 构建**。
对 cookbook 的价值集中在**算子与导出**这一侧。

Apache-2.0，与 cookbook 同许可证。调研版本 commit `0f52705`（2026-10-06，release 0.11.0），
候选清单见 [`../99-Todo/trtellm.md`](../99-Todo/trtellm.md)。

| 内容 | 对应条目 | 位置 |
| --- | --- | --- |
| [`StageProfiler/`](StageProfiler/README.md) | B1 | 按阶段分组的 `IProfiler`，暂作独立例子 |
| [`ONNXLessBuilder.md`](ONNXLessBuilder.md) | A3 | 讨论文档，待拍板后再决定是否写成例子 |
| A1 + A2 | 已落地在别处 | [`../05-Plugin/ONNXParserWithPlugin/`](../05-Plugin/ONNXParserWithPlugin/README.md) 的 `case_pytorch_to_onnx_to_trt` |

```bash
cd StageProfiler && python3 main.py
```

A4（`IPluginV3` 的规模化组织）、A5（词表裁剪）、B2、B3、B4、C 系列、D 系列均已确认不做，理由记在
`../99-Todo/trtellm.md`。

## 一条限制

上游的目标硬件是边缘设备，不是我们手上的 B200，其 `examples/` 多数需要具体 checkpoint 和边缘平台。
因此搬运方式是**取其机制、在 MNIST 规模上重建**，而不是照搬它的例子。
