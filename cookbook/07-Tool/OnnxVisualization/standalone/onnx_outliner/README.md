# Onnx outliner（独立副本）

这是 [`cookbook/tensorrt_cookbook/onnx_outliner`](../../../../tensorrt_cookbook/onnx_outliner) 的
**独立副本**，供 [`../main.py`](../main.py) 使用。
用法、依赖、与上游的差异见 [`../README.md`](../README.md)。

+ 文件结构

```txt
    config.py       OutlineConfig
    preprocess.py   P0 常量折叠
    graph_ir.py     P1 只读 GraphIR
    ordering.py     P2a recency 规范拓扑序
    discover.py     P2b 最长重复子串 + MDL 打分
    discover_parallel.py  P2c 图空间齐步生长
    verify.py       P3 凸性 / 对齐 / 接口一致性
    rewrite.py      P5 gs.Function + 调用节点
    rewrite_loop.py P5' ONNX Loop 后端
    outliner.py     P4+P6 选择、驱动、校验、报告
    __main__.py     CLI
```

+ 作为库使用

```python
from onnx_outliner import OutlineConfig, outline

report = outline("model.onnx", "model-outlined.onnx", OutlineConfig(strictness="L1", max_level=2))
print(report["patterns"])
```
