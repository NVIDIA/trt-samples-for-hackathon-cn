# Onnx outliner

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

+ 简略使用方法

```python
from tensorrt_cookbook import OutlineConfig, outline

report = outline("model.onnx", "model-outlined.onnx", OutlineConfig(strictness="L1", max_level=2))
print(report["patterns"])
```

```bash
python3 -m tensorrt_cookbook.onnx_outliner <model.onnx> -o <model-outlined.onnx> --report report.json
```
