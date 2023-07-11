#

## Steps to run

```shell
./command.sh
```
+ Output for reference: ./result-*.log


+ A python library for ONNX compute graph edition, which different from the library *onnx*.

+ Installation: `pip install nvidia-pyindex onnx-graphsungeon`

+ Document [Link](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html)

+ The example code here refers to the NVIDIA official repository about TensorRT tools [Link](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/examples).

+ Function:
  + Modify metadata/node / tensor / weight data of compute graph.
  + Modify subgraph: Add / Delete / Replace / Isolate
  + Optimize: constant folding / topological sorting / removing useless layers.