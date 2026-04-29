# Netron

+ A visualization tool for neural-network graphs, including ONNX and many other formats.

+ [Repo](https://github.com/lutzroeder/Netron) on GitHub.

+ Prepare ONNX models from `00-Data` first, then open them with Netron.

+ Steps to run.

```bash
netron model/*.onnx
```

+ We usually use this tool to inspect graph structure, node metadata, and tensor information.

+ Other related tools:
  + **Nsight Deep Learning Designer** [Link](https://developer.nvidia.com/nsight-dl-designer)
  + **Netron-online** [Link](https://netron.app/), which is an online version of Netron.
  + **onnx-modifier** [Link](https://github.com/ZhangGe6/onnx-modifier), which can edit ONNX in UI. It may become slower for large ONNX files, and supported edit types are limited.
