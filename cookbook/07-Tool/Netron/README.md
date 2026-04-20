# Netron

+ A visualization tool for neural-network graphs, including ONNX and many other formats.

+ [Repo](https://github.com/lutzroeder/Netron) on GitHub.

+ Original repository and download website: [Link](https://github.com/lutzroeder/Netron)

+ Prepare ONNX models from `00-Data` first, then open them with Netron.

+ Steps to run.

```bash
cd ../../00-Data
python3 get_model_part1.py
python3 get_model_part2.py
netron model/*.onnx
```

+ We usually use this tool to inspect graph structure, node metadata, and tensor information.

+ Other interesting tools:
  + **Netron-online** [Link](https://netron.app/), which is an online version of Netron.
  + **onnx-modifier** [Link](https://github.com/ZhangGe6/onnx-modifier), which can edit ONNX in UI. It may become slower for large ONNX files, and supported edit types are limited.
