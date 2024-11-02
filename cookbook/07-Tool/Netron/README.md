# Netron

+ A visualization tool of neural network, which supports **ONNX** and many other frameworks.

+ [Repo](https://github.com/lutzroeder/Netron) on GitHub.

+ Original repository and download website: [Link](https://github.com/lutzroeder/Netron)

+ We need to run `00-Data/get_model.py` firstly, then we can open `00-Data/model*.onnx` by Netron.

+ We usually use this tool to check metadata of compute graph, structure of network, node information and tensor information of the model.

+ Other interesting tools:
  + **Netron-online** [Link](https://netron.app/), which is an online version of Netron.
  + **onnx-modifier** [Link](https://github.com/ZhangGe6/onnx-modifier), which can edit ONNX in UI. But it becomes much slower if ONNX file is large, and the type of graph edition are limited.
