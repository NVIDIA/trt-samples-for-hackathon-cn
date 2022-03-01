# TensorRT Cookbook in Chinese

## 02-API —— TesnorRT API 用法示例
+ Layer，各 Layer 的用法及其参数的示例，无特殊情况均采用 TensorRT8 + explicit batch 模式
+ [TODO]，其他 API 及其参数的使用方法

### Layer
+ 环境：nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，TensoRT 8.2.3）
+ 运行方法：
    - 可将各 ./Layer/*.md 中的“初始示例代码”粘贴进 ./Layer/test.py，然后运行 `python test.py`
    - 对于某 Layer 其他参数的示例，部分示例提供了完整的代码（例如 ConvolutionNd 层的 num_groups 参数），可同上，将该代码粘贴进 ./Layer/test.py，然后运行 `python test.py`
    - 对于某 Layer 其他参数的示例，部分示例仅提供了代码片段（例如 ConvolutionNd 层的 stride_nd 参数），可将该代码片段粘贴进 ./Layer/test.py 替换初始示例代码“#---# 替换部分”之间的部分，然后运行 `python test.py`
+ 各 ./Layer/*.md 详细记录了所有 Layer 及其参数的用法，还有各参数作用下的输出结果和算法解释
+ GitLab/Github 的 markdown 不支持插入 Latex，所以在线预览时公式部分无法渲染，可以下载后使用支持 Latex 的 markdown 编辑软件（如 Typora）来查看。同时，目录中也提供了各 .md 渲染好了的 .pdf 文件可以直接查看。
+ 参考资料：
    - [TensorRTDoc](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#layers)
    - [TensorRTC++API](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api)
    - [TensorRTPythonAPI](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)

### [TODO]
+ TODO

