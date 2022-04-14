# TensorRT Cookbook in Chinese

## 02-API —— TesnorRT API 用法示例
+ DynamicShape+Shuffle，Dynamic shape 模式下一种常见的 shape 操作
+ Int8-QAT，使用 TensorRT 原生 API 搭建一个含有 Quantize / DeQuantize 层的网络
+ Layer，各 Layer 的用法及其参数的示例，无特殊情况均采用 TensorRT8 + explicit batch 模式
+ PrintNetworkInformation，打印 TensorRT 网络的逐层信息

### DynamicShape+Shuffle
+ 环境：nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，TensoRT 8.2.3）
+ 运行方法
```python
cd ./DynamicShape+FCLayer
python dynamicShape+FCLayer.py
```
+ 参考输出结果，见 ./DynamicShape+FCLayer/result.txt

### Int8-QAT
+ 环境：nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，TensoRT 8.2.3）
+ 运行方法：
```shell
cd ./Int8-QAT
python int8-QAT.py
```
+ 参考输出结果，见 ./Int8-QAT/result.txt

### Layer
+ 环境：nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，TensoRT 8.2.3）
+ 运行方法：
    - 可将各 ./Layer/*.md 中的“初始示例代码”粘贴进 ./Layer/test.py，然后运行 `python test.py`
    - 对于某 Layer 其他参数的示例，部分示例提供了完整的代码（例如 ConvolutionNd 层的 num_groups 参数），可同上，将该代码粘贴进 ./Layer/test.py，然后运行 `python test.py`
    - 对于某 Layer 其他参数的示例，部分示例仅提供了代码片段（例如 ConvolutionNd 层的 stride_nd 参数），可将该代码片段粘贴进 ./Layer/test.py 替换初始示例代码“#---# 替换部分”之间的部分，然后运行 `python test.py`
+ 各 ./Layer/*.md 详细记录了所有 Layer 及其参数的用法，还有各参数作用下的输出结果和算法解释
+ GitLab/Github 的 markdown 不支持插入 Latex，所以在线预览时公式部分无法渲染，可以下载后使用支持 Latex 的 markdown 编辑软件（如 Typora）来查看。同时，目录中也提供了各 .md 的 .pdf 文件版本可以直接查看。
+ 各 Layer 维度支持列表 [link](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#layers-matrix)
+ 各 Layer 精度支持列表 [link](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#layers-precision-matrix)
+ 各 Layer 流控制支持列表 [link](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#layers-flow-control-constructs)

### PrintNetworkInforrmation
+ 环境：nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，TensoRT 8.2.3）
+ 运行方法：
```shell
cd ./PrintNetworkInformation
python printNetworkInformation.py
```
+ 参考输出结果，见 ./PrintNetworkInformation/result.txt

