# AddScalarPlugin
+ 给输入张量所有元素同时加上一个标量值
+ 输入张量:
    - [0]: (nBatchSize, n1, n2, ...,nK) int32, K>=0
+ 输入参数:
    - [0]: scalar                       int32,
+ 输出张量:
    - [0]: (nBatchSize, n1, n2, ...,nK) float32/float16
+ 环境：nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，TensoRT 8.2.3）
+ 运行方法：`make test`
+ 参考输出结果，见 ./result.txt
