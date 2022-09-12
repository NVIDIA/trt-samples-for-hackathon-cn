#

## 运行方法

+ 需要使用 Anaconda

```shell
conda install caffe # pip install 装不了
python3 buildModelInTensorFlow1.py
mmconvert -sf tensorflow -in ./model.ckpt.meta -iw ./model.ckpt --dstNodeName y -df caffe -om model
# 修改 model.prototxt Reshape 节点附近，“dim: 1 dim: 3136”之间插入两行“dim: 1”，变成“dim: 1 dim: 1 dim: 1 dim: 3136”（不添加或者只添加一行的报错见 result-Dim2.txt 和 result-Dim3.txt）
python3 runModelInTensorRT.py
```

## 参考输出结果，见 ./result.log
