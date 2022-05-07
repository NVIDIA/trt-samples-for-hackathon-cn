# TensorRT Cookbook in Chinese

## 00-MNISTData —— 用到的数据
+ MNIST 数据集，可以从[这里](http://yann.lecun.com/exdb/mnist/)或者[这里](https://storage.googleapis.com/cvdf-datasets/mnist/)下载
+ 数据下载后放进本目录，形如 `./*.gz`，一共 4 个文件
+ 运行下列命令，提取 XXX 张训练图片到 ./train，YYY 张图片到 ./test（不加参数时默认提取 600 张训练图片，100 张测试图片，提取的图片为 .jpg 格式）
```shell
pip install -r requirement.txt
python extractMnistData.py XXX YYY
```
+ 该目录下有一张 8.png，单独作为 TensorRT 的输入数据使用

