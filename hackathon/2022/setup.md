# 配置开发环境
## 安装最新驱动
请根据自己的GPU型号，从NVIDIA官网下载并安装最新驱动。截至2022年3月，最新的驱动版本是510+。
## 安装nvidia-docker
为了在docker中正常使用GPU，请安装nvidia-docker。

- 如果你的系统是Ubuntu Linux
    - 请参考 [Installing Docker and The Docker Utility Engine for NVIDIA GPUs](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html) 安装nvidia-docker
- 如果你的系统是Windows 11
    - 请先参考 [Install Ubuntu on WSL2 on Windows 11 with GUI support](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview) 把WSL设置好
    - 然后参考 [Running Existing GPU Accelerated Containers on WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#ch05-running-containers) 安装nvidia-docker

# 下载并运行大赛专用镜像
## 下载镜像
```nvidia-docker pull registry.cn-hangzhou.aliyuncs.com/trt2022/dev```  
## 运行
新建目录，作为源代码目录。这里的位置和命名仅供举例用，实际上你可以自己决定。   
```mkdir ~/trt2022_src```    

创建并启动容器，取名trt2022，并把源代码目录挂载到容器的/target   
```nvidia-docker run -it --name trt2022 -v ~/trt2022_src:/target registry.cn-hangzhou.aliyuncs.com/trt2022/dev```     

将来退出这个容器之后，你仍然可以用上面给出的名字trt2022再把它启动起来，就像这样    
```nvidia-docker start -i trt2022```    

启动起来docker之后，你可以从/workspace找到需要你加速的模型encoder.onnx和decoder.onnx。

## 开发程序

前面创建的~/trt2022文件夹有两重身份：它既在你物理机的文件系统里，也在docker镜像里。所以你可以在物理机上对文件夹新建、编辑源文件，又可以在docker里面读取并运行里面的程序。

如果还对接下来怎么做充满疑惑，可以先看一看教学视频充充电。祝你好运！