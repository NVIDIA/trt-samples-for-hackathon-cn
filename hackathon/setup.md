# 安装docker
## 安装docker-ce
sudo apt-get update  
sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common  
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -  
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"  
sudo apt-get update  
sudo apt-get -y install docker-ce  

## 安装nvidia-docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -  
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)  
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list  
sudo apt-get update  
sudo apt-get install -y nvidia-docker2  
sudo systemctl restart docker  

# 下载并运行NGC(NVIDIA GPU Cloud)镜像
## 下载镜像
nvidia-docker pull nvcr.io/nvidia/tensorrt:21.02-py3  
## 运行
#创建并启动container，取名hackathon，并把物理机的/root/workspace挂载到container的/workspace  
mkdir /root/workspace  
nvidia-docker run --name hackathon -v /root/workspace:/workspace -it nvcr.io/nvidia/tensorrt:21.02-py3 bash  
#将来退出这个container，仍然可以用上面给出的名字启动起来，比如  
exit  
nvidia-docker start -i hackathon  
#以下命令都是在该container里面运行的  

# 编译运行例子程序
## 官方例子
#编译TensorRT官方例子  
cd /opt/tensorrt/samples/  
make -j  

## 安装依赖包
#在container里面安装python软件包  
pip3 install nvidia-pyindex  
pip3 install onnx_graphsurgeon  
pip3 install -U protobuf  
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html  
pip3 install torchsummary  

## Hackathon示例
#下载示例程序  
git clone https://github.com/NVIDIA/trt-samples-for-hackathon-cn  
#如果上面的repo无法访问，可用https://gitee.com/shining365/trt-samples-for-hackathon-cn 替代  
cd trt-samples-for-hackathon-cn  
#编译C++例子，其中的动态链接库被python例子使用  
make -j  
cd python  
#该例子生成的onnx被C++例子使用  
python3 app_onnx_resnet50.py  
/opt/tensorrt/bin/trtexec --verbose --onnx=resnet50.onnx --saveEngine=resnet50.trt  
/opt/tensorrt/bin/trtexec --verbose --onnx=resnet50.dynamic_shape.onnx --saveEngine=resnet50.dynamic_shape.trt --optShapes=input:1x3x128x128 --minShapes=input:1x3x128x128 --maxShapes=input:128x3x128x128  
cd ../build  
#可运行任意C++例子，比如  
./AppBasic  
