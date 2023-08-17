# NVIDIA TensorRT Hackathon 2023 —— 生成式AI模型优化赛

## 大赛报名入口
点击[报名入口](https://tianchi.aliyun.com/competition/entrance/531953/information)，注册阿里云账号，报名参赛。

## 大赛介绍
TensorRT 作为 NVIDIA 英伟达 GPU 上的 AI 推理加速库，在业界得到了广泛应用与部署。与此同时，TensorRT 开发团队也在持续提高产品的好用性：一方面让更多模型能顺利通过 ONNX 自动解析得到加速，另一方面对常见模型结构（如 MHA）的计算进行深度优化。这使得大部分模型不用经过手工优化，就能在 TensorRT 上跑起来，而且性能优秀。

过去的一年，是生成式 AI（或称“AI生成内容”） 井喷的一年。大量的图像和文本被计算机批量生产出来，有的甚至能媲美专业创作者的画工与文采。可以期待，未来会有更多的生成式AI模型大放异彩。在本届比赛中，我们选择生成式AI模型作为本次大赛的主题。

今年的 TensorRT Hackathon 是本系列的第三届比赛。跟往届一样，我们希望借助比赛的形式，提高选手开发 TensorRT 应用的能力，因此重视选手的学习过程以及选手与 NVIDIA 英伟达专家之间的沟通交流。我们期待选手们经过这场比赛，在 TensorRT 编程相关的知识和技能上有所收获。

## 赛题说明

本赛分初赛和复赛。

### 初赛

初赛是利用 TensorRT 加速带有 [ControlNet](https://github.com/lllyasviel/ControlNet) 的 Stable Diffusion canny2image pipeline，以优化后的运行时间和出图质量作为主要排名依据。

- 初赛期间我们将建立包含所有选手的技术交流群，供大家研讨用
- 我们专门为此次比赛准备了系列讲座，为了能更顺利地完成比赛，请参赛者观看学习
    - 讲座地址：https://www.bilibili.com/video/BV1jj411Z7wG/
    - 配套范例：[cookbook](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master)
- 初赛结束时将组织一次讲评，介绍优化该模型的技巧

初赛不提供开发机，参赛选手需要自备带有 GPU 的 Linux / Windows 11 (WSL2) 开发机，并在给定 docker 中用赛方提供的模型文件、开发工具完成模型在 TensorRT 中的构建、精度验证和性能测试，并提交最终代码。

### 复赛

- 复赛是开放赛题，各选手可自由选择公开的transformer模型，在TensorRT上优化运行
- TensorRT 即将发布用于大型语言模型推理的工具，我们推荐在复赛中使用此工具，评分时会有一定加分
- 复赛赛制与[2022年](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/Hackathon/2022/HackathonGuide.md#%E5%A4%8D%E8%B5%9B%E8%A1%A5%E5%85%85%E8%AF%B4%E6%98%8E)基本保持一致

# 配置开发环境

## 安装最新驱动

请根据自己的 GPU 型号，从 NVIDIA 官网下载并安装最新驱动。截至2023年6月，最新的驱动版本是530+。

## 安装nvidia-docker

为了在 docker 中正常使用GPU，请安装 nvidia-docker。

- 如果你的系统是Ubuntu Linux
  - 请参考 [Installing Docker and The Docker Utility Engine for NVIDIA GPUs](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html) 安装nvidia-docker
- 如果你的系统是Windows 11
  - 请先参考 [Install Ubuntu on WSL2 on Windows 11 with GUI support](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview) 把WSL设置好
  - 然后参考 [Running Existing GPU Accelerated Containers on WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#ch05-running-containers) 安装nvidia-docker

## 下载并运行大赛专用镜像

### 下载镜像

```registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2```

### 运行

/home/player/ControlNet作为源代码目录已在docker中，后续开发请在此目录中完成。

创建并启动容器，取名trt2023
```nvidia-docker run -it --name trt2023 registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2```

将来退出这个容器之后，你仍然可以用上面给出的名字trt2023再把它启动起来，就像这样
```nvidia-docker start -i trt2023```

启动起来 docker 之后，你需要将 PyTorch 模型转成 ONNX 格式，再转成 TensorRT engine 序列化文件（.plan），然后将 PyTorch 模型替换为 TensorRT engine ，最后生成图片。

### 开发程序

请注意及时保存你docker中的 /home/player/ControlNet 代码和文件

如果还对接下来怎么做充满疑惑，可以先看一看教学视频充充电。祝你好运！

# 初赛补充说明

+ 初赛仅提供测评服务器不提供开发机，参赛选手需要自备带有 GPU 的 Linux / Windows 11 (WSL2) 开发机，并在给定 docker 中用赛方提供的模型文件、开发工具完成模型在 TensorRT 中的构建、运行、精度验证和性能测试，并将代码提交至指定仓库以供测评服务器打分排名。
+ 初赛使用的镜像：`registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2`
  
  - 该镜像基于NVIDIA 英伟达官方镜像扩充而来，包含 CUDA 11.8，TensorRT 8.6.1 以及比赛用到的开发工具、模型文件、测试样例数据。请根据"配置开发环境"部分的说明进行使用。
  - 初赛不会提供 ONNX 模型，选手需要自行完成 PyTorch 到 TensorRT 的全部转换过程
  - 初赛使用 Perceptual Distance (PD) 评估生成图片的质量，/home/player/ControlNet/compute_score.py 已包含 PD score 的计算方法，即 TensorRT 生成图片与 PyTorch FP32 生成图片之间的差异程度，PD score 越小越好
  - 初赛包含 Clip 、 UNet 、 ControlNet 、VAE Encoder 和 Decoder 等较多模型，时间有限的话可以优先优化 UNet 和 ControlNet 模型
  - 与去年不同，本次初赛优化的是包含多次迭代的 pipeline，除了对单个模型的优化，还可以尝试 pipeline 层次的加速，以获取更高分数
+ 代码验证与提交
  
  - 请保证在 docker 里能正常运行你的代码，并且无论编译时还是运行时，都不依赖网络下载任何代码或数据，即代码需要是完整的、自包含的。如果确实需要在 docker 里面新增开发库或软件，请在交流群里反应给赛方，我们将添加到比赛用的 docker image 中
  - 比赛所需代码在 /home/player/ControlNet 目录下。 canny2image_TRT.py 文件中是需要加速的canny2image PyTorch pipeline。请修改其中的initialize（） 与 process（）函数来加速pipeline。（不要更改函数名称与参数表）每调用一次process 函数就需要能够生成一张新的图片。 计分程序会以process函数运行时间与生成图片与原始torch pipeline 生成图片之间的差距来决定分数。
  - 在使用TRT 加速pipeline 的过程中，选手会需要将PyTorch 中的模型转换为TRT engine。请将这个转换过程的脚本写入 preprocess.sh。 所有的中间文件请不要上传到git repo。 计分程序运行canny2image_TRT.py 前会先运行preprocess.sh 在本地生成TRT engine。所有preprocess.sh 会产生的中间文件（比如 .onnx, .plan, .so）请放到 ControlNet/ 目录下。 请不要使用绝对路径（使用相对路径时，根目录为preprocess.sh 所在地址 ）， 因为计分程序会clone 代码到新的地址进行测试。
  - 选手可以通过运行 compute_score.py 来预估运行时间和结果误差情况，其中调用process的参数大部分与计分程序使用的参数相同（除了 random seed 与 prompt），计分程序会使用不同的seed，prompt 来进行测试。最终测试图片大小为256x384。测试图片示例在/home/player/pictures_croped/ 文件夹中。
  - 正式提交前，请验证代码已符合要求：
    - 把 /home/player/ControlNet 作为代码根目录，包含完整代码
    - 运行 /home/player/ControlNet/preprocess.sh，检查 ~/ControlNet下面的 .onnx .plan 和 .so 是否正常生成
    - 运行 /home/player/ControlNet/compute_score.py，检查 pipeline 是否正常生成图片，并确认在标准输出得到PD score
    - 在docker 外面clone 代码， 确保代码可以以一下方式运行
      - docker run --rm -t --network none --gpus '0' --name hackathon -v /repo/anywhere:/repo registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2 bash -c "cd /repo && bash preprocess.sh"
      - docker run --rm -t --network none --gpus '0' --name hackathon -v /repo/anywhere:/repo registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2 bash -c "cd /repo && python3 compute_score.py"
  - 验证通过后提交代码：
    - 在 [https://gitee.com/](https://gitee.com//) 上创建代码仓库，设为私有，并把 xueli1993 加为开发者
    - 借助 git 将自己的代码上传到代码仓库
    - 把仓库的 git 地址填入天池提交页，正式提交
    - 首次提交代码时，请在天池页面点击“提交结果”->“修改地址”，在弹出的窗口中“git路径”中，请写入可用 git clone 命令顺利下载代码的URL
    - 请不要提交大文件 (.onnx .plan. so等) 到git，测试代码时不会使用git-lfs clone代码。
+ 排名依据
  
  - 优化后模型将在评测服务器上 GPU（A10）运行，得分考虑两个方面，即结果精度（PD score，大于12 为精度不合格）和推理时间（end-to-end 耗时）两方面
  - 得分与推理时间负相关，与PD score也负相关
  - 选手可以通过在自己开发机上运行 compute_score.py 来预估运行时间和结果误差情况。但注意实际评测是在评测服务器上完成的，不采用本地开发机上报告的结果
  - 对于成绩相同的选手，按提交时间交早的排名靠前。
  - 天池的在线排行榜并非实时更新，而是每隔一段时间更新，它未考虑运行时间的测量误差以及提交时间早晚，其结果仅供参考，初赛结束时赛方将给出最终排行榜。
