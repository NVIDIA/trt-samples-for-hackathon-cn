# 配置开发环境

## 安装最新驱动

请根据自己的 GPU 型号，从 NVIDIA 官网下载并安装最新驱动。截至2022年3月，最新的驱动版本是510+。

## 安装nvidia-docker

为了在 docker 中正常使用GPU，请安装 nvidia-docker。

- 如果你的系统是Ubuntu Linux
  - 请参考 [Installing Docker and The Docker Utility Engine for NVIDIA GPUs](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html) 安装nvidia-docker
- 如果你的系统是Windows 11
  - 请先参考 [Install Ubuntu on WSL2 on Windows 11 with GUI support](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview) 把WSL设置好
  - 然后参考 [Running Existing GPU Accelerated Containers on WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#ch05-running-containers) 安装nvidia-docker

## 下载并运行大赛专用镜像

### 下载镜像

```nvidia-docker pull registry.cn-hangzhou.aliyuncs.com/trt2022/dev```

### 运行

新建目录，作为源代码目录。这里的位置和命名仅供举例用，实际上你可以自己决定。
```mkdir ~/trt2022_src```

创建并启动容器，取名trt2022，并把源代码目录挂载到容器的/target
```nvidia-docker run -it --name trt2022 -v ~/trt2022_src:/target registry.cn-hangzhou.aliyuncs.com/trt2022/dev```

将来退出这个容器之后，你仍然可以用上面给出的名字trt2022再把它启动起来，就像这样
```nvidia-docker start -i trt2022```

启动起来docker之后，你可以从/workspace找到需要你加速的模型encoder.onnx和decoder.onnx。

### 开发程序

前面创建的~/trt2022文件夹有两重身份：它既在你物理机的文件系统里，也在docker镜像里。所以你可以在物理机上对文件夹新建、编辑源文件，又可以在docker里面读取并运行里面的程序。

如果还对接下来怎么做充满疑惑，可以先看一看教学视频充充电。祝你好运！

# 初赛补充说明

+ 初赛仅提供测评服务器不提供开发机，参赛选手需要自备带有 GPU 的 Linux / Windows 11 (WSL2) 开发机，并在给定 docker 中用赛方提供的模型文件、开发工具完成模型在 TensorRT 中的构建、运行、精度验证和性能测试，并将代码提交至指定仓库以供测评服务器打分排名。
+ 初赛使用的镜像：`registry.cn-hangzhou.aliyuncs.com/trt2022/dev`
  
  - 该镜像基于英伟达官方镜像扩充而来，包含 CUDA 11.6，TensorRT 8.2.2 以及比赛用到的开发工具、模型文件、测试样例数据。请根据"配置开发环境"部分的说明进行使用。
  - /workspace 含有比赛相关文件，默认只读，请勿修改
  - /workspace/encoder.onnx 和 /workspace/decoder.onnx 是 pyTorch 中训练好的 WeNet 模型的 encoder、decoder 两部分转换 ONNX 格式而成。选手的目标就是把它们转成优化后的 TensorRT engine 序列化文件（.plan）
  - encoder 相对容易，请优先完成
  - 对于 decoder，为了简化起见，请将输入张量 hyps_pad_sos_eos 的末维固定为 64，即在 TensorRT 中构建 engine 时，将该张量形状固为 [-1,10,64]，否则不能完成评测
+ 代码验证与提交
  
  - 请保证在 docker 里能正常运行你的代码，并且无论编译时还是运行时，都不依赖网络下载任何代码或数据，即代码需要是完整的、自包含的。如果确实需要在 docker 里面新增开发库或软件，请在交流群里反应给赛方，我们将添加到比赛用的 docker image 中
  - 在代码根目录下，请创建`build.sh`，并保证运行该`build.sh`时，在代码根目录下生成 encoder.plan 和 decoder.plan；如果有 plugin，在代码根目录下生成所有 .so
  - 正式提交前，请验证代码已符合要求：
    - 把 /target 作为代码根目录，把干净代码拷贝过去
    - 运行 /workspace/buildFromWorkspace.sh，检查 /target下面的 .plan 和 .so 是否正常生成
    - 运行 /workspace/testEncoderAndDecoder.py，检查 TRT engine 是否正常运行，并确认在标准输出得到评测表格
  - 验证通过后提交代码：
    - 在 [code.aliyun.com](https://code.aliyun.com) 上创建代码仓库，设为私有，并把 wili-Nvidia 加为 reporter
    - 注意：不要使用新版的`codeup.aliyun.com`
    - 借助 git 将自己的代码上传到代码仓库
    - 把仓库的 git 地址填入天池提交页，正式提交
    - 首次提交代码时，请在天池页面点击“提交结果”->“修改地址”，在弹出的窗口中“git路径”中，请写入可用 git clone 命令顺利下载代码的URL，比如 https://code.aliyun.com/your_name/your_project.git
+ 排名依据
  
  - 优化后模型将在评测服务器上 GPU（A30）运行，得分考虑两个方面，即结果精度（TensorRT 推理输出相较原始 pyTorch 推理输出的误差）和推理时间（end-to-end 耗时）两方面
  - 在误差不超过阈值（见 testEncoderAndDecoder.py）的条件下，得分与推理计算时间成反比；若误差超出阈值，则得分将在推理计算时间得分的基础上进行罚分，误差越大罚分越大。
  - 选手可以通过在自己开发机上运行 testEncoderAndDecoder.py 来预估运行时间和结果误差情况。但注意实际评测是在评测服务器上完成的，不采用本地开发机上报告的结果
  - 对于成绩相同的选手，按提交时间交早的排名靠前。
  - 天池的在线排行榜并非实时更新，而是每隔一段时间更新，它未考虑运行时间的测量误差以及提交时间早晚，其结果仅供参考，初赛结束时赛方将给出最终排行榜。

# 复赛补充说明

+ 对于参赛者，初赛的目的是通过优化同一个模型，让大家互相交流，了解 TensorRT 的一般用法，积累经验；复赛的目的是让大家把自己感兴趣的、认为有价值的、有一定难度的模型找出来，在英伟达专家的支持下把它优化好。
+ 对于英伟达，初赛的目的就是选拔；复赛的目的是扩大 TensorRT 在业界的应用规模，并通过积累大家创作的网上资源，引导更多人把 TensorRT 用好。
+ 复赛要自选题目，这样就失去了制订统一的客观评分的基础。不得不说，对于英伟达，复赛的评分公平性是个巨大的挑战。英伟达小心翼翼地制订评分规则，认真评审，可结果也许做不到让所有人都满意。对此，我们希望大家都放平心态，积极参赛，在复赛中收获知识与经验。我们会努力让所有优秀的作品都得到充分认可，即便没有足够的物质奖励分发给大家。

## 复赛赛题

+ 为了让选手的工作可被他人参考，**必须选择公开的模型**，而且**选手的代码也必须开源**。开源协议可以视需要自行选择。
+ 要求选手交付的内容如下：
  - 代码：从原始模型出发，直到运行优化好的 TensorRT 模型，全过程所需的所有脚本及代码，并且需要选手在公共的代码托管平台（建议用 github）将其开源出来。特别地，要求能在选手自选的某个英伟达官方 Docker 镜像（在[NGC](https://catalog.ngc.nvidia.com/containers)里面选择）中运行。
    - 使用 NGC 是为了给他人提供标准的复现环境。即便开发过程中没有使用 Docker，最终提交前也需要在 NGC Docker 中验证，并据此写报告。
    - NGC 里有多种镜像可选，有的预装了 TensorFlow，PyTorch 或 TensorRT。选手要从中选一个，然后在里面自行安装其他软件（python 库安装建议写个 requirement.txt）。
    - 选手一般需要从原始模型导出 ONNX，有的还要做 QAT，此过程可能涉及对原始模型的修改。选手既可以拷贝原模型代码到自己的代码仓库，也可以指明原模型代码位置及版本，并附上修改补丁。建议采用前者。
  - 报告：要用固定的模板、以 markdown 的形式发布在代码仓库根目录的 README.md 里面。
  - 报告的模板在本文的末尾

## 评分标准

英伟达专家小组对优化报告进评审的主要评分依据如下：

+ 主要得分
  - 模型本身有应用价值，暂无成熟优秀的TensorRT上的优化方案，有技术上的优化难度。20分
  - 代码干净，逻辑清晰。30分
  - 模型顺利运行，精度合格，加速良好。30分
  - 报告完整，可读性好，对TensorRT学习者有良好的参考价值。30分
+ 附加得分
  - 独立开发了CUDA代码或Plugin。20分
  - 用Nsight进行了Profiling，并进行了针对性优化。20分
  - 进行了INT8量化的工作（QAT/PTQ均可），在保证精度可接受的前提下进一步提升了性能。20分
  - 提交TensorRT bug，并得到导师确认。5分每个bug。
+ 初赛得分
  - 折算30分计入复赛。

## 优化报告的模板

大赛要求用统一模板写报告，是为了让所有报告都有共同的行文结构，方便评审。同时，本模板也尽量考虑实用性，让它可以称职地成为代码项目的主页说明书。
我们希望同学们用心打造这份报告。但愿这份报告就像一份TensorRT入门教程那样，通过一个具体的例子，详细介绍从原始模型到优化模型的全工作流程，从而传授经验，给人启发。

以下为模板具体内容。

---

### 总述

请简练地概括项目的主要贡献，使读者可以快速理解并复现你的工作，包括：

- 原始模型的名称及链接
- 优化效果（精度和加速比），简单给出关键的数字即可，在这里不必详细展开
- 在Docker里面代码编译、运行步骤的完整说明
  - 请做到只要逐行运行你给的命令，就能把代码跑起来，比如从docker pull开始

### 原始模型

#### 模型简介

请介绍模型的基本信息，可以包含但不限于以下内容：

- 用途以及效果
- 业界实际运用情况，比如哪些厂商、哪些产品在用
- 模型的整体结构，尤其是有特色的部分

#### 模型优化的难点

如果模型可以容易地跑在TensorRT上而且性能很好，就没有必要选它作为参赛题目并在这里长篇大论了。相信你选择了某个模型作为参赛题目必然有选择它的理由。
请介绍一下在模型在导出时、或用polygraphy/trtexec解析时、或在TensorRT运行时，会遇到什么问题。换句话说，针对这个模型，我们为什么需要额外的工程手段。

### 优化过程

这一部分是报告的主体。请把自己假定为老师，为TensorRT的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的TensorRT模型。

建议：

- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。

### 精度与加速效果

这一部分介绍优化模型在云主机上的运行效果，需要分两部分说明：

- 精度：报告与原始模型进行精度对比测试的结果，验证精度达标。
  - 这里的精度测试指的是针对“原始模型”和“TensorRT优化模型”分别输出的数据（tensor）进行数值比较。请给出绝对误差和相对误差的统计结果（至少包括最大值、平均值与中位数）。
  - 使用训练好的权重和有意义的输入数据更有说服力。如果选手使用了随机权重和输入数据，请在这里注明。
  - 在精度损失较大的情况下，鼓励选手用训练好的权重和测试数据集对模型优化前与优化后的准确度指标做全面比较，以增强说服力
- 性能：最好用图表展示不同batch size或sequence length下性能加速效果。
  - 一般用原始模型作为参考标准；若额外使用ONNX Runtime作为参考标准则更好。
  - 一般提供模型推理时间的加速比即可；若能提供压力测试下的吞吐提升则更好。

请注意：

- 相关测试代码也需要包含在代码仓库中，可被复现。
- 请写明云主机的软件硬件环境，方便他人参考。

### Bug报告（可选）

提交bug是对TensorRT的另一种贡献。发现的TensorRT、或cookbook、或文档和教程相关bug，请提交到[github issues](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues)，并请在这里给出链接。

对于每个bug，请标记上hackathon2022标签，并写好正文：

- 对于cookbook或文档和教程相关bug，说清楚问题即可，不必很详细。
- 对于TensorRT bug，首先确认在云主机上使用NGC docker + TensorRT 8.4 GA仍可复现，然后填写如下模板，并请导师复核确认（前面“评分标准”已经提到，确认有效可得附加分）：
  - Environment
    - TensorRT 8.4 GA
    - Versions of CUDA, CUBLAS, CuDNN used
    - Container used
    - NVIDIA driver version
  - Reproduction Steps
    - Provide detailed reproduction steps for the issue here, including any commands run on the command line.
  - Expected Behavior
    - Provide a brief summary of the expected behavior of the software. Provide output files or examples if possible.
  - Actual Behavior
    - Describe the actual behavior of the software and how it deviates from the expected behavior. Provide output files or examples if possible.
  - Additional Notes
    - Provide any additional context here you think might be useful for the TensorRT team to help debug this issue (such as experiments done, potential things to investigate).

### 经验与体会（可选）

欢迎在这里总结经验，抒发感慨。


