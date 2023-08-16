# 关于复赛与TensorRT-LLM
大语言模型是计算机行业未来的重要方向，英伟达希望借助复赛的机会，加强开发团队与开发者的交流，让开发者快速上手英伟达即将正式推出的大语言模型推理加速库TensorRT-LLM，并能在未来的工作中熟练运用。

TensorRT-LLM是对TensorRT的再封装。它改善了TensorRT模型的手工搭建方式，引入了plugin提高推理性能，并加入了大量新功能。  
  + 虽然取的名字提到LLM（Large Language Model，大语言模型），TensorRT-LLM可以用来搭建任意AI模型。
  + TensorRT-LLM现在没有ONNX parser，所以不能走ONNX workflow，必须手工搭建模型。
  + 大家拿到的TensorRT-LLM只是一个非公开的预览版本。在使用过程中可能会遇到一些问题，比如没有支持完善的feature。英伟达正在开发完善它。

英伟达非常感谢选手们的参与和投入。我们深知比赛的奖金有限，与选手们的付出不成比例，但是会精心组织赛程与评选，努力让比赛更有意义。

# 复赛赛题
+ 请选手自选大型语言模型，并使用 TensorRT 或 TensorRT-LLM 进行模型推理优化。我们鼓励大家使用TensorRT-LLM，使用TensorRT-LLM 更有得分优势（详细见“评分标准”）。 
+ 每队将获得一台云主机（包含一张 NVIDIA A10 GPU）用于比赛。因此，请选择能够在一张 A10 上跑起来的模型。
+ 为了让选手的工作可被他人参考，**您的代码需要开源**，开源协议可以视需要自行选择。

## 需要提交的内容 
要求选手提交的内容主要为代码与报告。
### 要提交的代码 
+ 选手要提交的代码需要在我们给定的docker中顺利运行。我们已经在云主机中预装了docker镜像：
    - 镜像名为: registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final
    - TensorRT 9.0 EA 安装目录为: /usr/local/TensorRT-9.0.0.2
    - TensorRT-LLM 代码目录为 /root/tensorrt_llm_july-release-v1/
####  如果使用TensorRT-LLM
  TensorRT-LLM 在docker中已经构建并且安装好了。TensorRT-LLM 中已经实现了一些模型（见 /root/tensorrt_llm_july-release-v1/examples/）。请选手仔细阅读代码中包含的 README 文件，直接进入examples目录跑各个现有模型（一些模型需要的.pth 与python 包还仍然要按照模型folder下面的README 进行下载与安装）。在选手理解TensorRT-LLM的用法之后，进行开发工作。  
  请选手把新开发的代码都放置于TensorRT-LLM代码目录树下，并且将代码上传到自己的代码repo上，并**在比赛结束前维持repo的个人可见状态**。请注意，**先commit原始TensorRT-LLM代码**，然后再做进一步开发，以方便比较代码更改。如果安装新的库与依赖，请写好安装脚本，并在报告中说明。
#### 如果使用 TensorRT 优化
  如果使用TensorRT，请选手提供从原始模型出发，直到运行优化好的 TensorRT 模型，全过程所需的所有脚本及代码到代码repo上面，将其开源出来。
### 要提交的报告
  - 要用固定的模板、以 markdown 的形式发布在代码仓库根目录的 README.md 里面。
  - 报告的模板在本文的末尾

## 评分标准
+ TensorRT-LLM 试用送分题：为了鼓励选手试用TensorRT-LLM，无论选手选择 TensorRT 还是 TensorRT-LLM 来做优化，完成送分题都能得分
  - 请在报告中写出 /root/tensorrt_llm_july-release-v1/examples/gpt/README 里面 “Single node, single GPU” 部分如下命令的输出（10分）
    - python3 run.py --max_output_len=8
  - 请在报告中写出 /root/tensorrt_llm_july-release-v1/examples/gpt/README 里面 “Summarization using the GPT model” 部分如下命令的rouge 分数（10分）
    - python3 summarize.py --engine_dirtrt_engine/gpt2/fp16/1-gpu --test_hf  --batch_size1  --test_trt_llm  --hf_model_location=gpt2 --check_accuracy --tensorrt_llm_rouge1_threshold=14

+ 主要得分：
  - 选题得分：请从以下题目选择1个来做，也可以在选2或3的同时叠加4。选TensorRT和选TensorRT-LLM的评分区别仅在选题上。前者最高分为30，后者最高分为100（如果选了2+4或3+4）。
    1. 用TensorRT优化模型（30分）
    2. 用TensorRT-LLM实现新模型（50分）。满足以下条件可给满分：
         - 仿照examples的代码组织形式，完成模型搭建，并可顺利输出文本（实现weight.py/build.py/run.py）
         - 通过摘要任务，可评估原始模型与加速模型的rouge score（实现summarize.py）
         - 为模型写好README.md
         - 与现有模型相比，新模型有一定难度（比如有特殊的结构或算子）
    3. 用TensorRT-LLM优化examples目录下的某个现有模型（50分）。满足以下条件可给满分：
         - 在保持精度的同时，显著提高了性能
         - 为算子提供了更高效率的实现
    4. 为TensorRT-LLM添加新feature，或者在模型上启用了现有feature（50分）  
      这里的feature是指为了更有效率地运行模型，在模型上采用的通用工具。比如现有feature包括int8 KV-cache，inflight batching，SmoothQuant等（这些feature可能在某些现有模型上不能正常工作）。你可以添加新的feature，比如新的量化方法，新的sampling方法等，并在现有模型或新模型中实现。视实现难度与工作量给分。  
      例如，以下为英伟达正在进行的feature开发工作，计划在9月发布：  
         - 在GPT-NEOX和LLaMA上实现GPTQ
         - 在Bloom上实现SmoothQuant  

  - 代码得分：代码干净，逻辑清晰（30分）
  - 报告得分：报告完整，可读性好，对 TensorRT 或 TensorRT-LLM 学习者有良好的参考价值（30分）

+ 附加得分
  - 独立开发了 Plugin 或开发 CUDA 代码（10分）
    - Plugin开发可使用 [OpenAI Triton](https://github.com/openai/triton)。如需在 TensorRT-LLM 中使用，请参考 TensorRT-LLM docker 中 /root/tensorrt_llm_july-release-v1/examples/openai_triton。
  - 用Nsight Systems/Nsight Compute进行了Profiling，并进行了针对性优化（10分）
  - 提交与开发过程相关的bug，并得到导师确认。提交多个bug不重复加分。（10分）

+ 初赛得分
  - 初赛原始得分除以100取整计入复赛。

+ 复赛结束后，英伟达将组织至少7名专家，基于上述标准对每个复赛选手评分，取所有专家的平均分作为该选手的最终成绩。

## 优化报告的模板

大赛要求用统一模板写报告，是为了让所有报告都有共同的行文结构，方便评审。同时，本模板也尽量考虑实用性，让它可以称职地成为代码项目的主页说明书。
我们希望同学们用心打造这份报告。但愿这份报告就像一份TensorRT入门教程那样，通过一个具体的例子，详细介绍从原始模型到优化模型的全工作流程，从而传授经验，给人启发。

以下为模板具体内容。

---

### 总述

请简练地概括项目的主要贡献，使读者可以快速理解并复现你的工作，包括：

- 本工作的选题（应为如下之一：1，2，3，4，2+4，3+4）
    - 如果是优化新模型，原始模型的名称及链接，并对该模型做个简要介绍
- 优化效果（例如给出精度和加速比），简单给出关键的数字即可，在这里不必详细展开
- 在Docker里面代码编译、运行步骤的完整说明
  - 请做到只要逐行运行你给的命令，就能把代码跑起来

### 主要开发工作

#### 开发工作的难点

请在这一节里总结你的工作难点与亮点。
- 如果使用 TensorRT 进行优化，请介绍一下在模型在导出时、或用polygraphy/trtexec解析时，或在使用TensorRT中，遇到了什么问题并解决了。换句话说，针对这个模型，我们为什么需要额外的工程手段。
- 如果使用 TensorRT-LLM 进行优化，描述以下方面可供选手参考：如果搭建了新模型， 请介绍模型结构有无特别之处，在模型的搭建过程中使用了什么算子，有没有通过plugin支持的新算子。如果支持新feature，请介绍这个feature具体需要修改哪些模块才能实现。如果优化已有模型，请介绍模型性能瓶颈以及解决方法。另外还可以包含工程实现以及debug过程中的难点。

### 开发与优化过程

这一部分是报告的主体。请把自己假定为老师，为 TensorRT 或 TensorRT-LLM 的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的 TensorRT 或 TensorRT-LLM 模型。或者你是如何一步步通过修改哪些模块添加了新feature的。

建议：

- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。

### 优化效果

这一部分介绍你的工作在云主机上的运行效果。如果是优化模型，需要分两部分说明：

- 精度：报告与原始模型进行精度对比测试的结果，验证精度达标。
  - 如果选用TensorRT-LLM，请跑summarize任务并使用 [Rouge](https://huggingface.co/spaces/evaluate-metric/rouge) 来对比模型优化前后的精度差距。如果精度良好，原始模型与优化模型的Rouge score的差异一般在1以内。例子见 TensorRT-LLM docker 中 /root/tensorrt_llm_july-release-v1/examples/gpt/summarize.py
  - 如果选用TensorRT，这里的精度测试指的是针对“原始模型”和“TensorRT优化模型”分别输出的数据（tensor）进行数值比较。请给出绝对误差和相对误差的统计结果（至少包括最大值、平均值与中位数）。
    - 使用训练好的权重和有意义的输入数据更有说服力。如果选手使用了随机权重和输入数据，请在这里注明。
    - 在精度损失较大的情况下，鼓励选手用训练好的权重和测试数据集对模型优化前与优化后的准确度指标做全面比较，以增强说服力。
- 性能：例如可以用图表展示不同batch size或sequence length下性能加速效果（考虑到可能模型可能比较大，可以只给batch size为1的数据）
  - 一般用原始模型作为baseline
  - 一般提供模型推理时间的加速比即可；若能提供压力测试下的吞吐提升则更好。

请注意：

- 相关测试代码也需要包含在代码仓库中，可被复现。
- 请写明云主机的软件硬件环境，方便他人参考。

### Bug报告（可选）

提交bug是对TensorRT/TensorRT-LLM的另一种贡献。发现的TensorRT/TensorRT-LLM或cookbook、或文档和教程相关bug，请提交到[github issues](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues)，并请在这里给出链接。  

对于每个bug，请标记上hackathon2023标签，并写好正文：

- 对于cookbook或文档和教程相关bug，说清楚问题即可，不必很详细。
- 对于TensorRT bug，首先确认在云主机上使用NGC docker + TensorRT 9.0.0.1可复现。
- 然后填写如下模板，并请导师复核确认（前面“评分标准”已经提到，确认有效可得附加分）：
  - Environment
    - TensorRT 9.0.0.1
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

### 送分题答案（可选）

如果你做了送分题，请把答案写在这里。

### 经验与体会（可选）

欢迎在这里总结经验，抒发感慨。
