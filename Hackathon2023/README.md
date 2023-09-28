# NVIDIA TensorRT Hackathon 2023 —— 生成式AI模型优化赛

## 大赛简介

TensorRT 作为 NVIDIA 英伟达 GPU 上的 AI 推理加速库，在业界得到了广泛应用与部署。与此同时，TensorRT 开发团队也在持续提高产品的好用性：一方面让更多模型能顺利通过 ONNX 自动解析得到加速，另一方面对常见模型结构（如 MHA）的计算进行深度优化。这使得大部分模型不用经过手工优化，就能在 TensorRT 上跑起来，而且性能优秀。

过去的一年，是生成式 AI（或称“AI生成内容”） 井喷的一年。大量的图像和文本被计算机批量生产出来，有的甚至能媲美专业创作者的画工与文采。可以期待，未来会有更多的生成式AI模型大放异彩。在本届比赛中，我们选择生成式AI模型作为本次大赛的主题。

今年的 TensorRT Hackathon 是本系列的第三届比赛。跟往届一样，我们希望借助比赛的形式，提高选手开发 TensorRT 应用的能力，因此重视选手的学习过程以及选手与 NVIDIA 英伟达专家之间的沟通交流。我们期待选手们经过这场比赛，在 TensorRT 编程相关的知识和技能上有所收获。

## 赛题说明

本赛分初赛和复赛，两轮比赛的详细规则和配置介绍见[初赛规则](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/Hackathon2023/preliminary_round.md)，[复赛规则](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/Hackathon2023/final_round.md)。

## 比赛日程表
|时间点|完成事项|建议的下一步活动|
|-|-|-|
|7月11日|报名开放|开始比赛|
|8月14日|初赛提交截止|等待初赛结果公布|
|8月17日|复赛开始：分配导师|选题|
|8月24日|选题审查：选题完毕，github项目主页建立，发布报告的“原始模型”部分|优化模型|
|9月07日|中期审查：代码完成过半（成功导出 TensorRT engine 或 已经掌握了TensorRT-LLM 使用方法，并搭建了部分模型）。|进一步优化模型，完成报告|
|9月21日|最终审查：完成代码和报告。请导师按文档审核、运行代码|等待比赛结果|
|9月29日|公布比赛结果|上网吐槽|

## 获奖团队

### 一等奖 (特别贡献奖)

|团队编号|团队名称|项目主页|原始模型|
|-|-|-|-|
|9|无声优化者（着）|https://github.com/Tlntin/Qwen-7B-Chat-TensorRT-LLM|Qwen-7B|

### 二等奖

|团队编号|团队名称|项目主页|原始模型|
|-|-|-|-|
|5|蹦极不拴绳|https://github.com/FeiGeChuanShu/trt2023|Qwen-7B|
|4|NaN-emm|https://github.com/yuanjiechen/trt_final|minigpt4|

### 三等奖

|团队编号|团队名称|项目主页|原始模型|
|-|-|-|-|
|2|xixixixixixi|https://github.com/xiatwhu/trt-hackathon-2023|gpt2-medium|
|36|iie|https://gitee.com/atlantik/iie-tensor-rt.git|Qwen-7B|
|8|你的耳朵长不长？|https://github.com/VOIDMalkuth/trt_hackathon_2023_final|Qwen-7B|

## 优胜奖

|团队编号|团队名称|项目主页|原始模型|
|-|-|-|-|
|1|野路子|https://github.com/chinasvt/trt-llm2|UniDiffuser|
|22|EddieWang|https://github.com/Eddie-Wang1120/Eddie-Wang-Hackthon2023|Whisper|
|3|美迪康-北航AI Lab|https://github.com/TRT2022/trtllm-llama|LLaMa|
|31|tokernel|https://github.com/WeiboXu/Hackathon2023|mpt-7B-65K|
|26|萱草花1201|https://github.com/jedibobo/trt2023-final-jedibobo|Galactica|
|25|HUST-1037|https://github.com/heptagonhust/NVIDIA_TensorRT_Hackathon_2023_Rematch|LLaMA-2 7B|
|39|哎呦喂|https://github.com/EdVince/whisper-trtllm|Whisper|
|35|404 not found|https://github.com/zuocebianpingmao/tensorrt_llm_july|Qwen-7B|
|37|学无止尽|https://gitee.com/chenmingwei53/trt2023_qwen7-b|Qwen-7B|
|12|Imagination|https://github.com/col-in-coding/TRT-Hackathon-2023-Final|SAM|
|19|有分就算成功|https://github.com/Ricky846/TrtLLM|DeciCoder-1b|
|15|开卷！|https://github.com/L-lowliet/tensorrt_llm_july-release-v1|cpm-bee|
|24|朵拉三人行|https://github.com/bsdcfp/trt2023-final|Aquila-7B|
|29|GPUKiller|https://github.com/pzhao-eng/hackatnon_final|camel-5b|
|7|ouys|https://github.com/shuo-ouyang/CPM-Bee-TRTLLM|CPM-Bee|
|10|我为祖国献石油|https://github.com/FrankyTang/tensorrt_llm_july-release-v1|UniAD|
|13|2023|https://github.com/19706/Hackathon2023_v2|WizardCoder-15B-V1.0|
|11|学习为主|https://github.com/Xuweijia-buaa/trt2023-final|viscpm|
|38|AMD yes!|https://github.com/misaka0316/Qwen-7B-chat-for-TRT-LLM|Qwen-7B-chat|
|16|挚文|https://gitee.com/jianpeng2000/trt2023_zhiwen_final|Qwen-7B|

### 入围奖
|团队编号|团队名称|项目主页|原始模型|
|-|-|-|-|
|34|队伍名字不能为空|https://github.com/chasingw/tensorrt_llm_trt_hackathon_2023|Qwen-7B|
|28|玩游戏一定要笑|https://github.com/smile2game/nvidia-rematch|llama|
|18|麦克阿瑟的战术核显卡|https://github.com/ccw1996/qwen-trt|通义千问|
|27|银河飞车2077|https://github.com/wuzy361/CoDeF_TRT|CoDeF|
|20|力学胡同|https://github.com/big91987/trt-llm-hackathon-2023|ziya-visual|
