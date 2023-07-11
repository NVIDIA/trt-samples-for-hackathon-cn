# 英伟达 TensorRT 加速 AI 推理 Hackathon 2022 —— Transformer 模型优化赛

## 说明

本比赛已经结束，相关资源（模型、数据、代码、讲座视频/PPT 等均已上传至本仓库，见下面介绍中的链接）

## 大赛报名入口

点击[报名入口](https://tianchi.aliyun.com/competition/entrance/531953/information)，注册阿里云账号，报名参赛。

## 大赛简介

深度学习深刻地改变了计算机应用程序的功能与形态，广泛渗透于我们生活。为了加速深度学习模型的推理，英伟达推出了 TensorRT。经过多年的版本迭代，TensorRT  在保持极致性能的同时，大大提高了易用性，已经成为 GPU 上推理计算的必备工具。

随着版本迭代，TensorRT 的编程接口在不断更新，编程最佳实践也在不断演化。开发者想知道，为了把我的模型跑在 TensorRT 上，最省力、最高效的方式是什么？

今天，英伟达联合阿里天池举办 TensorRT Hackathon 就是为了帮助开发者在编程实践中回答这一问题。英伟达抽调了 TensorRT  开发团队和相关技术支持团队的工程师组成专家小组，为开发者服务。参赛的开发者将在专家组的指导下在初赛中对给定模型加速；在复赛中自选模型进行加速，并得到专家组一对一指导。

我们希望借助比赛的形式，提高选手开发 TensorRT 应用的能力，因此重视选手的学习过程以及选手与英伟达专家之间的沟通交流。

## 赛题说明

本赛分初赛和复赛，两轮比赛的详细规则和配置介绍见 HackathonGuide.md

### 初赛

初赛是利用 TensorRT 加速 ASR 模型 WeNet（包含 encoder 和 decoder 两个部分）的推理过程，以优化后的运行时间和结果精度作为排名依据。

- 我们将建立包含所有选手的技术交流群，供大家研讨交流
- 我们为此次比赛准备了系列讲座，供大家观看学习，以便更加顺利地完成比赛
  - 讲座地址：[B站视频](https://www.bilibili.com/video/BV15Y4y1W73E)
  - 讲座 PPT：[讲座PPT](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/cookbook/50-Resource/TensorRT%E6%95%99%E7%A8%8B-TRT8.2.3-V1.1.pdf)
  - 配套范例：[cookbook](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/cookbook)
- 初赛结束时组织一次讲评，介绍优化该模型的技巧
  - 讲评地址：[B站视频](https://www.bilibili.com/video/BV1i3411G7vN)
  - 讲评 PPT：[讲座PPT](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/cookbook/50-Resource/Hackathon2022-%E5%88%9D%E8%B5%9B%E6%80%BB%E7%BB%93-Wenet%E4%BC%98%E5%8C%96-V1.1.pdf)

### 复赛

复赛是开放赛题，各选手可自由选择公开的 Transformer 模型，在 TensorRT 上优化运行，并提交优化工作报告，英伟达将组织专家小组评审你的优化工作并给出最终得分。最终，我们会公开选手们选择的模型及所做的优化工作，供大家相互交流学习。

## 比赛日程表
|时间点|完成事项|建议的下一步活动|
|-|-|-|
|4月2日|报名开放|开始比赛|
|5月20日|初赛提交截止|等待初赛结果公布|
|5月23日|复赛开始：分配导师|选题|
|5月27日|选题审查：选题完毕，github项目主页建立，发布报告的“原始模型”部分|优化模型|
|6月13日|中期审查：模型已经初步可以在TensorRT上运行。获得云主机访问方式|进一步优化模型，完成报告|
|6月27日|最终审查：完成代码和报告。请导师按文档审核、运行代码|等待比赛结果|
|7月6日|公布比赛结果|上网吐槽|

## 获奖团队

### 一等奖

|团队编号|团队名称|项目主页|原始模型|
|-|-|-|-|
|31|美迪康AI Lab|https://github.com/TRT2022/MST-plus-plus-TensorRT|[MST++](https://github.com/caiyuanhao1998/MST-plus-plus)|

### 二等奖

|团队编号|团队名称|项目主页|原始模型|
|-|-|-|-|
|1|ching|https://github.com/dingyuqing05/trt2022_wenet|[WeNet](https://github.com/wenet-e2e/wenet)|
|2|Quark|https://github.com/chenlamei/MobileVit_TensorRT|[MobileViT](https://github.com/wilile26811249/MobileViT)|

### 三等奖

|团队编号|团队名称|项目主页|原始模型|
|-|-|-|-|
|5|错误代码114|https://github.com/YukSing12/trt-hackathon-2022|[AnchorDETR](https://github.com/megvii-research/AnchorDETR)|
|12|试到秃头|https://github.com/huismiling/wenet_trt8|[WeNet](https://github.com/wenet-e2e/wenet)|
|28|小小蜜蜂|https://github.com/tuoeg/bee|[LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3)|

## 优胜奖

|团队编号|团队名称|项目主页|原始模型|
|-|-|-|-|
|3|摇阿摇|https://github.com/jhl13/YAY-TRT-Hackathon-2022|[SwinIR](https://github.com/JingyunLiang/SwinIR)|
|4|Good Luck To You!|https://github.com/Rythsman/TRT-Hackathon-2022-final|[MobileViT](https://github.com/apple/ml-cvnets)|
|6|云上浪|https://github.com/crouchggj/AnchorDETR_TRT_Hackathon|[AnchorDETR](https://github.com/megvii-research/AnchorDETR)|
|8|TRTRush|https://github.com/ustcdane/lite_transformer_trt|[Lite Transformer](https://github.com/mit-han-lab/lite-transformer)|
|9|杭师大的饭真好吃|https://github.com/Luckydog-lhy/Tensorrt_Mask2Former|[Mask2Former](https://github.com/facebookresearch/Mask2Former)|
|10|摸鱼小组|https://github.com/ModelACC/trt2022_levit|[LeViT](https://github.com/facebookresearch/LeViT)|
|11|摸鱼小分队|https://github.com/zhsky2017/TRT-Hackathon-2022-SegFormer|[SegFormer](https://github.com/NVlabs/SegFormer)|
|14|啊对对对队|https://github.com/misaka0316/TRT-for-Swin-Unet|[Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)|
|15|发际线与我作队|https://github.com/wozwdaqian/TensorRT-DAB-DETR|[DAB-DETR](https://github.com/IDEA-opensource/DAB-DETR)|
|16|你不队|https://github.com/hwddean/trt_segformer|[SegFormer](https://github.com/NVlabs/SegFormer)|
|18|暴雨如注|https://github.com/lxl24/SwinTransformerV2_TensorRT|[swin_transformer_v2](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py)|
|20|将个烂就|https://github.com/JustSuckItUp/doublekill|[MobileViT](https://github.com/wilile26811249/MobileViT)|
|21|SmilingFaces|https://github.com/liu-mengyang/trt-elan/|[ELAN](https://github.com/xindongzhang/ELAN)|
|22|好了是我|https://github.com/huangchaosuper/trt2022_final|[trocr](https://github.com/chineseocr/trocr-chinese)|
|27|肉蛋充饥| https://github.com/shuo-ouyang/trt-hackathon-2022|[Uniformer](https://github.com/Sense-X/UniFormer)|
|34|BIT-jedibobo|https://github.com/jedibobo/TRT-Hackathon2022-BIT-jedibobo|[CLIP](https://github.com/openai/CLIP)|
|36|aifeng166|https://github.com/BraveLii/trt-hackathon-swin-transformer|[swin_transformer_v2](https://github.com/microsoft/Swin-Transformer)|
|38|冰河映寒星|https://github.com/zspo/TRT2022_VilBERT|[ViLBERT](https://github.com/jiasenlu/vilbert_beta)|
|39|TensorRT_Tutorial|https://github.com/LitLeo/3m-asr-inference|[3M-ASR](https://github.com/tencent-ailab/3m-asr)|
|40|SUTPC|https://github.com/JQZhai/LeViT-TensorRT|[LeViT](https://github.com/facebookresearch/LeViT)|

### 入围奖

|团队编号|团队名称|项目主页|原始模型|
|-|-|-|-|
|13|edvince|https://github.com/EdVince/espnet-trt|[espnet](https://github.com/espnet/espnet)|
|25|智星云小分队|https://github.com/congyang12345/wenet|[WeNet](https://github.com/wenet-e2e/wenet)|
|26|摆一摆|https://github.com/Zu-X/TRT2022|[CvT](https://github.com/microsoft/CvT)|
