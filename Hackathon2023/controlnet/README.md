# Introduction

This is a [ControlNet 1.0](https://github.com/lllyasviel/ControlNet) canny2image pipeline optimized by NVIDIA TensorRT.

# Setup
## Environment
```registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2```
## run
Export torch models and convert them to TensorRT engine.

```bash preprocess.sh```

Generate images.

```python3 compute_score.py```
