#/bin/bash

set -xeuo pipefail

clear && nvidia-smi

#docker system prune -af &

docker build -t tensorrt-cookbook:wili -f tensorrt-cookbook.Dockerfile .

docker run \
    -it \
    --gpus $(nvidia-smi -L | wc -l) \
    --shm-size 32G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name qwen3-vl-wili \
    -v /home/scratch.wili_sw_1/:/sc \
    -v /home/scratch.wili_sw_1/work:/work \
    -v /home/scratch.trt_llm_data/llm-models:/llm-models \
    -v /home/scratch.trt_llm_data:/scratch.trt_llm_data \
    --user $(id -u):$(id -g) \
    tensorrt-cookbook:wili \
    /bin/bash
