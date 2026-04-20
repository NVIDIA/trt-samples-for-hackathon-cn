#/bin/bash

set -xeuo pipefail

clear && nvidia-smi

# docker system prune -af &

VERSION="${1:-26.03}"

docker run \
    -it \
    --gpus $(nvidia-smi -L | wc -l) \
    --shm-size 32G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name tensorrt-cookbook-${VERSION} \
    -v /home/wili/work:/work \
    -v /home/wili:/wili \
    nvcr.io/nvidia/pytorch:${VERSION}-py3 \
    /bin/bash
