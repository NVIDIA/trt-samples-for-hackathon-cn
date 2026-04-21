#/bin/bash

set -xeuo pipefail

clear && nvidia-smi

# docker system prune -af &

VERSION="${1:-26.03}"

docker build -t tensorrt-cookbook:${VERSION} -f - . <<EOF
FROM nvcr.io/nvidia/pytorch:${VERSION}-py3

RUN apt-get update && \
    apt-get install -y sudo passwd && \
    addgroup --gid 31193 wiligroup && \
    adduser --gecos GECOS -u 43427 -gid 31193 wili && \
    echo "wili:cuan" | chpasswd && \
    adduser wili sudo && \
    usermod -a -G wiligroup wili && \
    usermod -a -G wiligroup root && \
    usermod -a -G root wili && \
    echo 'wili ALL=(ALL) ALL' >> /etc/sudoers

USER wili

# Specify for this repo
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
EOF

docker run \
    -it \
    --gpus $(nvidia-smi -L | wc -l) \
    --shm-size 32G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name tensorrt-cookbook-${VERSION} \
    -v /home/scratch.wili_sw_1/work:/work \
    -v /home/scratch.wili_sw_1/:/sc \
    --user $(id -u):$(id -g) \
    tensorrt-cookbook:${VERSION} \
    /bin/bash
