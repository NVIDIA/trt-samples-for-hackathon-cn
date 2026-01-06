FROM nvcr.io/nvidia/pytorch:25.12-py3

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
