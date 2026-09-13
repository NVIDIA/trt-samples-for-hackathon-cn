#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Install a runnable `tritonserver` + TensorRT backend WITHOUT docker and WITHOUT root.
#
# Why this exists: `tritonserver` is only distributed inside `nvcr.io/nvidia/tritonserver`, the
# PyPI package of the same name is Python bindings only, and building from source pulls a large
# toolchain. But a container image is just tar files behind an HTTP API, and NGC serves this
# repository to anonymous tokens, so the two directories that matter can be unpacked directly.
#
# What it does:
#   1. anonymous token -> manifest list -> amd64 manifest -> image config (prints the TensorRT
#      version the image was built against, which the engine has to match),
#   2. downloads image layers from the last one backwards until it finds the one holding
#      `opt/tritonserver`, then extracts `bin/`, `lib/` and `backends/tensorrt/` only (~55 MB),
#   3. adds the two shared libraries that live outside `/opt/tritonserver` (`libb64.so.0d` from
#      Ubuntu, `libdcgm.so.4` from the CUDA repository), taken out of their `.deb` with `dpkg-deb`.
#
# Requirements: curl, python3, tar, dpkg-deb. No docker, no root, no apt install.
#
# Usage:
#   ./install-tritonserver-without-docker.sh [-o <install-dir>] [-t <image-tag>]
#
# Afterwards:
#   export TRITONSERVER_BIN=<install-dir>/opt/tritonserver/bin/tritonserver
#   export LD_LIBRARY_PATH=<install-dir>/opt/tritonserver/lib:$LD_LIBRARY_PATH
#   python3 main.py

set -euo pipefail

INSTALL_DIR="${PWD}/tritonserver-install"
IMAGE_TAG="26.07-py3"
REPOSITORY="nvidia/tritonserver"

while getopts "o:t:h" option; do
    case $option in
        o) INSTALL_DIR=$(realpath -m "$OPTARG") ;;
        t) IMAGE_TAG="$OPTARG" ;;
        h) sed -n '18,40p' "$0"; exit 0 ;;
        *) exit 1 ;;
    esac
done

WORK_DIR="${INSTALL_DIR}/.download"
mkdir -p "$WORK_DIR"

# The token expires in a couple of minutes, so it is fetched again for every blob rather than once.
get_token() {
    curl -s "https://nvcr.io/proxy_auth?scope=repository:${REPOSITORY}:pull&service=registry" |
        python3 -c "import sys, json; print(json.load(sys.stdin)['token'])"
}

fetch_blob() {  # <digest> <output file>
    curl -sL -H "Authorization: Bearer $(get_token)" -o "$2" "https://nvcr.io/v2/${REPOSITORY}/blobs/$1"
}

echo "==== 1/3 Resolving ${REPOSITORY}:${IMAGE_TAG}"
ACCEPT="application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.oci.image.index.v1+json"
curl -s -H "Authorization: Bearer $(get_token)" -H "Accept: ${ACCEPT}" \
    "https://nvcr.io/v2/${REPOSITORY}/manifests/${IMAGE_TAG}" > "${WORK_DIR}/manifest-list.json"

AMD64_DIGEST=$(python3 -c "
import json
manifest = json.load(open('${WORK_DIR}/manifest-list.json'))
if 'manifests' in manifest:  # A manifest list, pick the amd64 entry
    print(next(m['digest'] for m in manifest['manifests'] if m['platform']['architecture'] == 'amd64'))
else:  # Already a single-platform manifest
    print('')
")
if [ -n "$AMD64_DIGEST" ]; then
    curl -s -H "Authorization: Bearer $(get_token)" -H "Accept: ${ACCEPT}" \
        "https://nvcr.io/v2/${REPOSITORY}/manifests/${AMD64_DIGEST}" > "${WORK_DIR}/manifest.json"
else
    cp "${WORK_DIR}/manifest-list.json" "${WORK_DIR}/manifest.json"
fi

CONFIG_DIGEST=$(python3 -c "import json; print(json.load(open('${WORK_DIR}/manifest.json'))['config']['digest'])")
fetch_blob "$CONFIG_DIGEST" "${WORK_DIR}/config.json"
python3 -c "
import json
config = json.load(open('${WORK_DIR}/config.json'))['config']
wanted = ('TRT_VERSION', 'CUDA_VERSION', 'CUDNN_VERSION', 'TRITON_SERVER_VERSION', 'NVIDIA_TRITON_SERVER_VERSION')
for item in config.get('Env', []):
    if item.split('=')[0] in wanted:
        print('    ' + item)
"
echo "    ^ the engine served later must be built with THIS TensorRT version"

echo "==== 2/3 Extracting opt/tritonserver from the image layers"
python3 -c "
import json
for layer in json.load(open('${WORK_DIR}/manifest.json'))['layers']:
    print(layer['digest'], layer['size'])
" > "${WORK_DIR}/layers.txt"

N_LAYER=$(wc -l < "${WORK_DIR}/layers.txt")
echo "    ${N_LAYER} layers, scanning from the last one backwards"
FOUND=""
for n in $(seq "$N_LAYER" -1 1); do
    DIGEST=$(sed -n "${n}p" "${WORK_DIR}/layers.txt" | cut -d' ' -f1)
    SIZE=$(sed -n "${n}p" "${WORK_DIR}/layers.txt" | cut -d' ' -f2)
    [ "$SIZE" -lt 1000000 ] && continue  # Metadata-only layers cannot hold the server
    echo "    layer ${n} ($((SIZE / 1000000)) MB) ..."
    fetch_blob "$DIGEST" "${WORK_DIR}/layer.tar.gz"
    # The listing goes to a file rather than into `grep -q`: under `set -o pipefail`, `grep -q`
    # closing the pipe early makes `tar` die of SIGPIPE and the whole pipeline report 141, so the
    # layer that *does* contain the server would be reported as a miss.
    tar -tzf "${WORK_DIR}/layer.tar.gz" > "${WORK_DIR}/listing.txt" 2>/dev/null || true
    if grep -q "^opt/tritonserver/bin/tritonserver$" "${WORK_DIR}/listing.txt"; then
        echo "    found in layer ${n}, extracting"
        tar -xzf "${WORK_DIR}/layer.tar.gz" -C "$INSTALL_DIR" \
            opt/tritonserver/bin opt/tritonserver/lib opt/tritonserver/backends/tensorrt
        FOUND="yes"
        rm -f "${WORK_DIR}/layer.tar.gz"
        break
    fi
    rm -f "${WORK_DIR}/layer.tar.gz" "${WORK_DIR}/listing.txt"
done
[ -n "$FOUND" ] || { echo "Failed to find opt/tritonserver in any layer"; exit 1; }

echo "==== 3/3 Adding the shared libraries that live outside /opt/tritonserver"
# Both are hard dependencies of the binary, so the server will not even start without them:
#   libb64.so.0d  - base64 helper, Ubuntu universe
#   libdcgm.so.4  - GPU telemetry, needed even with --allow-metrics=false because it is linked in
DEB_DIR="${WORK_DIR}/deb"
mkdir -p "$DEB_DIR"
curl -sL -o "${DEB_DIR}/libb64.deb" "http://archive.ubuntu.com/ubuntu/pool/universe/libb/libb64/libb64-0d_1.2-4_amd64.deb"
DCGM_DEB=$(curl -s "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/" |
    grep -oE "datacenter-gpu-manager-4-core_[0-9.]+-1_amd64\.deb" | sort -V | tail -1)
echo "    ${DCGM_DEB}"
curl -sL -o "${DEB_DIR}/dcgm.deb" "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/${DCGM_DEB}"
for package in "${DEB_DIR}"/*.deb; do
    dpkg-deb -x "$package" "${DEB_DIR}/extracted"
done
# The parentheses matter: without them `-exec` binds to the last `-name` only, so `libb64.so.0d`
# is found and then not copied, and the server fails to start with that one library missing.
find "${DEB_DIR}/extracted" \( -name "libb64.so*" -o -name "libdcgm.so*" \) -exec cp -a {} "${INSTALL_DIR}/opt/tritonserver/lib/" \;

rm -rf "$WORK_DIR"

echo
echo "==== Done: $(du -sh "${INSTALL_DIR}" | cut -f1) in ${INSTALL_DIR}"
MISSING=$(LD_LIBRARY_PATH="${INSTALL_DIR}/opt/tritonserver/lib" ldd "${INSTALL_DIR}/opt/tritonserver/bin/tritonserver" | grep "not found" || true)
if [ -n "$MISSING" ]; then
    echo "Still missing shared libraries:"
    echo "$MISSING"
    exit 1
fi
echo "All shared libraries resolve. Use it with:"
echo "    export TRITONSERVER_BIN=${INSTALL_DIR}/opt/tritonserver/bin/tritonserver"
echo "    export LD_LIBRARY_PATH=${INSTALL_DIR}/opt/tritonserver/lib:\$LD_LIBRARY_PATH"
echo "    python3 main.py"
