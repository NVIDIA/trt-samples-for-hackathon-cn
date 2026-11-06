#!/usr/bin/env bash
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
# Install everything the cookbook needs, using the download caches under `dependence_packages/`.
#
#   ./setup_environment.sh
#
# The container is reset between sessions and loses every installed package, while `/work` survives.
# That is the whole reason the caches live in the repository rather than in `~/.cache`: they are
# there so the reinstall does not re-download half a gigabyte each time.
#
# They save download time and nothing else. The script assumes the network is reachable, so a cache
# miss is slow, never fatal, and there is no "refresh" step to remember - both caches fill
# themselves on first use and are consulted automatically afterwards.
#
#   dependence_packages/pip  pip's own download cache (`PIP_CACHE_DIR`). Consulted by *any*
#                            `pip install`, including ad-hoc ones such as `tensorrt_rtx` that are
#                            not in `requirements.txt` at all.
#   dependence_packages/apt  `.deb` files, which apt has no equivalent cache for here. Downloaded
#                            once, then reinstalled with `dpkg` after every container reset.

set -o errexit
set -o nounset
set -o pipefail

COOKBOOK_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_PATH="${COOKBOOK_PATH}/dependence_packages"
APT_CACHE="${CACHE_PATH}/apt"
PIP_CACHE="${CACHE_PATH}/pip"
export PIP_CACHE_DIR="${PIP_CACHE}"
APT_PACKAGE_LIST=(graphviz git-lfs)

mkdir -p "${APT_CACHE}" "${PIP_CACHE}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- apt packages
say "apt packages: ${APT_PACKAGE_LIST[*]}"

if ! ls "${APT_CACHE}"/*.deb > /dev/null 2>&1; then
    # `apt-get download` does not need root but does not resolve dependencies either, so ask apt for
    # the full URI list of the packages and everything they pull in. `--reinstall` makes apt list
    # packages that are already installed here, which is the whole point: they will not be installed
    # in the next container.
    echo "Downloading the .deb files (first run on this machine)"
    (
        cd "${APT_CACHE}"
        apt-get install --print-uris -y --reinstall "${APT_PACKAGE_LIST[@]}" \
            | grep -oP "(?<=^')[^']+" \
            | while read -r url; do
                  wget --quiet --show-progress "${url}" -O "$(basename "${url}")"
              done
    )
fi

echo "Installing $(ls "${APT_CACHE}"/*.deb 2> /dev/null | wc -l) .deb from ${APT_CACHE}"
# `dpkg -i` needs root. Nothing else in this script does, so the failure is made survivable: these
# two packages are optional for most of the cookbook.
if ! sudo dpkg --install --skip-same-version "${APT_CACHE}"/*.deb 2>&1 | tail -5; then
    echo "  [SKIP] could not install the .deb files (needs root). Run by hand:"
    echo "         sudo dpkg -i ${APT_CACHE}/*.deb"
fi

# ---------------------------------------------------------------- pip packages
say "pip packages (download cache: ${PIP_CACHE})"
python3 -m pip install -r "${COOKBOOK_PATH}/requirements.txt"

# ---------------------------------------------------------------- the cookbook itself
say "tensorrt_cookbook (editable)"
# `--no-build-isolation` reuses the setuptools already installed instead of fetching a fresh copy
# into a throwaway build environment on every run.
python3 -m pip install --no-build-isolation -e "${COOKBOOK_PATH}"

# ---------------------------------------------------------------- report
say "Checking"
python3 - <<'PYEOF'
import importlib.util
import shutil

module_list = [
    ("tensorrt_cookbook", "the cookbook package"),
    ("tensorrt", "TensorRT"),
    ("torch", "pyTorch"),
    ("onnx_graphsurgeon", "onnx-graphsurgeon"),
    ("onnxruntime", "ONNX Runtime"),
    ("polygraphy", "Polygraphy"),
    ("cupy", "CuPy"),
    ("transformers", "transformers"),
    ("graphviz", "graphviz (python binding)"),
]
n_missing = 0
for name, description in module_list:
    b_ok = importlib.util.find_spec(name) is not None
    n_missing += not b_ok
    print(f"  [{'OK  ' if b_ok else 'MISS'}] {description}")
for binary, description in [("dot", "graphviz `dot` binary (apt)"), ("git-lfs", "git-lfs (apt)")]:
    b_ok = shutil.which(binary) is not None
    n_missing += not b_ok
    print(f"  [{'OK  ' if b_ok else 'MISS'}] {description}")
print(f"\n  {n_missing} missing" if n_missing else "\n  Everything is present")
PYEOF
