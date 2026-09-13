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
"""Fetch the `gpt2` checkpoint into `00-Data/model/gpt2/`.

This is the **only** script in the cookbook that downloads a model, and it is deliberately here in
`00-Data` rather than inside the example that needs it
(`03-Workflow/pyTorch-KVCache-ONNX-TensorRT`). An example that pulls half a gigabyte the first time
it runs cannot be run offline, cannot be run reproducibly, and turns a network outage into a
confusing mid-case failure. Downloads are a separate, explicit, one-off step - the same convention
the MNIST dataset and `model-large.onnx` already follow in this directory.

Run it once:

```bash
cd <PathToCookbook>/00-Data
python3 get-model-gpt2.py
```

The download is resumable - re-run the script after an interruption and it picks up where it
stopped. Two environment variables cover the usual network problems:

+ `HF_HUB_DISABLE_XET=1` falls back from HuggingFace's Xet transfer protocol to plain HTTPS. Many
  corporate proxies and mirrors do not implement the Xet endpoints and answer
  `.../xet-read-token/...` with a `404`, which surfaces as a `ConnectionError` a few megabytes in.
  This script detects that failure and retries with the fallback on its own, so you rarely need to
  set it by hand.
+ `HF_ENDPOINT=https://hf-mirror.com` (or any other mirror) redirects the whole download when
  `huggingface.co` itself is unreachable or slow.
"""

import os
import sys
from pathlib import Path

# Only what the example actually opens. The full repository also carries TensorFlow, Flax, Rust and
# ONNX copies of the same weights, which would roughly quadruple the download for nothing.
ALLOW_PATTERN_LIST = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
]

MODEL_ID = "gpt2"
output_path = Path(__file__).parent / "model" / MODEL_ID

# `huggingface_hub` reads its transfer settings at import time, so switching the Xet protocol off
# after the import has no effect. To retry without it we have to re-exec the whole interpreter; this
# marker keeps that from looping forever.
RETRY_MARKER = "TRT_COOKBOOK_GPT2_RETRIED"

if __name__ == "__main__":
    if (output_path / "config.json").exists():
        print(f"{output_path} already exists, nothing to do")
        sys.exit(0)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("This script needs `huggingface_hub` (a dependency of `transformers`):")
        print("    pip install transformers")
        sys.exit(1)

    print(f"Downloading {MODEL_ID} (~528 MB) into {output_path}")
    print(f"Endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")

    try:
        # An interrupted run leaves its partial files behind; `snapshot_download` resumes them, so
        # re-running the script after a network failure does not start from zero.
        snapshot_download(MODEL_ID, local_dir=str(output_path), allow_patterns=ALLOW_PATTERN_LIST)
    except Exception as e:  # ConnectionError, HfHubHTTPError, ... - the hub raises several types
        b_xet_on = os.environ.get("HF_HUB_DISABLE_XET", "") not in ("1", "true", "True")
        if b_xet_on and RETRY_MARKER not in os.environ:
            # The Xet endpoints are commonly missing behind a proxy or on a mirror, and the failure
            # ("404 on .../xet-read-token/...") arrives only after the transfer has started.
            print(f"\nDownload failed: {type(e).__name__}: {e}")
            print("Retrying with HF_HUB_DISABLE_XET=1 (plain HTTPS)\n")
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            os.environ[RETRY_MARKER] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)  # never returns
        print(f"\nDownload failed: {type(e).__name__}: {e}")
        print("Try a mirror, for example:")
        print(f"    HF_ENDPOINT=https://hf-mirror.com python3 {Path(__file__).name}")
        print("Or fetch the files listed in ALLOW_PATTERN_LIST by hand into:")
        print(f"    {output_path}")
        sys.exit(1)

    print("Finish")
