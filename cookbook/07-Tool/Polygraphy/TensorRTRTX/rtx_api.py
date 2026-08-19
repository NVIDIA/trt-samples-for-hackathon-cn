# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
"""The half of the TensorRT-RTX story that only the Python API shows.

Run me twice, with POLYGRAPHY_USE_TENSORRT_RTX unset and set to 1.
"""
import os
import sys

def main() -> None:
    # 1. The switch is read once, at import time. Setting it afterwards does nothing at all.
    before = os.environ.get("POLYGRAPHY_USE_TENSORRT_RTX", "<unset>")
    import polygraphy
    from polygraphy import config as polygraphy_config
    os.environ["POLYGRAPHY_USE_TENSORRT_RTX"] = "1"  # deliberately too late
    print(f"    polygraphy {polygraphy.__version__}")
    print(f"    POLYGRAPHY_USE_TENSORRT_RTX at import : {before}")
    print(f"    os.environ set to 1 after the import  : config.USE_TENSORRT_RTX = {polygraphy_config.USE_TENSORRT_RTX}")

    from polygraphy.mod.trt_importer import lazy_import_trt
    trt = lazy_import_trt()
    print(f"    module polygraphy resolved            : {trt.__name__} {trt.__version__}")

    # 2. Precision flags. The CLI accepts and drops them; the API refuses them.
    from polygraphy.backend.trt import CreateConfig
    from polygraphy.logger import G_LOGGER
    for keyword in ["fp16", "int8", "bf16", "fp8"]:
        with G_LOGGER.verbosity(G_LOGGER.CRITICAL):  # CreateConfig chats at INFO
            try:
                CreateConfig(**{keyword: True})
                verdict = "accepted"
            except Exception as exception:  # PolygraphyException
                verdict = f"{type(exception).__name__}: {str(exception).splitlines()[0][:74]}"
        print(f"    {f'CreateConfig({keyword}=True)':<24} -> {verdict}")
    return

if __name__ == "__main__":
    main()
    print("    " + "-" * 60)
    sys.exit(0)
