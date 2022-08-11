#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
from parseTrtexecLog import parse_build_log, parse_profiling_log
import subprocess
from trex import EnginePlan, layer_type_formatter, to_dot, render_dot

onnxFile = "./model/model.onnx"
trtFile = "./model/model.plan"
# build
buildLogfile = "./model/build.log"
buildMetadataJsonFile = "./model/build.metadata.json"
buildTimingCacheFile = "./model/build.timingCache.cache"
# profile
profileLogFile = "./model/profile.log"
profileJsonFile = "./model/profile.json"
profileMetadatadJsonFile = "./model/profile.metadata.json"
profileTimingJsonFile = "./model/profile.timing.json"
# draw
graphJsonFile = "./model/graph.json"

# build engine -----------------------------------------------------------------
cmd_line = "trtexec --verbose --profilingVerbosity=detailed --buildOnly --workspace=4096 --onnx=%s --saveEngine=%s --timingCacheFile=%s" % \
    (onnxFile,trtFile,buildTimingCacheFile)
with open(buildLogfile, "w") as f:
    log = subprocess.run(cmd_line.split(" "), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    f.write(log.stdout)

with open(buildMetadataJsonFile, "w") as f:
    json.dump(parse_build_log(buildLogfile), f)

print("\nSucceeded building engine\n\t%s\n\t%s\n\t%s" % (buildLogfile, buildTimingCacheFile, buildMetadataJsonFile))

# profile engine ---------------------------------------------------------------
cmd_line = "trtexec --verbose --profilingVerbosity=detailed --noDataTransfers --useCudaGraph --separateProfileRun --useSpinWait --loadEngine=%s --exportProfile=%s --exportTimes=%s --exportLayerInfo=%s" % \
    (trtFile, profileJsonFile, profileTimingJsonFile, graphJsonFile)

with open(profileLogFile, "w") as f:
    log = subprocess.run(cmd_line.split(" "), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    f.write(log.stdout)

with open(profileMetadatadJsonFile, "w") as f:
    json.dump(parse_profiling_log(profileLogFile), f)

print("\nSucceeded profiling engine\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s" % (profileLogFile, profileJsonFile, profileTimingJsonFile, graphJsonFile, profileMetadatadJsonFile))

# draw graph -------------------------------------------------------------------
plan = EnginePlan(graphJsonFile)
formatter = layer_type_formatter
graph = to_dot(plan, formatter, display_regions=True, expand_layer_details=False)
render_dot(graph, graphJsonFile, "svg")
print("\nSucceeded drawing graph\n\t%s" % graphJsonFile)
