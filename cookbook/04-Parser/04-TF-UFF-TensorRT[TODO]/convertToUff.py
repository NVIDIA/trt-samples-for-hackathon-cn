#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import uff

inputFile  = "./model.pb"
outputFile = '.'+inputFile.split('.')[1]+'.uff'

uff.from_tensorflow_frozen_model(inputFile,
    output_nodes=['y'],
    output_filename=outputFile,
    preprocessor=None,
    write_preprocessed=False,    
    text=False,
    quiet=False,
    debug_mode=False,
    #input_node=['x'],    
    #list_nodes=False,
    return_graph_info=False
)
