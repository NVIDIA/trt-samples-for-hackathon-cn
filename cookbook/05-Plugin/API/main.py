#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

import ctypes
from cuda import cudart
import numpy as np
import os
import tensorrt as trt
from glob import glob

soFile = "./AddScalarPlugin.so"
np.set_printoptions(precision=3, linewidth=100, suppress=True)
np.random.seed(31193)
cudart.cudaDeviceSynchronize()

def getAddScalarPlugin(scalar):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "AddScalar":
            parameterList = []
            parameterList.append(trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

os.chdir("/w/gitlab/tensorrt-cookbook/05-Plugin/API/")

# Load default plugin creators
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')

pluginRegistry = trt.get_plugin_registry()
print("Count of default plugin creators = %d" % len(pluginRegistry.plugin_creator_list))

# Attributions of Plugin Registry
print("pluginRegistry.error_recorder =", pluginRegistry.error_recorder)  # ErrorRecorder can be set into EngineInspector, usage of ErrorRecorder refer to 09-Advance/ErrorRecorder
pluginRegistry.parent_search_enabled = True  # whether search plugin creators in parent directory, default value is True

# Load local plugin creators
for soFile in glob("./*.so"):
    if True:  # common method
        ctypes.cdll.LoadLibrary(soFile)
    else:  # use TensorRT API, but there are some problems, do not use this temporarily
        handle = pluginRegistry.load_library(soFile)
        #pluginRegistry.deregister_library(handle)  # deregiste the library
print("Count of total plugin creators = %d" % len(pluginRegistry.plugin_creator_list))  # one more plugin creator "AddScalar" added

#pluginRegistry.deregister_library(?)  # deregiste the library

# print information of all plugin creators
print("TensorRTVersion Namespace PluginVersion Name")
for creator in pluginRegistry.plugin_creator_list:
    print("%4s            %s        %s             %s" % (creator.tensorrt_version, 
                                                            ("\"\"" if creator.plugin_namespace == "" else creator.plugin_namespace), 
                                                            creator.plugin_version,
                                                            creator.name))

for creator in pluginRegistry.plugin_creator_list:
    if creator.name == "AddScalar" and creator.plugin_version == "1":  # check name and version during selecting plugin
        
        # print the necessary parameters for creating the plugin
        for i, pluginField in enumerate(creator.field_names):
            print("%2d->%s, %s, %s, %s" % (i, pluginField.name, pluginField.type, pluginField.size, pluginField.data))
        
        # We can registe and deregiste a plugin creator in Plugin Registry, but not required
        #pluginRegistry.deregister_creator(creator)  # deregiste the plugin creator
        #pluginRegistry.register_creator(creator)  # registe the plugin creator again
        
        # feed the PluginCreator with parameters
        pluginFieldCollection = trt.PluginFieldCollection()
        pluginField = trt.PluginField("scalar", np.float32(1.0), trt.PluginFieldType.FLOAT32)
        # tensorrt.PluginFieldType: FLOAT16, FLOAT32, FLOAT64, INT8, INT16, INT32, CHAR, DIMS, UNKNOWN
        print(pluginField.name, pluginField.type, pluginField.size, pluginField.data)
        
        pluginFieldCollection.append(pluginField)  # use like a list
        #pluginFieldCollection.insert(1,pluginField)
        #pluginFieldCollection.extend([pluginField])
        #pluginFieldCollection.clear()
        #pluginFieldCollection.pop(1)
        plugin = creator.create_plugin(creator.name, pluginFieldCollection)  # create a plugin by parameters

        plugin.__class__ = trt.IPluginV2Ext  # change class of plugin from IPluginV2 to IPluginV2Ext, we still do not have IPluginV2Dynamic class
        
        # methods not work in python API
        # plugin.supports_format(trt.float32, None)  # nvinfer1::TensorFormat::kLINEAR
        #plugin.attach_to_context(None, None)  
        #plugin.detach_from_context()
        #plugin.configure_with_format([[2]], [[2]], trt.float32, None, 1)  # nvinfer1::TensorFormat::kLINEAR
        #plugin.configure_plugin([[2]],[[2]],[trt.float32],[trt.float32],[False],[False], None, 1)  # nvinfer1::TensorFormat::kLINEAR
        #plugin.execute_async(1, [None], [None], None, 0)  # address of input / output / workspace memory
        #plugin.initialize()
        #plugin.terminate()
        #plugin.destroy()
        
        # methods work (but useless) in python API
        print("plugin.plugin_type =", plugin.plugin_type)
        print("plugin.plugin_namespace =", plugin.plugin_namespace)
        print("plugin.plugin_version =", plugin.plugin_version)
        print("plugin.num_outputs =", plugin.num_outputs)
        print("plugin.serialization_size =", plugin.serialization_size)
        print("plugin.tensorrt_version =", plugin.tensorrt_version)
        print("plugin.clone() =", plugin.clone())
        print("plugin.get_output_data_type(0, [trt.float32]) =", plugin.get_output_data_type(0, [trt.float32]))
        print("plugin.get_output_shape(0, [trt.Dims([2])])) =", plugin.get_output_shape(0, [trt.Dims([2])]))  # output is always ((0))?
        print("plugin.get_workspace_size(1) =", plugin.get_workspace_size(1))  # output is always 0?
        
        pluginString = plugin.serialize()
        plugin = creator.deserialize_plugin(creator.name, pluginString)  # create a plugin by memory of serialized plugin
    
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

inputT0 = network.add_input("inputT0", trt.float32, [-1])
profile.set_shape(inputT0.name, [1], [2], [4])
config.add_optimization_profile(profile)

pluginLayer = network.add_plugin_v2([inputT0], plugin)
print(pluginLayer.plugin)  # other members and methods refer to 02-API/Layer

print("Finish")

# methods not work
#trt.get_builder_plugin_registry(None)  # nvinfer1::EngineCapability