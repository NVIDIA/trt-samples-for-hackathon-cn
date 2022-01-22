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

import os
import ctypes
from glob import glob
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

nLength     = 10
nEmbed      = 128
maskValue   = [-9e9,-6e4]
npToNumber  = {np.float32:0,          np.float16:1,          np.int32:3}
npToTrt     = {np.float32:trt.float32,np.float16:trt.float16,np.int32:trt.int32}
soFilePath  = './LodPreGruPlugin.so'
np.random.seed(97)

def preGruCPU(inputList,outDatatype):  
    [sequence0,sequence1,sequence2,sequence3,sequence4,sequence5,sequence6,
     lod0,lod2,lod4,lod6,width0,width2,width4,width6,embedBook] = inputList
    nGroup = lod0.shape[0] - 1
    [nWidth0,nWidth2,nWidth4,nWidth6] = [len(width0),len(width2),len(width4),len(width6)]
    
    out0 = np.zeros((nGroup,nWidth0,nEmbed),dtype=np.float32)    
    out2 = np.zeros((nGroup,nWidth0,nEmbed),dtype=np.float32)
    out5 = np.zeros((nGroup,nWidth0,nEmbed),dtype=np.float32)
    
    for r in range(nGroup):
        lodL = lod0[r]
        lodR = lod0[r+1]
        nValidWidth = lodR-lodL
        for c in range(min(nWidth0,nValidWidth)):
            index0 = sequence0[lodL+c]
            index1 = sequence1[lodL+c]
            value0 = embedBook[index0,:]
            value1 = embedBook[index1,:]           
            out0[r,c,:] = value0
            out2[r,c,:] = value0 + value1
            out5[r,nValidWidth-1-c,:] = value0 + value1            
                
    out3 = np.zeros((nGroup,nWidth2,nEmbed),dtype=np.float32)
    out6 = np.zeros((nGroup,nWidth2,nEmbed),dtype=np.float32)
    for r in range(nGroup):
        lodL = lod2[r]
        lodR = lod2[r+1]
        nValidWidth = lodR-lodL
        for c in range(min(nWidth2,nValidWidth)):    
            index0 = sequence2[lodL+c]
            index1 = sequence3[lodL+c]
            value0 = embedBook[index0,:]
            value1 = embedBook[index1,:]           
            out3[r,c,:] = value0 + value1
            out6[r,nValidWidth-1-c,:] = value0 + value1                
                
    out4 = np.zeros((nGroup,nWidth4,nEmbed),dtype=np.float32)
    for r in range(nGroup):
        lodL = lod4[r]
        lodR = lod4[r+1]
        nValidWidth = lodR-lodL
        for c in range(min(nWidth4,nValidWidth)):
            index0 = sequence4[lodL+c]
            index1 = sequence5[lodL+c]
            value0 = embedBook[index0,:]
            value1 = embedBook[index1,:]           
            out4[r,c,:] = value0 + value1

    out1 = np.zeros((nGroup,nWidth6,nEmbed),dtype=np.float32)
    for r in range(nGroup):
        lodL = lod6[r]
        lodR = lod6[r+1]
        nValidWidth = lodR-lodL
        for c in range(min(nWidth6,nValidWidth)):            
            index = sequence6[lodL+c]
            out1[r,c,:] = embedBook[index,:]
                        
    out7    = np.zeros([nGroup,nWidth0,1],dtype=np.float32)
    out8    = np.zeros([nGroup,nWidth2,1],dtype=np.float32)
    out9    = np.zeros([nGroup,nWidth4,1],dtype=np.float32)
    out10   = np.zeros([nGroup,nWidth6,1],dtype=np.float32)
    out11   = np.full([nGroup,nWidth0,1],maskValue[int(outDatatype==np.float16)],dtype=np.float32)
    out12   = np.full([nGroup,nWidth2,1],maskValue[int(outDatatype==np.float16)],dtype=np.float32)
    out13   = np.full([nGroup,nWidth4,1],maskValue[int(outDatatype==np.float16)],dtype=np.float32)
    for i in range(nGroup):
        out7[i,:(lod0[i+1]-lod0[i]),:] = 1
        out8[i,:(lod2[i+1]-lod2[i]),:] = 1
        out9[i,:(lod4[i+1]-lod4[i]),:] = 1
        out10[i,:(lod6[i+1]-lod6[i]),:] = 1
        out11[i,:(lod0[i+1]-lod0[i]),:] = 0    
        out12[i,:(lod2[i+1]-lod2[i]),:] = 0
        out13[i,:(lod4[i+1]-lod4[i]),:] = 0

    out14 = np.diff(lod0)
    out15 = np.diff(lod2)
    out16 = np.diff(lod4)
    out17 = np.diff(lod6)
    return [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11,out12,out13,out14,out15,out16,out17]

def cleanTrash(outputH0,inputH1):  # clean the trash data in the output of GPU
    nValidWidth = np.diff(inputH1)
    for i in range(outputH0.shape[0]):
        outputH0[i,nValidWidth[i]:,:] = 0
    return outputH0

def getLodPreGruPlugin(datatype):
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "LodPreGruPlugin":
            p0 = trt.PluginField("datatype", np.array([datatype],dtype=np.int32), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0]))
    return None

def buildEngine(logger,outDatatype):
    builder                     = trt.Builder(logger)
    network                     = builder.create_network(1)
    profile                     = builder.create_optimization_profile()
    config                      = builder.create_builder_config()
    config.max_workspace_size   = 1 << 30
    config.flags                = int(outDatatype==np.float16)

    inputTL = [ None for i in range(16) ]
    for i in range(7):
        inputTL[i] = network.add_input('inputT'+str(i), trt.int32, [-1])
        profile.set_shape(inputTL[i].name, [1],[2000],[4000])
        
    for i in range(7,11):
        inputTL[i] = network.add_input('lod'+str(i*2-14), trt.int32, [-1])
        profile.set_shape(inputTL[i].name, [1+1],[20+1],[40+1])
        
    for i in range(11,15):
        inputTL[i] = network.add_input('width'+str(i*2-22), trt.int32, [-1])
        profile.set_shape(inputTL[i].name, [1],[50],[100])
        
    inputTL[15] = network.add_input('book', npToTrt[outDatatype], [nLength,nEmbed])
    
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTL, getLodPreGruPlugin(npToNumber[outDatatype]))

    for i in range(pluginLayer.num_outputs):
        network.mark_output(pluginLayer.get_output(i))
    
    return builder.build_engine(network, config)

def run(nGroup,widthList,outDatatype):
    print("test", nGroup, widthList, outDatatype)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    engine = buildEngine(logger,outDatatype)
    if engine == None:
        print("Failed building engine!")
        return None
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    #[ context.set_binding_shape(i, [(widthList[i]*2-nGroup+1)*nGroup//2]) for i in range(7) ]
    [ context.set_binding_shape(i, [nGroup+1]) for i in range(7,11) ]
    [ context.set_binding_shape(i, [widthList[i-11]]) for i in range(11,15) ]
    context.set_binding_shape(15,[nLength,nEmbed])    
    
    if outDatatype == np.float32:
        mesh0 = np.tile(np.arange(nEmbed,dtype=np.float32),[nLength,1])
        mesh1 = np.tile(np.arange(nLength,dtype=np.float32),[nEmbed,1]).transpose()
        book  = np.array(mesh1 + mesh0 / 1000,dtype=outDatatype)
    else:
        book  = np.random.randint(0,255,[nLength,nEmbed],dtype=np.int32).astype(np.float16)
    
    inputH0List = []
    inputD0List = []
    inputH1List = []
    inputD1List = []
    inputH2List = []
    inputD2List = []
    for i in range(7):
        lod             = np.arange(widthList[i//2]-nGroup+1,widthList[i//2]+1,dtype=np.int32)
        lod[lod<1]      = 1
        width           = np.zeros(np.max(lod))
        data0           = np.random.randint(0,nLength,np.sum(lod),dtype=np.int32)        
        lod             = np.array([0] + np.cumsum(lod).tolist(),dtype=np.int32)
        inputH0List.append(data0)
        inputD0List.append( cuda.mem_alloc(inputH0List[-1].nbytes) )
        if i%2 == 0:        
            inputH1List.append(lod)
            inputD1List.append( cuda.mem_alloc(inputH1List[-1].nbytes) )
            inputH2List.append(width)
            inputD2List.append( cuda.mem_alloc(inputH2List[-1].nbytes) )
        context.set_binding_shape(i,[lod[-1]])

    inputH2List.append(book)
    inputD2List.append( cuda.mem_alloc(inputH2List[-1].nbytes) )

    #[ print("Bind"+str(i)+"->",engine.get_binding_shape(i),context.get_binding_shape(i)) for i in range(16+14) ]
    print("All bind:",context.all_binding_shapes_specified)
    stream  = cuda.Stream()
    
    outputH0List = []
    outputD0List = []
    for i in range(18):
        outputH0List.append( np.empty(context.get_binding_shape(16+i), dtype = trt.nptype(engine.get_binding_dtype(16+i))) )
        outputD0List.append( cuda.mem_alloc(outputH0List[-1].nbytes) )
    
    [ cuda.memcpy_htod_async(inputD0List[i], np.ascontiguousarray(inputH0List[i]), stream) for i in range(7) ]
    [ cuda.memcpy_htod_async(inputD1List[i], np.ascontiguousarray(inputH1List[i]), stream) for i in range(4) ]
    [ cuda.memcpy_htod_async(inputD2List[i], np.ascontiguousarray(inputH2List[i]), stream) for i in range(5) ]
    
    context.execute_async_v2([ *[int(i) for i in inputD0List], 
                               *[int(i) for i in inputD1List], 
                               *[int(i) for i in inputD2List], 
                               *[int(i) for i in outputD0List] ], stream.handle)
    
    for i in range(18):
        cuda.memcpy_dtoh_async(outputH0List[i], outputD0List[i], stream)

    stream.synchronize()
    
    outputH0CPUList = preGruCPU([*inputH0List,*inputH1List,*inputH2List],outDatatype)
    
    outputH0List[0] = cleanTrash(outputH0List[0], inputH1List[0])
    outputH0List[1] = cleanTrash(outputH0List[1], inputH1List[3])
    outputH0List[2] = cleanTrash(outputH0List[2], inputH1List[0])
    outputH0List[3] = cleanTrash(outputH0List[3], inputH1List[1])
    outputH0List[4] = cleanTrash(outputH0List[4], inputH1List[2])
    outputH0List[5] = cleanTrash(outputH0List[5], inputH1List[0])
    outputH0List[6] = cleanTrash(outputH0List[6], inputH1List[1])
    
    for i in range(18):
        print("Check result",i,":", "True" if np.all( outputH0List[i] == outputH0CPUList[i] ) else "False")
    '''            
    for i in range(14):
        print("output"+str(i),context.get_binding_shape(i),engine.get_binding_dtype(i))
        print("Device")
        print(outputH0List[i][:,:,0])        
        print("Host")
        print(outputH0CPUList[i][:,:,0])                
    
    for i in range(14,18):
        print("output"+str(i),context.get_binding_shape(16+i),engine.get_binding_dtype(16+i))
        print("Device")
        print(outputH0List[i])
        print("Host")
        print(outputH0CPUList[i])
    '''
        
if __name__ == '__main__':
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    np.set_printoptions(threshold=1e6)
    cuda.Device(0).make_context()
    
    run(4,[2,4,8,16],np.float32)
    run(4,[2,4,8,16],np.float16)
    run(40,np.random.randint(2,100,40,dtype=np.int32),np.float32)
    run(40,np.random.randint(2,100,40,dtype=np.int32),np.float16)

    cuda.Context.pop()
    print("test finish!")

