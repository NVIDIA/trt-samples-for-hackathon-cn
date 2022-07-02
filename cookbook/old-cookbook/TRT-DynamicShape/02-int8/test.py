import numpy as np
import os
from glob import glob
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

cacheFile    = "./calibration.cache"
calibCount   = 10
np.random.seed(97)

class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibCount, inputShape, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)            
        self.calibCount = calibCount
        self.shape      = inputShape                                                                # (N,C,H,W)
        self.cacheFile  = cacheFile        
        self.dIn        = cuda.mem_alloc(trt.volume(self.shape) * trt.DataType.FLOAT.itemsize)          
        self.oneBatch   = self.batchGenerator()
            
    def batchGenerator(self):
        for i in range(self.calibCount):
            print("> calibration ", i, self.shape)
            data = np.random.rand(np.prod(self.shape)).astype(np.float32)
            yield np.ascontiguousarray(data)
    
    def get_batch_size(self):                                                                       # do NOT change name
        return self.shape[0]

    def get_batch(self, nameList=[], inputNodeName = ['input']):                                    # do NOT change name
        try:
            data = next(self.oneBatch)                        
            cuda.memcpy_htod(self.dIn, data)
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):                                                               # do NOT change name
        if os.path.exists(self.cacheFile):
            print( "cahce file: %s" %(self.cacheFile) )
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache              
            
    def write_calibration_cache(self, cache):                                                       # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)

def buildEngine(logger, cIn, cOut, haveCache):
    builder = trt.Builder(logger)
    builder.max_batch_size = 4
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    if haveCache:   # dynamic shape network
        inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (-1, cIn, -1, -1))
        #inputTensor.set_dynamic_range(0.0,1.0)                                                     # calibrate manually
    else:           # static shape network    
        inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (1, cIn, 4, 5))

    wW     = 3
    window = np.ones(cIn * cOut * wW * wW, dtype=np.float32)
    bias   = np.zeros(cOut, dtype=np.float32)
    conv = network.add_convolution(inputTensor, cOut, (wW, wW), window, bias)
    conv.padding = (1,1)
    #conv.get_input(0).set_dynamic_range(0.0,1.0)                                                   # calibrate manually
    #conv.get_output(0).set_dynamic_range(0.0,27.0)
    print("convInput->", conv.get_input(0).shape)
    print("convOutput->", conv.get_input(0).shape)
    
    network.mark_output(conv.get_output(0))
    network.get_output(0).dtype = trt.DataType.INT8
    network.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)

    if haveCache:   # dynamic shape network
        profile = builder.create_optimization_profile()
        profile.set_shape(inputTensor.name, (1,cIn,1,1),(1,cIn,4,5),(4,cIn,9,12))
        config = builder.create_builder_config()
        config.add_optimization_profile(profile)
        config.max_workspace_size = 1 << 30    
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = MyCalibrator(calibCount, (1,cIn,4,5), cacheFile)
        return builder.build_engine(network, config)

    else:           # static shape network    
        builder.max_workspace_size = 1 << 30
        builder.int8_mode = True
        builder.strict_type_constraints = True
        builder.int8_calibrator = MyCalibrator(calibCount, (1,cIn,4,5), cacheFile)
        return builder.build_cuda_engine(network)

def run(dimIn, cOut):
    print("test->", dimIn)
    
    logger = trt.Logger(trt.Logger.INFO)
    trtFile = "./engine-int8-"+str(dimIn[1])+"-"+str(cOut)+".trt"
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engineStr = f.read() 
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
        if engine == None:
            print("Failed loading engine!")
            return
        print("succeeded loading engine!")
    else:
        if not os.path.isfile(cacheFile):                                                           # create calibration cache with static shape in the first time
            engine = buildEngine(logger, dimIn[1], cOut, False)
            if engine == None:
                print("Failed building cache!")
                return
            print("succeeded building cache!")
        engine = buildEngine(logger, dimIn[1], cOut, True)                                          # use dynamic shape when the calibration cache exist
        if engine == None:
            print("Failed building engine!")
            return
        print("succeeded building engine!")
        engineStr = engine.serialize()
        with open(trtFile, 'wb') as f:
            f.write(engineStr)
    
    context = engine.create_execution_context()
    context.set_binding_shape(0,dimIn)
    stream  = cuda.Stream()
    print("EngineBinding0->", engine.get_binding_shape(0))
    print("EngineBinding1->",engine.get_binding_shape(1))
    print("ContextBinding0->",context.get_binding_shape(0))
    print("ContextBinding1->",context.get_binding_shape(1))

    nIn       = dimIn[0] * dimIn[1] * dimIn[2] * dimIn[3]
    data      = np.ones(nIn, dtype=np.float32).reshape(dimIn)
    input1_h  = np.ascontiguousarray(data)
    input1_d  = cuda.mem_alloc(input1_h.nbytes)
    output1_h = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    output1_d = cuda.mem_alloc(output1_h.nbytes)
        
    cuda.memcpy_htod_async(input1_d, input1_h, stream)
    context.execute_async(dimIn[0], [int(input1_d), int(output1_d)], stream.handle)
    cuda.memcpy_dtoh_async(output1_h, output1_d, stream)
    stream.synchronize()
    
    print(output1_h.shape, engine.get_binding_dtype(1))
    print(output1_h)
    
if __name__ == '__main__':
    #[ os.remove(item) for item in glob("./*.trt") + glob("./*.cache")]
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run((1,3,4,5),2)
    run((1,3,1,1),2)
    run((4,3,9,12),2)

