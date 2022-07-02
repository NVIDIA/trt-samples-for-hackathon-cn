import os
import ctypes
from glob import glob
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

soFilePath = './ScalarAdditionPlugin.so'
cacheFile  = './calibration.cache'
calibCount = 10
cIn        = 4
hIn        = 4
wIn        = 5
data       = np.linspace(0,1,cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)
addend     = 0.5                                                                                    # could only change before engine built

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
            data = np.random.rand(np.prod(self.shape)).astype(np.float32)                           # input data range [0,1]
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

def getPluginScalarAddition(addend):
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'ScalarAdditionPlugin':
            p1 = trt.PluginField("addend", np.array([addend],dtype=np.float32), trt.PluginFieldType.FLOAT32)
            fc = trt.PluginFieldCollection([p1,])
            return c.create_plugin(name = 'ScalarAdditionPlugin', field_collection=fc)
    return None

def buildEngine(logger, dataType):
    plugin = getPluginScalarAddition(addend)
    if plugin == None:
        print('Plugin not found')
        return

    builder = trt.Builder(logger)
    builder.max_batch_size = 4
    builder.max_workspace_size = 3 << 30
    builder.fp16_mode = (dataType == 'float16')
    builder.int8_mode = (dataType == 'int8')
    builder.strict_type_constraints = True
    builder.int8_calibrator = MyCalibrator(calibCount, (1,cIn,hIn,wIn), cacheFile)
    network = builder.create_network(1<<0)
    
    inputTensor = network.add_input('input', trt.DataType.FLOAT, (cIn,hIn,wIn))

    pluginLayer = network.add_plugin_v2(inputs = [inputTensor,], plugin = plugin)

    network.mark_output(pluginLayer.get_output(0))
    network.get_output(0).dtype = [trt.DataType.INT8 if dataType == 'int8' else trt.DataType.FLOAT][0]
    network.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)                       # need shuffle if using CHW4 
    return builder.build_cuda_engine(network)
    
def run(dataType):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)            

    trtFile = "./engine-"+dataType+".trt"
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engineStr = f.read()         
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
        if engine == None:
            print("Failed loading engine!")
            return
            print("succeeded loading engine!")
    else:
        engine = buildEngine(logger, dataType)
        if engine == None:
            print("Failed building engine!")
            return None
        print("succeeded building engine!")
        with open(trtFile, 'wb') as f:
            f.write( engine.serialize() )

    context = engine.create_execution_context()
    stream  = cuda.Stream()

    hIn1    = np.ascontiguousarray(data)
    dIn1    = cuda.mem_alloc(hIn1.nbytes)
    hOut1   = np.empty(engine.get_binding_shape(1), dtype = trt.nptype(engine.get_binding_dtype(1)))
    dOut1   = cuda.mem_alloc(hOut1.nbytes)
        
    cuda.memcpy_htod_async(dIn1, hIn1, stream)
    context.execute_async(1, [int(dIn1), int(dOut1)], stream.handle)
    cuda.memcpy_dtoh_async(hOut1, dOut1, stream)
    stream.synchronize()
    
    print("output-> ", hOut1.shape, engine.get_binding_dtype(1))
    print(hOut1)
    print("test " + dataType + " finish!")

if __name__ == '__main__':
    #[ os.remove(item) for item in glob("./*.trt") + glob("./*.cache")]
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    #run('float32')
    #run('float16')
    run('int8')
    print("test finish!")

