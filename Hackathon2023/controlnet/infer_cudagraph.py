from cuda.bindings import runtime as cudart


class cudagraph_engine():

    def __init__(self, engine, context, tensor_list, stream):
        # _, self.stream = cudart.cudaStreamCreate()
        self.stream = stream
        for i in range(len(tensor_list)):
            name = engine.get_binding_name(i)
            context.set_tensor_address(name, tensor_list[i].data_ptr())
        context.execute_async_v3(self.stream)
        # cudart.cudaStreamSynchronize(stream)
        cudart.cudaStreamBeginCapture(
            self.stream,
            cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        context.execute_async_v3(self.stream)
        _, graph = cudart.cudaStreamEndCapture(self.stream)
        _, self.instance = cudart.cudaGraphInstantiate(graph, 0)

    def infer(self):
        cudart.cudaGraphLaunch(self.instance, self.stream)
        # cudart.cudaStreamSynchronize(self.stream)
