from collections import OrderedDict
from copy import deepcopy
import numpy as np
import onnx
import onnx_graphsurgeon as gs

onnxFilePath = "/workspace/"
#onnxFilePath = "./" # local host
sourceOnnx = "./decoderV2.onnx"
destinationOnnx = "./decoderV3.onnx"

bConvertToStaticNetwork = False
bDebug = False

nWili = 0
bNot = False
nNot = 0
bLayerNormPlugin = True
nLayerNormPlugin = 0
bShapeOperation = True
nShapeOperation = 0
b2DMM = True
n2DMM = 0

#graph = gs.import_onnx(onnx.load(sourceOnnx))
graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))

if bConvertToStaticNetwork:
    graph.inputs[0].shape = [3, 17, 256]
    graph.inputs[1].shape = [3]
    graph.inputs[2].shape = [3, 10, 64]
    graph.inputs[3].shape = [3, 10]
    graph.inputs[4].shape = [3, 10]
else:
    graph.inputs[0].shape = ['B', 'T', 256]
    graph.inputs[1].shape = ['B']
    graph.inputs[2].shape = ['B', 10, 64]
    #graph.inputs[2].shape = ['B',10,'T2']

# Round 0: ceate useful constant tensor or collect useful shape tensor
wiliConstant0 = gs.Constant("wiliConstant0", np.ascontiguousarray(np.array([0], dtype=np.int64)))  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!
wiliConstant1 = gs.Constant("wiliConstant1", np.ascontiguousarray(np.array([1], dtype=np.int64)))
wiliConstant10 = gs.Constant("wiliConstant10", np.ascontiguousarray(np.array([10], dtype=np.int64)))
wiliConstant63 = gs.Constant("wiliConstant63", np.ascontiguousarray(np.array([63], dtype=np.int64)))
wiliConstant64 = gs.Constant("wiliConstant64", np.ascontiguousarray(np.array([64], dtype=np.int64)))
wiliConstant256 = gs.Constant("wiliConstant256", np.ascontiguousarray(np.array([256], dtype=np.int64)))
wiliConstant4233 = gs.Constant("wiliConstant4233", np.ascontiguousarray(np.array([4233], dtype=np.int64)))

wiliConstantS0 = gs.Constant("wiliConstantS0", np.array(0, dtype=np.int64)).to_variable(np.dtype(np.int64), []).to_constant(np.array(0, dtype=np.dtype(np.int64)))
wiliConstantS1 = gs.Constant("wiliConstantS1", np.array(1, dtype=np.int64)).to_variable(np.dtype(np.int64), []).to_constant(np.array(1, dtype=np.dtype(np.int64)))
wiliConstantS63 = gs.Constant("wiliConstantS63", np.array(63, dtype=np.int64)).to_variable(np.dtype(np.int64), []).to_constant(np.array(63, dtype=np.dtype(np.int64)))
wiliConstantRange63 = gs.Constant("wiliConstantRange63", np.ascontiguousarray(np.arange(63, dtype=np.int64).reshape(1, 63)))

data = np.ones([1, 63, 63], dtype=np.int32)
for i in range(63):
    data[0, i, :(i + 1)] = 0
wiliConstant1x63x63 = gs.Constant("wiliConstant1x63x63", np.ascontiguousarray(data.astype(bool)))

wiliShapeV = gs.Variable("wiliShapeV-" + str(nWili), np.dtype(np.int64), [3])
wiliShapeN = gs.Node("Shape", "wiliShapeN-" + str(nWili), inputs=[graph.inputs[0]], outputs=[wiliShapeV])
graph.nodes.append(wiliShapeN)
nWili += 1

# shape = [], value = ['B']
bTensorScalar = gs.Variable("bTensorScalar", np.dtype(np.int64), [])
wiliGatherN = gs.Node("Gather", "wiliGatherN-" + str(nWili), inputs=[wiliShapeV, wiliConstantS0], outputs=[bTensorScalar], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliGatherN)
nWili += 1

# shape = [1,], value = ['B']
bTensor = gs.Variable("bTensor", np.dtype(np.int64), [1])
wiliUnsqueezeN = gs.Node("Unsqueeze", "wiliUnsqueezeN-" + str(nWili), inputs=[bTensorScalar, wiliConstant0], outputs=[bTensor])
graph.nodes.append(wiliUnsqueezeN)
nWili += 1

# shape = [], value = ['T']
tTensorScalar = gs.Variable("tTensorScalar", np.dtype(np.int64), [])
wiliGatherN = gs.Node("Gather", "wiliGatherN-" + str(nWili), inputs=[wiliShapeV, wiliConstantS1], outputs=[tTensorScalar], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliGatherN)
nWili += 1

# shape = [1,], value = ['T']
tTensor = gs.Variable("tTensor", np.dtype(np.int64), [1])
wiliUnsqueezeN = gs.Node("Unsqueeze", "wiliUnsqueezeN-" + str(nWili), inputs=[tTensorScalar, wiliConstant0], outputs=[tTensor])
graph.nodes.append(wiliUnsqueezeN)
nWili += 1

# shape = [2,], value = ['B','T']
bCommaTTensor = gs.Variable("bCommaTTensor", np.dtype(np.int64), [2])
wiliConcatN = gs.Node("Concat", "wiliConcatN-" + str(nWili), inputs=[bTensor, tTensor], outputs=[bCommaTTensor], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliConcatN)
nWili += 1

# shape = [1,], value = ['B'* 10]
b10Tensor = gs.Variable("b10Tensor", np.dtype(np.int64), [1])
wiliMulN = gs.Node("Mul", "wiliMulN-" + str(nWili), inputs=[bTensor, wiliConstant10], outputs=[b10Tensor])
graph.nodes.append(wiliMulN)
nWili += 1

# shape = [3,], value = ['B'*10,'T',256]
b10CommaTComma256Tensor = gs.Variable("b10CommaTComma256Tensor", np.dtype(np.int64), [3])
wiliConcatN = gs.Node("Concat", "wiliConcatN-" + str(nWili), inputs=[b10Tensor, tTensor, wiliConstant256], outputs=[b10CommaTComma256Tensor], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliConcatN)
nWili += 1

# shape = [2,], value = ['B',10]
bComma10Tensor = gs.Variable("bComma10Tensor", np.dtype(np.int64), [2])
wiliConcatN = gs.Node("Concat", "wiliConcatN-" + str(nWili), inputs=[bTensor, wiliConstant10], outputs=[bComma10Tensor], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliConcatN)
nWili += 1

# shape = [2,], value = ['B'*10,63]
b10Comma63Tensor = gs.Variable("b10Comma63Tensor", np.dtype(np.int64), [2])
wiliConcatN = gs.Node("Concat", "wiliConcatN-" + str(nWili), inputs=[b10Tensor, wiliConstant63], outputs=[b10Comma63Tensor], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliConcatN)
nWili += 1

# shape = [2,], value = ['B'*10,64]
b10Comma64Tensor = gs.Variable("b10Comma64Tensor", np.dtype(np.int64), [2])
wiliConcatN = gs.Node("Concat", "wiliConcatN-" + str(nWili), inputs=[b10Tensor, wiliConstant64], outputs=[b10Comma64Tensor], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliConcatN)
nWili += 1

# shape = [1,], value = ['B'*10*63]
b1063Tensor = gs.Variable("b1063Tensor", np.dtype(np.int64), [1])
wiliMulN = gs.Node("Mul", "wiliMulN-" + str(nWili), inputs=[b10Tensor, wiliConstant63], outputs=[b1063Tensor])
graph.nodes.append(wiliMulN)
nWili += 1

# shape = [4,], value = ['B'*10,1,1,'T']
b10Comma1Comma1CommaTTensor = gs.Variable("b10Comma1Comma1CommaTTensor", np.dtype(np.int64), [4])
wiliConcatN = gs.Node("Concat", "wiliConcatN-" + str(nWili), inputs=[b10Tensor, wiliConstant1, wiliConstant1, tTensor], outputs=[b10Comma1Comma1CommaTTensor], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliConcatN)
nWili += 1

# shape = [4,], value = ['B',10,63,4233]
bComma10Comma63Comma4233Tensor = gs.Variable("bComma10Comma63Comma4233Tensor", np.dtype(np.int64), [4])
wiliConcatN = gs.Node("Concat", "wiliConcatN-" + str(nWili), inputs=[bTensor, wiliConstant10, wiliConstant63, wiliConstant4233], outputs=[bComma10Comma63Comma4233Tensor], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliConcatN)
nWili += 1

# shape = [2,], value = ['B'*10*63,256]
b1063Comma256Tensor = gs.Variable("b1063Comma256Tensor", np.dtype(np.int64), [2])
wiliConcatN = gs.Node("Concat", "wiliConcatN-" + str(nWili), inputs=[b1063Tensor, wiliConstant256], outputs=[b1063Comma256Tensor], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliConcatN)
nWili += 1

# shape = [3,], value = ['B'*10,63,256]
b10Comma63Comma256Tensor = gs.Variable("b10Comma63Comma256Tensor", np.dtype(np.int64), [3])
wiliConcatN = gs.Node("Concat", "wiliConcatN-" + str(nWili), inputs=[b10Tensor, wiliConstant63, wiliConstant256], outputs=[b10Comma63Comma256Tensor], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliConcatN)
nWili += 1

# Round 1: Not
if bNot:
    for node in graph.nodes:
        if "Not" in node.name:
            castV = gs.Variable("castV-" + str(5000 + nNot), np.dtype(bool), None)
            castN = gs.Node("Cast", "CastN-" + str(6000 + nNot), inputs=[node.inputs[0]], outputs=[castV], attrs=OrderedDict([('to', onnx.TensorProto.BOOL)]))
            graph.nodes.append(castN)
            nNot += 1
            node.inputs = [castV]
            continue

# Round 2: Layer Normalization
if bLayerNormPlugin:
    for node in graph.nodes:
        if node.op == 'ReduceMean' and \
            node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
            node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
            node.o().o(0).o().op == 'ReduceMean' and \
            node.o().o(0).o().o().op == 'Add' and \
            node.o().o(0).o().o().o().op == 'Sqrt' and \
            node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1) and \
            node.o().o(0).o().o().o().o().o().op == 'Mul' and \
            node.o().o(0).o().o().o().o().o().o().op == 'Add':

            inputTensor = node.inputs[0]

            lastMultipyNode = node.o().o(0).o().o().o().o().o()
            index = ['weight' in i.name for i in lastMultipyNode.inputs].index(True)
            b = np.array(deepcopy(lastMultipyNode.inputs[index].values.tolist()), dtype=np.float32)
            constantB = gs.Constant("LayerNormB-" + str(nLayerNormPlugin), np.ascontiguousarray(b.reshape(-1)))  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!

            lastAddNode = node.o().o(0).o().o().o().o().o().o()
            index = ['bias' in i.name for i in lastAddNode.inputs].index(True)
            a = np.array(deepcopy(lastAddNode.inputs[index].values.tolist()), dtype=np.float32)
            constantA = gs.Constant("LayerNormA-" + str(nLayerNormPlugin), np.ascontiguousarray(a.reshape(-1)))

            inputList = [inputTensor, constantB, constantA]
            layerNormV = gs.Variable("LayerNormV-" + str(nLayerNormPlugin), np.dtype(np.float32), None)
            layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=inputList, outputs=[layerNormV])
            graph.nodes.append(layerNormN)
            nLayerNormPlugin += 1

            for n in graph.nodes:
                if lastAddNode.outputs[0] in n.inputs:
                    index = n.inputs.index(lastAddNode.outputs[0])
                    n.inputs[index] = layerNormN.outputs[0]
            lastAddNode.outputs = []
            continue

# Round 3: Shape operation
if bShapeOperation:
    for node in graph.nodes:

        if node.op == 'Expand' and node.name == 'Expand_111':
            node.inputs[0] = wiliConstantRange63
            node.inputs[1] = b10Comma63Tensor
            nShapeOperation += 2

        if node.op == 'And' and node.name == 'And_152':
            node.op = 'Or'
            node.name = 'wiliOr-' + str(nShapeOperation)
            nShapeOperation += 1
            node.inputs[0] = node.i().i().i().i().outputs[0]  # Remove Not
            node.inputs[1] = wiliConstant1x63x63

            outTensor = node.o().outputs[0]
            node.o().outputs = []
            unsqueezeV = gs.Variable("wiliUnsqueezeV-" + str(nShapeOperation), np.dtype(bool), None)
            unsqueezeN = gs.Node("Unsqueeze", "wiliUnsqueezeN-" + str(nShapeOperation), inputs=[node.outputs[0], wiliConstant1], outputs=[outTensor])
            graph.nodes.append(unsqueezeN)
            nShapeOperation += 1
            for node in graph.nodes:
                if node.op == "Softmax" and node.name in ['Softmax_217', 'Softmax_358', 'Softmax_499', 'Softmax_640', 'Softmax_781', 'Softmax_922']:
                    node.i().inputs[0] = outTensor
                    node.o().inputs[0] = outTensor
                    nShapeOperation += 2
                    continue

        if node.op == "Not" and node.name == "Not_1064":
            for oldNode in graph.nodes:
                if oldNode.op == "GreaterOrEqual" and oldNode.name == "GreaterOrEqual_115":
                    node.inputs[0] = oldNode.outputs[0]

        if node.op == 'Expand' and node.name == 'Expand_43':
            node.inputs[1] = bCommaTTensor
            nShapeOperation += 1

        if node.op == 'Reshape' and node.name == 'Reshape_60':
            node.i().inputs[0] = node.i().i().i().inputs[0]  # Remove Not and Expand
            node.inputs[1] = b10Comma1Comma1CommaTTensor

            outTensor = node.outputs[0]
            for node in graph.nodes:
                if node.op == "Softmax" and node.name in ['Softmax_279', 'Softmax_420', 'Softmax_561', 'Softmax_702', 'Softmax_843', 'Softmax_984']:
                    node.i().inputs[0] = outTensor
                    node.o().inputs[0] = outTensor
                    nShapeOperation += 2
                    continue

        if node.op == 'Reshape' and node.name == 'Reshape_1087':
            node.inputs[1] = bComma10Tensor

        if node.op == 'Reshape' and node.name == 'Reshape_73':
            node.inputs[1] = b10Comma64Tensor

        if node.op == 'Reshape' and node.name == 'Reshape_77':
            node.inputs[1] = b10Tensor

        if node.op == 'Range' and node.name == 'Range_27':
            node.inputs[1] = tTensorScalar

        if node.op == 'Gather' and node.name == 'Gather_154':
            value = node.inputs[0].values
            wiliConstantGather154 = gs.Constant("wiliConstantGather154", np.ascontiguousarray(value * 16))
            node.inputs[0] = wiliConstantGather154
            node.o(1).o().inputs[0] = node.outputs[0]

        if node.op == 'Slice' and node.name == 'Slice_164':
            node.inputs[2] = wiliConstant63

        if node.op == 'Reshape' and node.name == 'Reshape_1038':
            node.o().inputs[0] = node.inputs[0]

            node.inputs[1] = bComma10Comma63Comma4233Tensor

            graph.outputs[0].name = 'deprecated[decoder_out]'
            node.outputs[0].name = 'decoder_out'
            graph.outputs[0] = node.outputs[0]

        if node.op == 'Reshape' and node.name == 'Reshape_22':
            node.i().inputs[0] = graph.inputs[0]
            node.inputs[1] = b10CommaTComma256Tensor
        '''
        # Error operation but keep output correct!
        if node.op == 'ArgMax' and node.name == 'ArgMax_1091':
            node.inputs[0] = node.i().inputs[0]
        '''

# Round 4: 2D Matrix multiplication
if b2DMM:
    for node in graph.nodes:
        if node.op == 'LayerNorm' and node.name in ['LayerNormN-' + str(i) for i in range(2, 18, 3)]:

            reshape1V = gs.Variable("wiliReshape1V-" + str(n2DMM), np.dtype(np.float32), ['B*10*63', 256])
            reshape1N = gs.Node("Reshape", "wiliReshape1N-" + str(n2DMM), inputs=[node.outputs[0], b1063Comma256Tensor], outputs=[reshape1V])
            graph.nodes.append(reshape1N)
            n2DMM += 1

            lastNode = node.o().o().o().o().o()  # Add
            outputTensor = lastNode.outputs[0]

            reshape2V = gs.Variable("wiliReshape2V-" + str(n2DMM), np.dtype(np.float32), ['B*10', 63, 256])
            reshape2N = gs.Node("Reshape", "wiliReshape2N-" + str(n2DMM), inputs=[reshape2V, b10Comma63Comma256Tensor], outputs=[outputTensor])
            graph.nodes.append(reshape2N)
            n2DMM += 1

            lastNode.outputs[0] = reshape2V
            node.o().inputs[0] = reshape1V

# for  debug
if bDebug:
    for node in graph.nodes:
        if node.name == "LayerNormN-2":
            #graph.outputs.append( node.inputs[0] )
            graph.outputs.append(node.outputs[0])
            #graph.outputs = [node.outputs[0]]

graph.cleanup()
onnx.save(gs.export_onnx(graph), destinationOnnx)

print("finish decoder onnx-graphsurgeon!")
print("%4d Not" % (nNot))
print("%4d LayerNormPlugin" % nLayerNormPlugin)
print("%4d ShapeOperation" % nShapeOperation)
print("%4d 2DMM" % n2DMM)
