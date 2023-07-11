from collections import OrderedDict
from copy import deepcopy
import numpy as np
import onnx
import onnx_graphsurgeon as gs

sourceOnnx = "./encoderV2.onnx"
destinationOnnx = "./encoderV3.onnx"

bConvertToStaticNetwork = False
debugNodeList = []

nWili = 0
bSimplifyOutput = True
bNot = False
nNot = 0
bNotV2 = True
nNotV2 = 0
bMaskPlugin = False
nMaskPlugin = 0
bLayerNormPlugin = True
nLayerNormPlugin = 0
bConstantFold = True
nConstantFold = 0
bReshapeMatmulToConv = True
nReshapeMatmulToConv = 0
bExpand = True
nExpand = 0
b2DMM = True
n2DMM = 0
bAttentionMM = False
nAttentionMM = 0

#graph = gs.import_onnx(onnx.load(sourceOnnx))
graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))

if bConvertToStaticNetwork:
    graph.inputs[0].shape = [3, 17, 256]
    graph.inputs[1].shape = [3]
else:
    graph.inputs[0].shape = ['B', 'T', 80]
    graph.inputs[1].shape = ['B']

# Round 0: ceate useful constant tensor or collect useful shape tensor
wiliConstant0 = gs.Constant("wiliConstant0", np.ascontiguousarray(np.array([0], dtype=np.int64)))  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!
wiliConstant1 = gs.Constant("wiliConstant1", np.ascontiguousarray(np.array([1], dtype=np.int64)))
wiliConstant3 = gs.Constant("wiliConstant3", np.ascontiguousarray(np.array([3], dtype=np.int64)))
wiliConstant4 = gs.Constant("wiliConstant4", np.ascontiguousarray(np.array([4], dtype=np.int64)))
wiliConstantM4 = gs.Constant("wiliConstantM4", np.ascontiguousarray(np.array([-4], dtype=np.int64)))  # minus four
wiliConstant64 = gs.Constant("wiliConstant64", np.ascontiguousarray(np.array([64], dtype=np.int64)))
wiliConstant256 = gs.Constant("wiliConstant256", np.ascontiguousarray(np.array([256], dtype=np.int64)))
wiliConstantS0 = gs.Constant("wiliConstantS0", np.array(0, dtype=np.int64)).to_variable(np.dtype(np.int64), []).to_constant(np.array(0, dtype=np.dtype(np.int64)))
wiliConstantS1 = gs.Constant("wiliConstantS1", np.array(1, dtype=np.int64)).to_variable(np.dtype(np.int64), []).to_constant(np.array(1, dtype=np.dtype(np.int64)))
wiliConstantS2 = gs.Constant("wiliConstantS2", np.array(2, dtype=np.int64)).to_variable(np.dtype(np.int64), []).to_constant(np.array(2, dtype=np.dtype(np.int64)))

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

# shape = [1,], value = ['t4'], t4 = floor('T'/4) - 1
for node in graph.nodes:
    if node.op == 'Relu' and node.name == 'Relu_38':
        shapeV = gs.Variable("wiliShapeV-" + str(nWili), np.dtype(np.int64), [4])
        shapeN = gs.Node("Shape", "wiliShapeN-" + str(nWili), inputs=[node.outputs[0]], outputs=[shapeV])
        graph.nodes.append(shapeN)
        nWili += 1

        t4TensorScalar = gs.Variable("t4TensorScalar", np.dtype(np.int64), [])
        gatherN = gs.Node("Gather", "wiliGatherN-" + str(nWili), inputs=[shapeV, wiliConstantS2], outputs=[t4TensorScalar], attrs=OrderedDict([('axis', 0)]))
        graph.nodes.append(gatherN)
        nWili += 1

        t4Tensor = gs.Variable("t4Tensor", np.dtype(np.int64), [1])
        unsqueezeN = gs.Node("Unsqueeze", "wiliUnsqueezeN-" + str(nWili), inputs=[t4TensorScalar, wiliConstant0], outputs=[t4Tensor])
        graph.nodes.append(unsqueezeN)
        nWili += 1
'''
# Old method to get t4Tensor
for node in graph.nodes:
    if node.op == 'Unsqueeze' and node.name == 'Unsqueeze_56':
        t4Tensor = node.outputs[0] # shape = [1,], value = ['t4'], t4 = floor('T'/4) - 1

# Error method to get t4Tensor
t4_Tensor = gs.Variable("t4_Tensor", np.dtype(np.int64), [1])
wiliT4_N = gs.Node("Div","wiliDivN-"+str(nWili), inputs = [tTensor, wiliConstant4],outputs=[t4_Tensor])
graph.nodes.append(wiliT4_N)
nWili += 1

t4Tensor = gs.Variable("t4Tensor", np.dtype(np.int64), [1]) # shape = [1,], value = ['t4'], t4 = floor('T'/4) - 1
wiliT4N = gs.Node("Sub","wiliSubN-"+str(nWili), inputs = [t4_Tensor, wiliConstant1],outputs=[t4Tensor])
graph.nodes.append(wiliT4N)
nWili += 1
'''

# shape = [1,], value = ['B'*'t4']
bt4Tensor = gs.Variable("bt4Tensor-" + str(nWili), np.dtype(np.int64), [1])
wiliMulN = gs.Node("Mul", "wiliMulN-" + str(nWili), inputs=[bTensor, t4Tensor], outputs=[bt4Tensor])
graph.nodes.append(wiliMulN)
nWili += 1

# shape = [2,], value = ['B'*'t4',256]
bt4Comma256Tensor = gs.Variable("bt4Comma256Tensor-" + str(nWili), np.dtype(np.int64), [2])
wiliConcatN = gs.Node("Concat", "wiliConcatN-" + str(nWili), inputs=[bt4Tensor, wiliConstant256], outputs=[bt4Comma256Tensor], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliConcatN)
nWili += 1

# shape = [3,], value = ['B','t4',256]
bCommat4Comma64Tensor = gs.Variable("bCommat4Comma64Tensor-" + str(nWili), np.dtype(np.int64), [3])
wiliConcatN = gs.Node("Concat", "wiliConcatN-" + str(nWili), inputs=[bTensor, t4Tensor, wiliConstant256], outputs=[bCommat4Comma64Tensor], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliConcatN)
nWili += 1

# shape = [4,], value = ['B','t4',4,64]
bCommat4Comma4Comma64Tensor = gs.Variable("bCommat4Comma4Comma64Tensor-" + str(nWili), np.dtype(np.int64), [4])
wiliConcatN = gs.Node("Concat", "wiliConcatN-" + str(nWili), inputs=[bTensor, t4Tensor, wiliConstant4, wiliConstant64], outputs=[bCommat4Comma4Comma64Tensor], attrs=OrderedDict([('axis', 0)]))
graph.nodes.append(wiliConcatN)
nWili += 1

# Round 0.5: output tensor
if bSimplifyOutput:
    for node in graph.nodes:
        graph.outputs[0].name = 'deprecated[encoder_out]'
        graph.outputs[1].name = 'deprecated[encoder_out_lens]'

    graph.outputs = graph.outputs[:2]
    '''
    wiliDivV0 = gs.Variable("wiliDevV-0", np.dtype(np.int32), ['B'])
    wiliDivN0 = gs.Node("Div", "wiliDivN-0", inputs=[graph.inputs[1], wiliConstant4], outputs=[wiliDivV0])
    graph.nodes.append(wiliDivN0)

    wiliAddV0 = gs.Variable("wiliAddV-0", np.dtype(np.int32), ['B'])
    wiliAddN0 = gs.Node("Add", "wiliAddN-0", inputs=[wiliDivV0, wiliConstant1], outputs=[wiliAddV0])
    graph.nodes.append(wiliAddN0)
    '''

    wiliAddV0 = gs.Variable("wiliAddV-0", np.dtype(np.int32), ['B'])
    wiliAddN0 = gs.Node("Add", "wiliAddN-0", inputs=[graph.inputs[1], wiliConstant3], outputs=[wiliAddV0])
    graph.nodes.append(wiliAddN0)

    wiliDivV0 = gs.Variable("wiliDevV-0", np.dtype(np.int32), ['B'])
    wiliDivN0 = gs.Node("Div", "wiliDivN-0", inputs=[wiliAddV0, wiliConstant4], outputs=[wiliDivV0])
    graph.nodes.append(wiliDivN0)

    wiliMinV0 = gs.Variable("encoder_out_lens", np.dtype(np.int32), ['B'])
    wiliMinN0 = gs.Node("Min", "wiliMinN-0", inputs=[wiliDivV0, t4Tensor], outputs=[wiliMinV0])
    graph.nodes.append(wiliMinN0)

    graph.outputs[1] = wiliMinV0

# Round 1: Not, deprecated since V4
if bNot:
    for node in graph.nodes:
        if node.name == 'Not_30':
            castV = gs.Variable("wiliCastV-" + str(nNot), np.dtype(bool), None)
            castN = gs.Node("Cast", "wiliCastN-" + str(nNot), inputs=[node.i().outputs[0]], outputs=[castV], attrs=OrderedDict([('to', onnx.TensorProto.BOOL)]))
            graph.nodes.append(castN)
            nNot += 1
            node.inputs = [castV]

            castV = gs.Variable("wiliCastV-" + str(nNot), np.dtype(np.int32), None)
            castN = gs.Node("Cast", "wiliCastN-" + str(nNot), inputs=node.outputs, outputs=[castV], attrs=OrderedDict([('to', onnx.TensorProto.INT32)]))
            graph.nodes.append(castN)
            nNot += 1
            node.o().inputs[0] = castV
            node.o().o().outputs[0].dtype = np.dtype(np.int32)
            node.o().o().o().outputs[0].dtype = np.dtype(np.int32)
            continue

        if node.op == 'Not' and node.name != 'Not_30':
            castV = gs.Variable("wiliCastV-" + str(nNot), np.dtype(bool), None)
            castN = gs.Node("Cast", "wiliCastN-" + str(nNot), inputs=[node.inputs[0]], outputs=[castV], attrs=OrderedDict([('to', onnx.TensorProto.BOOL)]))
            graph.nodes.append(castN)
            nNot += 1
            node.inputs = [castV]
            continue

# Round 1.1: Not version 2
if bNotV2:
    for node in graph.nodes:
        if node.op == 'Slice' and node.name == 'Slice_79':
            # adjust node before Slice_79
            greaterOrEqualNode = node.i().i().i()
            greaterOrEqualNode.op = 'Less'
            greaterOrEqualNode.name = 'LessN-' + str(nNotV2)
            nNotV2 += 1

            castV = gs.Variable("wiliCastV-" + str(nNotV2), np.dtype(np.int32), None)
            castN = gs.Node("Cast", "wiliCastN-" + str(nNotV2), inputs=[greaterOrEqualNode.outputs[0]], outputs=[castV], attrs=OrderedDict([('to', onnx.TensorProto.INT32)]))
            graph.nodes.append(castN)
            nNotV2 += 1

            # adjust Slice_79
            node.inputs[0] = castV
            node.inputs[2] = wiliConstantM4  # end
            node.inputs[3] = wiliConstant1  # axes
            node.inputs[4] = wiliConstant4  # step

            slice84Node = node.o()
            tensor613 = slice84Node.outputs[0]
            tensor613.dtype = np.dtype(np.int32)

            unsqueezeTo3DN = gs.Node("Unsqueeze", "wiliUnsqueezeTo3DN-" + str(nNotV2), inputs=[node.outputs[0], wiliConstant1], outputs=[tensor613])
            graph.nodes.append(unsqueezeTo3DN)
            nNotV2 += 1

            slice84Node.outputs = []
            continue

        if node.op == 'Not' and node.name != 'Not_30':
            castV = gs.Variable("castV-" + str(nNotV2), np.dtype(bool), None)
            castN = gs.Node("Cast", "CastN-" + str(nNotV2), inputs=[node.inputs[0]], outputs=[castV], attrs=OrderedDict([('to', onnx.TensorProto.BOOL)]))
            graph.nodes.append(castN)
            nNotV2 += 1
            node.inputs = [castV]
            continue

# Round 1.2: Not version 2, adjust node after Slice_79
if bNotV2:
    for node in graph.nodes:
        if node.op == 'Unsqueeze' and node.name == 'wiliUnsqueezeTo3DN-2':
            castV0 = node.o(0).o().o().o().outputs[0]
            castV1 = node.o(12 + 1).o().o().outputs[0]

    for node in graph.nodes:
        if node.op == "Where" and node.o().op == 'Softmax':
            # unknown reason to make the output be correct only
            #node.inputs[0] = castV0
            #nNotV2 += 1
            continue

        if node.op == "Where" and node.i(2).op == 'Softmax':
            node.inputs[0] = castV0
            nNotV2 += 1
            continue

        if node.op == "Where" and node.o().op == 'Conv':
            node.inputs[0] = castV1
            nNotV2 += 1
            continue
        if node.op == "Where" and node.i(2).op == 'Conv':
            node.inputs[0] = castV1
            nNotV2 += 1
            continue

# Round 1.9: Mask Plugin - Bad Performance
if bMaskPlugin:
    if (True):  # one output
        maskV0 = gs.Variable("wiliMaskV0-" + str(nMaskPlugin), np.dtype(np.int32), ['B', 1, 't4'])  # 0 / 1
        maskN = gs.Node("Mask", "wiliMaskN-" + str(nMaskPlugin), inputs=[graph.inputs[0], graph.inputs[1]], outputs=[maskV0])
        graph.nodes.append(maskN)
        nMaskPlugin += 1

        maskV1 = gs.Variable("wiliMaskV1-" + str(nMaskPlugin), np.dtype(bool), ['B', 1, 't4'])  # 0 / 1
        castN = gs.Node("Equal", "wiliEqualN-" + str(nMaskPlugin), inputs=[maskV0, wiliConstant1], outputs=[maskV1])
        graph.nodes.append(castN)
        nMaskPlugin += 1

        maskV2 = gs.Variable("wiliMaskV2-" + str(nMaskPlugin), np.dtype(bool), ['B', 1, 1, 't4'])  # 1 / 0
        unsqueezeN = gs.Node("Unsqueeze", "wiliUnsqueezeN-" + str(nMaskPlugin), inputs=[maskV1, wiliConstant1], outputs=[maskV2])
        graph.nodes.append(unsqueezeN)
        nMaskPlugin += 1

        for node in graph.nodes:
            if node.op == "Where" and node.o().op == 'Softmax':
                node.inputs[0] = maskV2
                nMaskPlugin += 1
                continue
            if node.op == "Where" and node.i(2).op == 'Softmax':
                node.inputs[0] = maskV2
                nMaskPlugin += 1
                continue
            if node.op == "Where" and node.o().op == 'Conv':
                node.inputs[0] = maskV1
                nMaskPlugin += 1
                continue
            if node.op == "Where" and node.i(2).op == 'Conv':
                node.inputs[0] = maskV1
                nMaskPlugin += 1
                continue

    else:
        # two output
        maskV0 = gs.Variable("wiliMaskV0", np.dtype(np.float32), ['B', 1, 1, 't4'])  # 0 / -6e6
        maskV1 = gs.Variable("wiliMaskV1", np.dtype(np.float32), ['B', 1, 1, 't4'])  # 1 / 0
        maskN = gs.Node("Mask", "wiliMaskN-" + str(nMaskPlugin), inputs=[graph.inputs[0], graph.inputs[1]], outputs=[maskV0, maskV1])
        graph.nodes.append(maskN)

        squeezeAfterMaskV = gs.Variable("wiliSqueezeAfterMaskV", np.dtype(np.float32), ['B', 1, 't4'])
        SqueezeAfterMaskN = gs.Node("Squeeze", "wiliSqueezeAfterMaskN", inputs=[maskV1, wiliConstant1], outputs=[squeezeAfterMaskV])
        graph.nodes.append(SqueezeAfterMaskN)

        for node in graph.nodes:
            if node.op == "Where" and node.o().op == 'Softmax':
                node.i().outputs = []
                node.op = "Add"
                node.inputs = [maskV0, node.inputs[2]]
                nMaskPlugin += 1
            if node.op == "Where" and node.i(2).op == 'Softmax':
                node.i().outputs = []
                node.op = "Mul"
                node.inputs = [maskV1, node.inputs[2]]
                nMaskPlugin += 1
            if node.op == "Where" and node.o().op == 'Conv':
                node.i().outputs = []
                node.op = "Mul"
                node.inputs = [squeezeAfterMaskV, node.inputs[2]]
                nMaskPlugin += 1
            if node.op == "Where" and node.i(2).op == 'Conv':
                node.i().outputs = []
                node.op = "Mul"
                node.inputs = [squeezeAfterMaskV, node.inputs[2]]
                nMaskPlugin += 1

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

            if lastAddNode.outputs[0] in graph.outputs:  # the last LayerNorm provide one of the graph's output, and do not unsqueeze to 4 dimension
                # oldLastAdd -> graph.outputs[0] ===> LayerNorm -> Squeeze -> graph.outputs[0]
                layerNormN.outputs[0].name = 'encoder_out'
                index = graph.outputs.index(lastAddNode.outputs[0])
                graph.outputs[index] = layerNormN.outputs[0]
            else:  # other LayerNorm contain the subsequent Squeeze operation
                for n in graph.nodes:
                    if lastAddNode.outputs[0] in n.inputs:
                        index = n.inputs.index(lastAddNode.outputs[0])
                        n.inputs[index] = layerNormN.outputs[0]

                lastAddNode.outputs = []

            nLayerNormPlugin += 1
            continue

# Round 3: constant fold
if bConstantFold:
    for node in graph.nodes:
        if node.op == 'Slice' and node.name == 'Slice_74':
            node.inputs[2] = t4Tensor
            nConstantFold += 1

            table5000x256 = node.inputs[0].values[0]
            for i in range(1, 24, 2):
                trashNode = node.o(i).o().o()  # Transpose
                factor256x256 = node.o(i).inputs[1].values

                newTable = np.matmul(table5000x256, factor256x256).transpose().reshape(1, 4, 64, 5000)[:, :, :, :256]
                constantData = gs.Constant("wiliConstant-" + str(nConstantFold), np.ascontiguousarray(newTable))
                sliceV = gs.Variable("wiliSliceV-" + str(nConstantFold), np.dtype(np.float32), [1, 4, 64, 't4'])
                sliceN = gs.Node(
                    "Slice",
                    "wiliSliceN-" + str(nConstantFold),
                    inputs=[
                        constantData,  # data
                        wiliConstant0,  # start=0
                        t4Tensor,  # end
                        wiliConstant3,  # axes=3
                        wiliConstant1,  # step=1
                    ],
                    outputs=[sliceV]
                )
                graph.nodes.append(sliceN)
                node.o(i).o().o().o().inputs[1] = sliceV
                trashNode.outputs = []
                nConstantFold += 1

            continue

# Round 4: Reshape + Matmul -> Convolution, by Hongwei
if bReshapeMatmulToConv:
    for node in graph.nodes:
        if node.op == "Relu" and node.name == 'Relu_38':
            matmulNode = node.o(2).o().o()
            addNode = matmulNode.o()
            mulNode = addNode.o()
            convKernel = matmulNode.inputs[1].values.transpose(1, 0).reshape(256, 256, 1, 19).astype(np.float32)
            convKernelV = gs.Constant("wiliConvKernelV-" + str(nReshapeMatmulToConv), np.ascontiguousarray(convKernel))
            nReshapeMatmulToConv += 1
            convBias = addNode.inputs[0].values
            convBiasV = gs.Constant("wiliConvBiasV-" + str(nReshapeMatmulToConv), np.ascontiguousarray(convBias))
            nReshapeMatmulToConv += 1

            convV = gs.Variable("wiliConvV-" + str(nReshapeMatmulToConv), np.dtype(np.float32), ['B', 256, 't4', 1])
            convN = gs.Node("Conv", "wiliConvN-" + str(nReshapeMatmulToConv), inputs=[node.outputs[0], convKernelV, convBiasV], outputs=[convV])
            convN.attrs = OrderedDict([
                ('dilations', [1, 1]),
                ('kernel_shape', [1, 19]),
                ('pads', [0, 0, 0, 0]),
                ('strides', [1, 1]),
            ])
            graph.nodes.append(convN)
            nReshapeMatmulToConv += 1

            squeezeV = gs.Variable("wiliSqueezeV-" + str(nReshapeMatmulToConv), np.dtype(np.float32), ['B', 256, 't4'])
            squeezeN = gs.Node("Squeeze", "wiliSqueezeN-" + str(nReshapeMatmulToConv), inputs=[convV, wiliConstant3], outputs=[squeezeV])
            graph.nodes.append(squeezeN)
            nReshapeMatmulToConv += 1

            transposeV = gs.Variable("wiliTransposeV-" + str(nReshapeMatmulToConv), np.dtype(np.float32), ['B', 't4', 256])
            transposeN = gs.Node("Transpose", "wiliTransposeN-" + str(nReshapeMatmulToConv), inputs=[squeezeV], outputs=[transposeV], attrs=OrderedDict([('perm', [0, 2, 1])]))
            graph.nodes.append(transposeN)
            nReshapeMatmulToConv += 1

            mulNode.inputs[0] = transposeV

# Round 5: Expand_23
if bExpand:
    for node in graph.nodes:
        if node.op == 'Expand' and node.name == 'Expand_23':
            node.i().i().inputs[1] = tTensorScalar
            concatV = gs.Variable("wiliConcatV-" + str(nExpand), np.dtype(np.int64), [2])
            concatN = gs.Node("Concat", "wiliConcatN-" + str(nExpand), inputs=[bTensor, tTensor], outputs=[concatV], attrs=OrderedDict([('axis', 0)]))
            graph.nodes.append(concatN)
            nExpand += 1

            node.inputs[1] = concatV

# Round 6: 2D Matrix multiplication
if b2DMM:
    for node in graph.nodes:
        if node.op == 'MatMul' and node.name != 'MatMul_61' and \
            node.o().op == 'Add' and \
            node.o().o().op == 'Sigmoid' and \
            node.o().o().o().op == 'Mul' and \
            node.o().o().o().o().op == 'MatMul' and \
            node.o().o().o().o().o().op == 'Add' and \
            node.o().o().o().o().o().o().op == 'Mul':

            reshape1V = gs.Variable("wiliReshape1V-" + str(n2DMM), np.dtype(np.float32), ['B*t4', 256])
            reshape1N = gs.Node("Reshape", "wiliReshape1N-" + str(n2DMM), inputs=[node.inputs[0], bt4Comma256Tensor], outputs=[reshape1V])
            graph.nodes.append(reshape1N)
            n2DMM += 1

            node.inputs[0] = reshape1V

            lastNode = node.o().o().o().o().o().o()  # Mul[0.5]

            reshape2V = gs.Variable("wiliReshape2V-" + str(n2DMM), np.dtype(np.float32), ['B', 't4', 256])
            reshape2N = gs.Node("Reshape", "wiliReshape2N-" + str(n2DMM), inputs=[lastNode.inputs[0], bCommat4Comma64Tensor], outputs=[reshape2V])
            graph.nodes.append(reshape2N)
            n2DMM += 1

            lastNode.inputs[0] = reshape2V
    '''
    # reshape within Attention, worse performance
    for node in graph.nodes:
        if node.op == 'Reshape' and node.i(0).op == 'Add' and node.i(1).op == 'Concat' and node.o().op != "Mul":

            node.inputs[1] = shape3V

            if node.o().op == 'Add':
                LayerNormN = node.i().i(1).i()
                layerNormTensor = node.i().i(1).i().outputs[0]

                reshapeV = gs.Variable("reshapeV-"+str(n2DMM), np.dtype(np.float32), ['B','t4',256])
                reshapeN = gs.Node("Reshape","wiliReshapeN-"+str(n2DMM), inputs = [reshapeV, bt4Comma256Tensor],outputs=[layerNormTensor])
                graph.nodes.append(reshapeN)
                n2DMM += 1

                LayerNormN.outputs[0] = reshapeV
    '''

if bAttentionMM:
    for node in graph.nodes:
        if node.op == 'LayerNorm' and node.name == int(node.name[11:]) % 5 == 1:
            qM = node.o(1).inputs[1].values
            qB = node.o(1).o().inputs[0].values
            kM = node.o(2).inputs[1].values
            kB = node.o(2).o().inputs[0].values
            vM = node.o(3).inputs[1].values
            vB = node.o(3).o().inputs[0].values
            bigFactor = np.concatenate([qM, kM, vM], axis=1)
            bigBias = np.concatenate([qB, kB, vB], axis=0)

            bigFactorTensor = gs.Constant("bigFactorTensor" + str(nAttentionMM), np.ascontiguousarray(bigFactor))
            bigBiasTensor = gs.Constant("bigBiasTensor" + str(nAttentionMM), np.ascontiguousarray(bigBias))
            nAttentionMM += 1

            qReshapeN = node.o(1).o().o()
            kReshapeN = node.o(2).o().o()
            vReshapeN = node.o(3).o().o()

            matMulV = gs.Variable("wiliMatMul1V-" + str(nAttentionMM), np.dtype(np.float32), ['B*t4', 256 * 3])
            matMulN = gs.Node("MatMul", "wiliMatMulN-" + str(nAttentionMM), inputs=[node.outputs[0], bigFactorTensor], outputs=[matMulV])
            graph.nodes.append(matMulN)
            nAttentionMM += 1

            addV = gs.Variable("wiliAddV-" + str(nAttentionMM), np.dtype(np.float32), ['B*t4', 256 * 3])
            addN = gs.Node("Add", "wiliAddN-" + str(nAttentionMM), inputs=[matMulV, bigBiasTensor], outputs=[addV])
            graph.nodes.append(addN)
            nAttentionMM += 1

            split0V = gs.Variable("wiliSplit0V-" + str(nAttentionMM), np.dtype(np.float32), ['B*t4', 256])
            split1V = gs.Variable("wiliSplit1V-" + str(nAttentionMM), np.dtype(np.float32), ['B*t4', 256])
            split2V = gs.Variable("wiliSplit2V-" + str(nAttentionMM), np.dtype(np.float32), ['B*t4', 256])
            splitN = gs.Node("Split", "wiliSplitN-" + str(nAttentionMM), inputs=[addV], outputs=[split0V, split1V, split2V], attrs=OrderedDict([('axis', 1)]))
            graph.nodes.append(splitN)
            nAttentionMM += 1

            qReshapeN.inputs[0] = split0V
            qReshapeN.inputs[1] = bCommat4Comma4Comma64Tensor
            kReshapeN.inputs[0] = split1V
            kReshapeN.inputs[1] = bCommat4Comma4Comma64Tensor
            vReshapeN.inputs[0] = split2V
            vReshapeN.inputs[1] = bCommat4Comma4Comma64Tensor

# for  debug
if len(debugNodeList) > 0:
    for node in graph.nodes:
        if node.name in debugNodeList:
            #graph.outputs.append( node.inputs[0] )
            graph.outputs.append(node.outputs[0])
            #graph.outputs = [node.outputs[0]]

graph.cleanup()
onnx.save(gs.export_onnx(graph), destinationOnnx)

print("finish encoder onnx-graphsurgeon!")
print("%4d Not" % nNot)
print("%4d NotV2" % nNotV2)
print("%4d mask" % nMaskPlugin)
print("%4d LayerNormPlugin" % nLayerNormPlugin)
print("%4d ConstantFold" % nConstantFold)
print("%4d ReshapeMatmulToConv" % nReshapeMatmulToConv)
print("%4d Expand" % nExpand)
print("%4d 2DMM" % n2DMM)
print("%4d Wili" % nWili)
print("%4d AttentionMM" % nAttentionMM)
