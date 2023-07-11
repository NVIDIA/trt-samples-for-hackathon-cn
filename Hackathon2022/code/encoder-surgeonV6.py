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
bNotV3 = True
nNotV3 = 0
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
bAttentionPlugin = True
nAttentionPlugin = 0

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

# Round 1: Not version 3, adjust Not to fit Attention Plugin
if bNotV3:
    for node in graph.nodes:
        if node.op == 'Slice' and node.name == 'Slice_79':
            # adjust node before Slice_79
            greaterOrEqualNode = node.i().i().i()
            greaterOrEqualNode.op = 'Less'
            greaterOrEqualNode.name = 'LessN-' + str(nNotV3)
            nNotV3 += 1

            castV = gs.Variable("wiliCastV-" + str(nNotV3), np.dtype(np.int32), None)
            castN = gs.Node("Cast", "wiliCastN-" + str(nNotV3), inputs=[greaterOrEqualNode.outputs[0]], outputs=[castV], attrs=OrderedDict([('to', onnx.TensorProto.INT32)]))
            graph.nodes.append(castN)
            nNotV3 += 1

            # adjust Slice_79
            node.inputs[0] = castV
            node.inputs[2] = wiliConstantM4  # end
            node.inputs[3] = wiliConstant1  # axes
            node.inputs[4] = wiliConstant4  # step

            slice84Node = node.o()
            tensor613 = slice84Node.outputs[0]
            tensor613.dtype = np.dtype(np.int32)

            unsqueezeTo3DN = gs.Node("Unsqueeze", "wiliUnsqueezeTo3DN-" + str(nNotV3), inputs=[node.outputs[0], wiliConstant1], outputs=[tensor613])
            graph.nodes.append(unsqueezeTo3DN)
            nNotV3 += 1

            slice84Node.outputs = []
            continue

        if node.op == 'Not' and node.name != 'Not_30':
            castV = gs.Variable("castV-" + str(nNotV3), np.dtype(bool), None)
            castN = gs.Node("Cast", "CastN-" + str(nNotV3), inputs=[node.inputs[0]], outputs=[castV], attrs=OrderedDict([('to', onnx.TensorProto.BOOL)]))
            graph.nodes.append(castN)
            nNotV3 += 1
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

# Round 3: constant fold, removed for adopting Attention plugin
if bConstantFold:
    for node in graph.nodes:
        if node.op == 'Slice' and node.name == 'Slice_74':
            node.inputs[0].values = node.inputs[0].values[:, :256, :]
            nConstantFold += 1
            break

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

# Round 7: Attention Plugin, by Xuewei Li
if bAttentionPlugin:
    tensorTable = graph.tensors()
    para = {}
    inputTable = {}
    outputTable = {}

    for name in tensorTable:
        if "self_attn.pos_bias_u" in name:
            para[name] = np.array(deepcopy(tensorTable[name].values.tolist()), dtype=np.float32)

        if "self_attn.pos_bias_v" in name:
            para[name] = np.array(deepcopy(tensorTable[name].values.tolist()), dtype=np.float32)
            tensor = tensorTable[name]
            test = tensor.outputs[0].o().o().i(1, 0).i(0).i(0)
            if test.op == "MatMul":
                t = test.inputs[1]
                para[name[:len(name) - 20] + "self_attn.linear_pos.weight"] = np.array(deepcopy(t.values.tolist()), dtype=np.float32)
            else:
                raise Exception("not correct!")

        if "self_attn.linear_q.bias" in name or "self_attn.linear_k.bias" in name\
                or "self_attn.linear_v.bias" in name or "self_attn.linear_out.bias" in name:
            tensor = tensorTable[name]
            para[name] = np.array(deepcopy(tensor.values.tolist()), dtype=np.float32)
            weight_node = tensor.outputs[0].i(1, 0)
            if weight_node.op == "MatMul":
                para[name[:-4] + "weight"] = np.array(deepcopy(weight_node.inputs[1].values.tolist()), dtype=np.float32)
                if "self_attn.linear_out.bias" in name:
                    ot = weight_node.inputs[0]
                    ot_node = ot.inputs[0]
                    outputTable[name[:-25]] = ot.name
                    outputTable[name[:-25] + "node"] = ot_node
                if "self_attn.linear_q.bias" in name:
                    it = weight_node.inputs[0]
                    inputTable[name[:-23]] = it.name
            else:
                raise Exception("not correct!")

    input1 = tensorTable["603"]
    mask = tensorTable["613"]
    test = tensorTable["551"]
    input2 = gs.Variable("CastVariable-", np.dtype(np.int32), None)
    castN = gs.Node("Cast", "CastnNode-", inputs=[mask], outputs=[input2], attrs=OrderedDict([('to', 1)]))
    graph.nodes.append(castN)

    index = 0
    for name in inputTable:
        input0 = tensorTable[inputTable[name]]
        inputList = [input0, input1, input2]
        temp = para[name + 'self_attn.pos_bias_u']
        inputList.append(gs.Constant(name + "-pos_bias_u", np.ascontiguousarray(para[name + 'self_attn.pos_bias_u'].reshape(-1))))
        inputList.append(gs.Constant(name + "-pos_bias_v", np.ascontiguousarray(para[name + 'self_attn.pos_bias_v'].reshape(-1))))

        q_weight = para[name + 'self_attn.linear_q.weight']
        k_weight = para[name + 'self_attn.linear_k.weight']
        v_weight = para[name + 'self_attn.linear_v.weight']

        qkv_weight = np.stack((q_weight, k_weight, v_weight))

        inputList.append(gs.Constant(name + "-linear_qkv_weight", np.ascontiguousarray(qkv_weight.reshape(-1))))
        inputList.append(gs.Constant(name + "-linear_q_bias", np.ascontiguousarray(para[name + 'self_attn.linear_q.bias'].reshape(-1))))
        inputList.append(gs.Constant(name + "-linear_k_bias", np.ascontiguousarray(para[name + 'self_attn.linear_k.bias'].reshape(-1))))
        inputList.append(gs.Constant(name + "-linear_v_bias", np.ascontiguousarray(para[name + 'self_attn.linear_v.bias'].reshape(-1))))
        inputList.append(gs.Constant(name + "-linear_out_weight", np.ascontiguousarray(para[name + 'self_attn.linear_out.weight'].reshape(-1))))
        inputList.append(gs.Constant(name + "-linear_out_bias", np.ascontiguousarray(para[name + 'self_attn.linear_out.bias'].reshape(-1))))
        inputList.append(gs.Constant(name + "-linear_pos_weight", np.ascontiguousarray(para[name + 'self_attn.linear_pos.weight'].reshape(-1))))

        ForwardAttentionN = gs.Node("Attention", "Attention" + name, inputs=inputList, outputs=[tensorTable[outputTable[name]]])
        # ForwardAttentionN.inputs[0].dtype = np.dtype(np.float32)
        # ForwardAttentionN.inputs[1].dtype = np.dtype(np.float32)
        # ForwardAttentionN.inputs[2].dtype = np.dtype(np.int32)
        graph.nodes.append(ForwardAttentionN)
        outputTable[name + "node"].outputs = []
        index = index + 1

        nAttentionPlugin += 1

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
print("%4d NotV3" % nNotV3)

print("%4d LayerNormPlugin" % nLayerNormPlugin)
print("%4d ConstantFold" % nConstantFold)
print("%4d ReshapeMatmulToConv" % nReshapeMatmulToConv)
print("%4d Expand" % nExpand)
print("%4d 2DMM" % n2DMM)
print("%4d Wili" % nWili)
print("%4d AttentionMM" % nAttentionMM)
print("%4d AttentionPlugin" % nAttentionPlugin)