"""Build a small ONNX model that exercises every `RefitTransformKind`, plus its weight files.

The weights are written in the `.wts` text format documented in `02-API/Network`, so the C++ side
can read them without linking protobuf: the point of the example is that the deploy step needs no
ONNX parser at all.
"""

import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).parent
N_CHANNEL = 4
SHAPE = (1, N_CHANNEL, 8, 8)

def make_weights(seed: int) -> dict:
    """Every initializer of the model, by ONNX name."""
    rng = np.random.default_rng(seed)
    return {
        "conv.weight": rng.normal(0, 0.3, (N_CHANNEL, N_CHANNEL, 3, 3)).astype(np.float32),
        "conv.bias": rng.normal(0, 0.3, (N_CHANNEL, )).astype(np.float32),
        "bn.scale": rng.uniform(0.5, 1.5, (N_CHANNEL, )).astype(np.float32),
        "bn.bias": rng.normal(0, 0.3, (N_CHANNEL, )).astype(np.float32),
        "bn.mean": rng.normal(0, 0.3, (N_CHANNEL, )).astype(np.float32),
        "bn.var": rng.uniform(0.5, 1.5, (N_CHANNEL, )).astype(np.float32),
        # A float64 initializer, so the parser has to cast it -> kDOUBLE_TO_FLOAT
        "gain.double": rng.uniform(0.9, 1.1, (1, N_CHANNEL, 1, 1)).astype(np.float64),
    }

def build_model(weights: dict, path: Path) -> None:
    initializer = [numpy_helper.from_array(v, name=k) for k, v in weights.items()]

    nodes = [
        helper.make_node("Conv", ["x", "conv.weight", "conv.bias"], ["conv_out"], pads=[1, 1, 1, 1]),
        helper.make_node("BatchNormalization", ["conv_out", "bn.scale", "bn.bias", "bn.mean", "bn.var"], ["bn_out"], epsilon=1e-3),
        helper.make_node("Relu", ["bn_out"], ["relu_out"]),
        # A Constant node: its value lives in a node attribute, not an initializer -> kCONSTANT_NODE
        helper.make_node("Constant", [], ["const_out"], value=numpy_helper.from_array(np.full((1, N_CHANNEL, 1, 1), 0.25, dtype=np.float32))),
        helper.make_node("Mul", ["relu_out", "const_out"], ["scaled"]),
        # ConstantOfShape: value is an attribute too -> kCONSTANT_OF_SHAPE
        helper.make_node("ConstantOfShape", ["shape"], ["ones"], value=numpy_helper.from_array(np.array([1.5], dtype=np.float32))),
        helper.make_node("Mul", ["scaled", "ones"], ["scaled2"]),
        helper.make_node("Cast", ["gain.double"], ["gain"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["scaled2", "gain"], ["y"]),
    ]
    shape_initializer = numpy_helper.from_array(np.array(SHAPE, dtype=np.int64), name="shape")

    graph = helper.make_graph(
        nodes,
        "refit-observer-demo",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(SHAPE))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(SHAPE))],
        initializer + [shape_initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return

def write_wts(weights: dict, path: Path) -> None:
    """`name count hex0 hex1 ...`, big-endian IEEE-754. See 02-API/Network/weight_transport.py."""
    with open(path, "w") as f:
        f.write(f"{len(weights)}\n")
        for name, array in weights.items():
            flat = array.reshape(-1)
            # float64 initializers are written as float64 hex, 16 characters per value
            pack = ">d" if array.dtype == np.float64 else ">f"
            f.write(f"{name} {flat.size} {8 if array.dtype == np.float64 else 4}")
            for value in flat:
                f.write(" " + struct.pack(pack, float(value)).hex())
            f.write("\n")
    return

def main() -> None:
    for tag, seed in [("v1", 31193), ("v2", 97)]:
        weights = make_weights(seed)
        build_model(weights, HERE / f"model-{tag}.onnx")
        write_wts(weights, HERE / f"model-{tag}.wts")
        print(f"    model-{tag}.onnx  {(HERE / f'model-{tag}.onnx').stat().st_size:>7,} bytes"
              f"   model-{tag}.wts  {(HERE / f'model-{tag}.wts').stat().st_size:>7,} bytes")

    data = np.random.default_rng(7).normal(0, 1, SHAPE).astype(np.float32)
    np.save(HERE / "input.npy", data)

    # Reference outputs, so the C++ side has something to be checked against.
    import onnxruntime
    for tag in ["v1", "v2"]:
        session = onnxruntime.InferenceSession(str(HERE / f"model-{tag}.onnx"), providers=["CPUExecutionProvider"])
        output = session.run(None, {"x": data})[0]
        np.save(HERE / f"reference-{tag}.npy", output)
        print(f"    reference-{tag}.npy  sum = {output.sum():.6f}")

    # Flat text the C++ side reads without numpy.
    for name, array in [("input", data), ("reference-v1", np.load(HERE / "reference-v1.npy")), ("reference-v2", np.load(HERE / "reference-v2.npy"))]:
        with open(HERE / f"{name}.txt", "w") as f:
            f.write(f"{array.size}\n")
            f.write(" ".join(f"{x:.9g}" for x in array.reshape(-1)))
            f.write("\n")
    return

if __name__ == "__main__":
    main()
    print("Finish")
