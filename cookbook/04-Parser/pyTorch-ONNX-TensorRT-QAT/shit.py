import torch

torch.backends.quantized.engine = "qnnpack"
pt_inputs = tuple(torch.from_numpy(x) for x in sample_inputs)
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
q_model = torch.quantization.prepare(model, inplace=False)
q_model = torch.quantization.convert(q_model, inplace=False)

traced_model = torch.jit.trace(q_model, pt_inputs)
buf = io.BytesIO()
torch.jit.save(traced_model, buf)
buf.seek(0)
q_model = torch.jit.load(buf)

q_model.eval()
output = q_model(*pt_inputs)

f = io.BytesIO()
torch.onnx.export(q_model, pt_inputs, f, input_names=input_names, example_outputs=output,
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
f.seek(0)

