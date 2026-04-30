# MoE layer

+ MoE layer.

+ Steps to run.

```bash
python3 main.py
```

+ Mixture-of-Experts layer: route each token to its selected experts and combine their weighted outputs. See `case_api_minimal` for the minimal API and `case_api_methods` for the weight / bias / quantization setters.

+ Note: `IMoELayer` is currently supported only on SM 10.x (Blackwell) / SM 11.x (Thor); on other hardware `add_moe` may return `None` or the build may fail, which the example guards against.

+ Input / output data type and shape:
  + `hidden_states` of type `T` with shape `[batchSize, seqLen, hiddenSize]`; `selected_experts_for_tokens` of type `M` with shape `[batchSize, seqLen, topK]`; `scores_for_selected_experts` of type `T` with shape `[batchSize, seqLen, topK]`.
  + Optional expert weights (set via `set_gated_weights` / biases): `fc_gate_weights`, `fc_up_weights` of shape `[numExperts, hiddenSize, moeInterSize]`, `fc_down_weights` of shape `[numExperts, moeInterSize, hiddenSize]`.
  + Output of type `T` with shape `[batchSize, seqLen, hiddenSize]`.
  + `T` in [float32, float16, bfloat16], `M` (selected experts) is int32.

+ Available values of `trt.MoEActType`.

| Name | Comment                                                    |
| :--- | :-------------------------------------------------------- |
| NONE | No activation applied between the gate and down projection |
| SILU | SiLU / Swish activation $f(x) = x \cdot \sigma(x)$        |

## Attributes

| Attribute              | Description                                                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| activationType         | Activation type for the MoE layer. Supported values are `MoEActType::kNONE` and `MoEActType::kSILU`.                               |
| quantizationToType     | Type for quantizing the mul output within the MoE layer. Currently only `DataType::kFP8` is supported.                             |
| quantizationBlockShape | Quantization block shape for the mul output. Only used when `setQuantizationDynamicDblQ` is called. (Not supported yet.)           |
| dynQOutputScaleType    | Generated scale type when dynamically quantizing the mul output. Only used with `setQuantizationDynamicDblQ`. (Not supported yet.) |
| swigluParamLimit       | Limit for the SWIGLU parameter. (Not supported yet.)                                                                               |
| swigluParamAlpha       | Alpha for the SWIGLU parameter. (Not supported yet.)                                                                               |
| swigluParamBeta        | Beta for the SWIGLU parameter. (Not supported yet.)                                                                                |
