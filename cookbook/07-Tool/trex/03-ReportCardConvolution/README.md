# 03 - Report Card: Convolution

Analyse the engine's **convolution** layers as implicit GEMMs, in the spirit of
`trt-engine-explorer`'s `report_card_convolutions_overview` /
`df_preprocessing.annotate_convolutions`.

For every `Convolution` layer, `annotate_convolutions(plan)` computes:

+ **MACs** - fused multiply-accumulates: `N*K*P*Q*C*R*S / groups`
+ **arithmetic intensity** - MACs per byte of I/O + weights
+ **compute / memory efficiency** - MACs (or bytes) per millisecond
+ the equivalent GEMM dimensions **M, N, K** (`M=N*P*Q`, `N=K`, `K=C*R*S`)

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # analyse the JSON files (no GPU required)
```

## Output

+ `case_conv_table` - text table of latency / MACs / arithmetic intensity / M,N,K per convolution.
+ `conv_metrics.png` - 2x2 bar panel: latency (colored by precision), MACs, arithmetic intensity, footprint.
+ `conv_roofline.png` - arithmetic intensity vs compute efficiency scatter (marker size / color = latency);
  a 2D replacement for the original plotly 3D M-N-K scatter (the M/N/K view lives in the GEMM example).
