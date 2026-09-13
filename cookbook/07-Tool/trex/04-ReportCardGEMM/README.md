# 04 - Report Card: GEMM

Inspect the engine's **convolutions expressed as matrix multiplies** (M, N, K)
and relate those dimensions to latency. This is the cookbook re-implementation
of `trt-engine-explorer`'s `report_card_gemm_MNK` /
`report_card_gemm_MNK_scatter` / `report_card_perf_scatter`.

A convolution is an implicit GEMM `(M, K) x (K, N)` with
`M = N*P*Q`, `N = K_out`, `K = C*R*S` (computed by `annotate_convolutions`,
see the `03-ReportCardConvolution` example).

The originals render **interactive plotly 3D scatters**; here the 3D M-N-K view
is projected to 2D Matplotlib figures.

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # analyse the JSON files (no GPU required)
```

## Figures produced

| File                          | Content                                                          |
| ----------------------------- | ---------------------------------------------------------------- |
| `gemm_mnk_bars.png`           | grouped bar of M, N, K per convolution (log scale)               |
| `gemm_mnk_vs_latency.png`     | M / N / K each vs latency, colored by footprint (2D port of the plotly MNK scatter) |
| `gemm_mnk_projection.png`     | M vs N scatter, marker size = K, color = latency (2D projection of the 3D M-N-K scatter) |
