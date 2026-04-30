# ReverseSequence layer

+ ReverseSequence layer.

+ Steps to run.

```bash
python3 main.py
```

+ Reverse each sequence of the input tensor along `sequence_axis`, with a per-batch reversal length given by the `sequence_lens` input tensor. See `case_simple` for a `[SL, BS, H]` layout and `case_batch_prior` for a `[BS, SL, H]` layout.

+ Input / output data type and shape:
  + Input tensor `input` of type `T` with shape `shape0`; `sequence_lens` tensor of type `T1` with shape `[shape0[batch_axis]]`.
  + Output tensor of type `T` with the same shape `shape0`.
  + `T` in [float16, float32, bfloat16, int32, int64, int8, bool], `T1` in [int32, int64].

+ Attributes

| Name          | Description                                                       | Default | Range          |
| :------------ | :--------------------------------------------------------------- | :------ | :------------- |
| batch_axis    | Dimension index treated as the batch axis                        | 1       | valid axis     |
| sequence_axis | Dimension index treated as the sequence axis (reversed per item) | 0       | valid axis     |
