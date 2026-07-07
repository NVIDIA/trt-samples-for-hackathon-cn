# Dist collective layer

+ Distributed collective communication layer that exchanges or reduces a tensor across multiple ranks/GPUs.

+ Steps to run.

```bash
python3 main.py
```

+ Packages `nccl4py` (imported as `nccl`) and at least 2 GPUs are needed.

+ Available values of `trt.CollectiveOperation`.

| Name           | Comment                                                                        | Output shape                                      |
| -------------- | ------------------------------------------------------------------------------ | ------------------------------------------------- |
| ALL_REDUCE     | Every rank reduces the inputs of all ranks; all ranks receive the same result. | $[d_0, d_1, ..., d_n]$                            |
| ALL_GATHER     | Every rank gathers the inputs of all ranks, concatenated along dimension 0.    | $[nb\_rank \cdot d_0, d_1, ..., d_n]$             |
| BROADCAST      | The root rank broadcasts its input to all ranks.                               | $[d_0, d_1, ..., d_n]$                            |
| REDUCE         | All ranks reduce their inputs onto the root rank only.                         | $[d_0, d_1, ..., d_n]$ (root only)                |
| REDUCE_SCATTER | All ranks reduce, then the result is scattered along dimension 0.              | $[d_0 / nb\_rank, d_1, ..., d_n]$                 |
| ALL_TO_ALL     | Each rank exchanges an equal-sized block with every other rank.                | $[d_0, d_1, ..., d_n]$                            |
| GATHER         | All ranks send their input to the root rank, which concatenates them.          | $[nb\_rank \cdot d_0, d_1, ..., d_n]$ (root only) |
| SCATTER        | The root rank splits its input and distributes one chunk to each rank.         | $[d_0 / nb\_rank, d_1, ..., d_n]$                 |

+ Available values of `trt.ReduceOperation`.

| Name | Comment                                                                                       |
| ---- | --------------------------------------------------------------------------------------------- |
| SUM  | $output = \sum_{r} input_r$                                                                   |
| PROD | $output = \prod_{r} input_r$                                                                  |
| MAX  | $output = \max_{r} input_r$                                                                   |
| MIN  | $output = \min_{r} input_r$                                                                   |
| AVG  | $output = \frac{1}{nb\_rank}\sum_{r} input_r$                                                 |
| NONE | No reduction; used for non-reducing collectives (ALL_GATHER, BROADCAST, GATHER, SCATTER, ...) |

+ A reduction operation (non-`NONE`) is required for `ALL_REDUCE`, `REDUCE`, and `REDUCE_SCATTER`; it must be `NONE` for the other operations.

+ Input / output data type: `float32`, `float16`, `bfloat16`, `float8`, `int64`, `int32`, `int8`, `uint8`, `bool`.

+ Shape: input `[d0, d1, ..., dn]` with `n >= 1`; where the operation involves `nb_rank`, `d0` must be divisible by `nb_rank`. Output shape depends on the operation (see the table above).

+ Attributes of the DistCollective layer.

| Attribute            | Description                                                                                                                                                                              | Data Type           | Default |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------- |
| collective_operation | Specifies the collective operation type (see the `trt.CollectiveOperation` table).                                                                                                       | CollectiveOperation | -       |
| reduce_op            | The reduction operation for reduction-type collectives. Required (non-`NONE`) for `ALL_REDUCE`, `REDUCE`, and `REDUCE_SCATTER`; must be `NONE` for others.                               | ReduceOperation     | -       |
| root                 | Integer rank identifier for root-based operations (`BROADCAST`, `REDUCE`, `GATHER`, `SCATTER`). Must be >= 0 for those operations; use -1 for collective operations without a root rank. | int                 | -1      |
| num_ranks / nb_rank  | Number of participating ranks. When > 1, enables multi-device execution and affects output shape calculations based on the operation type.                                               | int                 | 1       |
| groups               | Optional array of rank IDs defining a communication group; empty by default (all ranks participate).                                                                                     | int[]               | []      |
| group_size           | Count of elements in the groups array — must be 0 when empty, > 0 when non-empty.                                                                                                        | int                 | 0       |
