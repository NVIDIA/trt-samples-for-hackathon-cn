# NMS layer

+ NMS (Non-Maximum Suppression) layer.

+ Steps to run.

```bash
python3 main.py
```

+ Filter overlapping bounding boxes per class using their IoU and score thresholds, see `case_simple`. `case_deprecated` shows the older 3-argument constructor.

+ Input tensors:
  + Boxes, type `T1`, shape `[batchSize, numInputBoundingBoxes, numClasses, 4]` or `[batchSize, numInputBoundingBoxes, 4]`.
  + Scores, type `T1`, shape `[batchSize, numInputBoundingBoxes, numClasses]`.
  + MaxOutputBoxesPerClass, int32, shape `[]` (scalar).
  + IoUThreshold (optional), float32, shape `[]`, range $[0, 1]$, default 0.0.
  + ScoreThreshold (optional), float32, shape `[]`, default 0.0.

+ Output tensors:
  + SelectedIndices, type `T2`, shape `[NumOutputBoxes, 3]`, each row is `(batchIndex, classIndex, boxIndex)`.
  + NumOutputBoxes, int32, shape `[]` (scalar).

+ Data type: `T1` (Boxes, Scores) in [float16, float32, bfloat16]; `T2` (SelectedIndices) in [int32, int64]. Each of Boxes, Scores and SelectedIndices can have up to $2^{31}-1$ elements.

+ Available values of `trt.BoundingBoxFormat`.

| Name         | Comment                                                              |
| :----------- | :------------------------------------------------------------------ |
| CORNER_PAIRS | Coordinates are $(x_1, y_1, x_2, y_2)$, the two diagonal corners    |
| CENTER_SIZES | Coordinates are $(x_{center}, y_{center}, width, height)$           |

+ Attributes

| Name                      | Description                                                            | Default | Range                       |
| :------------------------ | :-------------------------------------------------------------------- | :------ | :-------------------------- |
| bounding_box_format (fmt) | Interpretation of the 4 box coordinates                               | CORNER_PAIRS | CORNER_PAIRS, CENTER_SIZES |
| topk_box_limit (limit)    | Max filtered boxes per batch item (device cap: 2000 on SM 5.3/6.2, else 5000) | 5000 | $\le$ device-specific maximum |
| indices_type              | Data type of the SelectedIndices output                              | INT32   | int32, int64                |
