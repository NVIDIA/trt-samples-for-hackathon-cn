# Non-Maximum Suppression Layer

+ Simple example
+ bounding_box_format
+ topk_box_limit

---

## Simple example

+ Refer to SimpleExample.py

+ Input tensor:
  + Boxes, shape = \[nB, nC, 4\]
  + Scores, shape = \[nB, nC, 1\]
  + MaxOutputBoxesPerClass, shape = \[\]
  + IoUThreshold, shape = \[\]
  + ScoreThreshold, shape = \[\]

+ Attribute:
  + bounding_box_format \[trt.BoundingBoxFormat\]
  + topk_box_limit \[int\]
  
+ Output tensor:
  + SelectedIndices, shape = \[-1, 3\]
  + NumOutputBoxes, shape = \[\]

---

## bounding_box_format

+ Refer to BoundingBoxFormat.py

+ Avialable activation function.

| trt.BoundingBoxFormat |                Description                 |
| :-------------------: | :----------------------------------------: |
|     CORNER_PAIRS      | (x1, y1, x2, y2), pair of diagonal corners |
|     CENTER_SIZES      |    (x_center, y_center, width, height)     |

+ Default value: bounding_box_format = trt.BoundingBoxFormatCORNER_PAIRS

---

## topk_box_limit

+ Refer to TopkBoxLimit.py

+ Default value: topk_box_limit = 5000
