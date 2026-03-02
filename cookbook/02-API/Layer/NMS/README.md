# Non-Maximum Suppression Layer

+ Steps to run.

```bash
python3 main.py
```

+ Alternative values of `trt.BoundingBoxFormat`

| Name |                Comment                 | Default value|
| :-------------------: | :----------------------------------------: | :-: |
|     CORNER_PAIRS      | (x1, y1, x2, y2), pair of diagonal corners | *  |
|     CENTER_SIZES      |    (x_center, y_center, width, height)     | |

+ Ranges of parameters

|         Name         |     Range     |
| :------------------: | :-----------: |
| topk_box_limit | 5000 |

+ Default values of parameters

|     Name     |      Comment       |
| :----------: | :----------------: |
|  bounding_box_format  |       trt.BoundingBoxFormat.CORNER_PAIRS        |

+ Input tensor:
  + Boxes, shape = \[nB, nC, 4\]
  + Scores, shape = \[nB, nC, 1\]
  + MaxOutputBoxesPerClass, shape = \[\]
  + IoUThreshold, shape = \[\]
  + ScoreThreshold, shape = \[\]

+ Output tensor:
  + SelectedIndices, shape = \[-1, 3\]
  + NumOutputBoxes, shape = \[\]
