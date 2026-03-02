# IfCondition structure

+ Steps to run.

```bash
python3 main.py
```

+ The IfCondition structure contains usage of `ConditionLayer`, `IfConditionalInputLayer` and `IfConditionalOutputLayer`.

+ The output tensor of the IfCondition structure is from `IfConditionOutputLayer`, while member function `get_output()` is also provided in `ConditionLayer`, `IfConditionInputLayer` and `IfConditionConditionLayer`, but their output are None.

+ Ranges of parameters

|             Name              |       Range       |
| :---------------------------: | :---------------: |
|   Rank of condition tensor    |         0         |
| Data type of condition tensor | trt.DataType.BOOL |
