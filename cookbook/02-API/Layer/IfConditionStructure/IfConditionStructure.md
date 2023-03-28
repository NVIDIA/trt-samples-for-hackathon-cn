# IfCondition structure

+ Simple example

---

## Simple example

+ Refer to SimpleExample.py
+ Computation porcess:
+ 
```python
if inputT0[0,0,0,0] != 0:
    return inputT0 * 2
else:
    return inputT0
```

+ The output tensor of the IfCondition structure is from IfConditionOutputLayer layer (In fact member function get_output() is also provided in IfConditionInputLayer layer and IfConditionConditionLayer layer, but their output are locked as None).

+ The IfCondition structure contains usage of ConditionLayer, IfConditionalInputLayer and IfConditionalOutputLayer.
