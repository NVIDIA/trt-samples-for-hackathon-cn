# Assertion Layer

+ Buildtime Check
+ message
+ Runtime Check

---

## Simple example

+ Refer to SimpleExample.py, check during build time.

+ Run with default code, no problem with the network, receiving information **Succeeded building engine!**

+ Use code line 42 rather line 40, receiving error information:

```txt
[TRT] [E] 4: [graphShapeAnalyzer.cpp::processCheck::581] Error Code 4: Internal Error (IAssertionLayer (Unnamed Layer* 5) [Assertion]: condition[0] is false: 0. inputT0.shape[3] is not 4!)
```

---

## Message

+ Refer to Message.py, adjust content of the assertion layer after constructor.

+ Input tensor is the same as SimpleExample.py

+ Receiving error information:

```txt
[11/15/2022-07:08:11] [TRT] [E] 4: [graphShapeAnalyzer.cpp::processCheck::722] Error Code 4: Internal Error (IAssertionLayer (Unnamed Layer* 5) [Assertion]: condition[0] is false: 0. Edited message!)

```

---

## Runtime Check

+ Refer to RuntimeCheck.py, check during run time.

+ Check whether the length of the second dimension of the ipnut tensors is the same.

+ Using data2 (shape [1,4]) as the second input tensor will receive error information when calling context related functions.

```txt
[11/15/2022-07:21:01] [TRT] [E] 7: [shapeMachine.cpp::executeContinuation::738] Error Code 7: Internal Error (IAssertionLayer (Unnamed Layer* 6) [Assertion]: condition[0] is false: (EQUAL (# 1 (SHAPE inputT0)) (# 1 (SHAPE inputT1))). inputT0.shape[1] != inputT1.shape[1] Condition '==' violated: 3 != 4. Instruction: CHECK_EQUAL 3 4.)
```
