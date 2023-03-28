# Assertion Layer

+ Buildtime Check
+ message
+ Runtime Check

---

## Simple example

+ Refer to SimpleExample.py

+ Check during build time.

+ Run with default code, no problem with the network, receiving information:

```txt
Succeeded building engine!
```

+ Use code line 41 rather line 39, receiving error information:

```txt
[TRT] [E] 4: [graphShapeAnalyzer.cpp::processCheck::581] Error Code 4: Internal Error (IAssertionLayer (Unnamed Layer* 5) [Assertion]: condition[0] is false: 0. inputT0.shape[3] is not 4!)
```

---

## Message

+ Refer to Message.py

+ Adjust content of the assertion information after constructor.

+ Receiving error information:

```txt
[11/15/2022-07:08:11] [TRT] [E] 4: [graphShapeAnalyzer.cpp::processCheck::722] Error Code 4: Internal Error (IAssertionLayer (Unnamed Layer* 5) [Assertion]: condition[0] is false: 0. Edited message!)

```

---

## Runtime Check

+ Refer to RuntimeCheck.py

+ Check during run time, check whether the lengths of the second dimension in the two input tensors are the same.

+ Using data2 (shape [1,4]) as the second input tensor will receive error information when calling context related functions.

```txt
[11/15/2022-07:21:01] [TRT] [E] 7: [shapeMachine.cpp::executeContinuation::738] Error Code 7: Internal Error (IAssertionLayer (Unnamed Layer* 6) [Assertion]: condition[0] is false: (EQUAL (# 1 (SHAPE inputT0)) (# 1 (SHAPE inputT1))). inputT0.shape[1] != inputT1.shape[1] Condition '==' violated: 3 != 4. Instruction: CHECK_EQUAL 3 4.)
```
