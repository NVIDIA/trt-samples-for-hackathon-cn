# Assertion Layer

+ Common
+ Buildtime Check
+ message
+ Runtime Check

---

## Common

+ Input tensor
  + T0

+ Output tensor
  + None

+ Data type
  + T0: bool

+ Shape
  + T1.shape == [] or [1]

+ Attribution and default value
  + message = "", customized information of the assertion layer

---

## Buildtime Check

+ Refer to BuildtimeCheck.py

+ Check error during build time.

+ No error while using default code, but receiving error below while replacing code line 36 with line 35:

```txt
[TRT] [E] 4: [graphShapeAnalyzer.cpp::processCheck::862] Error Code 4: Internal Error (IAssertionLayer (Unnamed Layer* 5) [Assertion]: condition[0] is false: 0. inputT0.shape[3] != 5)
```

---

## Message

+ Refer to Message.py

+ Edit content of the assertion information after constructor.

---

## Runtime Check

+ Refer to RuntimeCheck.py

+ Check error during run time.

+ No error while using default code, but receiving error below while replacing code line 64 with line 63:

```txt
[ 0]Input -> DataType.FLOAT (-1, -1, 4, 5) (1, 3, 4, 5) inputT0
[ 1]Input -> DataType.FLOAT (-1, -1) (1, 4) inputT1
[03/24/2023-23:46:30] [TRT] [E] 7: [shapeMachine.cpp::executeContinuation::864] Error Code 7: Internal Error (IAssertionLayer (Unnamed Layer* 6) [Assertion]: condition[0] is false: (EQUAL (# 1 (SHAPE inputT0)) (# 1 (SHAPE inputT1))). inputT0.shape[1] != inputT1.shape[1] Condition '==' violated: 3 != 4. Instruction: CHECK_EQUAL 3 4.)
[ 2]Output-> DataType.INT32 (1,) (0) (Unnamed Layer* 7) [Identity]_output
```
