# Event

+ Demonstrates `IExecutionContext.set_input_consumed_event` and `IExecutionContext.get_input_consumed_event`.

+ Relation with `08-Advance/MultiStream/main.py`:
  + `MultiStream` uses CUDA events for inter-stream scheduling.
  + This example focuses on TensorRT input-lifetime synchronization event.

+ Steps to run.

```bash
python3 main.py
```
