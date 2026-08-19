# torch.compile Backend (AOT vs JIT)

+ The two Torch-TensorRT front ends side by side, and how to choose.

+ Steps to run.

```shell
python3 main.py
```

## Same builder, different everything else

Both front ends end in the same TensorRT builder, so a model that works in both
gives **bit-identical** results (`max |eager - TensorRT| = 0.0e+00` for all three
spellings). What differs is the workflow around the build.

| | AOT `ir="dynamo"` | JIT `torch.compile(backend="tensorrt")` |
| --- | --- | --- |
| when the engine is built | during the compile call | during the first inference |
| what you get back | a `GraphModule` | a wrapper around the original `nn.Module` |
| can be saved | **yes**, engine embedded | **no** |
| needs `torch.export` to succeed | **yes** | no |
| input outside the shape range | hard failure | recompiles |

`torch.compile(model, backend="tensorrt")`, `backend="torch_tensorrt"` and
`torch_tensorrt.compile(model, ir="torch_compile", ...)` are three spellings of
the same path.

## When the seconds land

```
AOT: compile call  7.27 s, first inference 0.000 s
JIT: compile call  0.00 s, first inference 7.122 s
```

Neither is cheaper -- the same engine is built either way. AOT puts the cost in a
build step; JIT puts it in whichever request arrives first. `torch.compile`
returns instantly because it has not looked at the model yet.

## The case that decides it for you

```python
def forward(self, x):
    y = self.linear(x)
    if y.sum() > 0:        # branches on a VALUE
        return torch.relu(y)
    return torch.tanh(y)
```

```
AOT torch.export: GuardOnDataDependentSymNode -- cannot export, so ir='dynamo' is unavailable
JIT torch.compile: ok, 2 frames compiled, max |eager - TensorRT| = 0.0e+00
```

`torch.export` has to produce one graph, and it can neither pick a branch nor
represent both, so it refuses. There is no `ExportedProgram`, so **the entire AOT
path is unavailable** -- not slower, unavailable.

Dynamo does not have to decide: it compiles up to the branch, evaluates the
condition in Python, and compiles the continuation as a second frame. Both frames
go to TensorRT and the result is exact.

If a model contains control flow like this, JIT is the only option (or rewrite
the branch with `torch.cond`).

## Saveability

```
AOT: saved 65 KB
JIT: cannot save -- ValueError: Input model is of type nn.Module.
```

The JIT wrapper holds no compiled artifact; the engines live in dynamo's cache
for this process only. To ship a JIT-compiled model, either recompile AOT, or use
[`../EngineCaching/`](../EngineCaching/README.md) so the next process reloads the
engines instead of rebuilding them.

## Backend options

The options dictionary is the same for both front ends; accepted keys are the
fields of `torch_tensorrt.dynamo.CompilationSettings`. `torch_executed_ops` pins
an operator to PyTorch regardless of converter support, which splits the graph
around it:

```
torch_executed_ops={}                              : 1 engine, ['_run_on_acc_0']
torch_executed_ops={'torch.ops.aten.relu.default'} : 2 engines, ['_run_on_acc_0', '_run_on_gpu_1', '_run_on_acc_2']
```

`_run_on_gpu_1` is the piece left to PyTorch, sitting between two engines. Useful
to route around a converter bug or to keep one operator in fp32.

## Related

+ [`../DynamicShapes/`](../DynamicShapes/README.md) — what each front end does when an input leaves the declared range.
+ [`../SaveLoad/`](../SaveLoad/README.md) — persisting the AOT result.
+ [`../EngineCaching/`](../EngineCaching/README.md) — making the JIT rebuild cheap.
