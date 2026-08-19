# Lazy loaders vs immediately evaluated functions

+ Polygraphy's two API styles, and why every other example in `More/` mixes them.

+ Steps to run.

```bash
python3 main.py
```

## Almost every loader exists twice

```python
EngineFromNetwork(...)       # CamelCase, lazy      -> a callable
engine_from_network(...)     # snake_case, immediate -> the object
```

```
lazy      EngineFromNetwork(...) :      0.05 ms -> EngineFromNetwork
immediate engine_from_network(...):  11239.91 ms -> ICudaEngine
```

Milliseconds against seconds: constructing the lazy form builds **nothing**. It
is a description of the work, not the result. Both can be handed to `TrtRunner`,
which calls a callable itself — that is why the two styles are so easy to mix up.

## The trap: a lazy loader is a recipe, not a cache

```
first  call :   8.35 s
second call :   8.29 s   same object? False
```

**Calling one twice builds twice.** Handing the same `build_engine` to two
runners, or calling it once to inspect the engine and again to run it, pays the
full build cost each time and produces two unrelated engines. Nothing warns; the
only symptom is the wall clock.

To build once and reuse, call the loader yourself and pass the *result* around —
or use the immediate API, which makes that the default.

## Why the lazy style exists

```
lazy loader  deepcopy : ok
lazy loader  pickle   : ok
built engine deepcopy : TypeError: cannot pickle 'tensorrt.tensorrt.ICudaEngine' object
built engine pickle   : TypeError: cannot pickle 'tensorrt.tensorrt.ICudaEngine' object
```

An `ICudaEngine` is a live CUDA object and cannot cross a process boundary. A
lazy loader is plain Python, so it can be sent to a `multiprocessing` worker that
then builds its own engine.

**Rule of thumb**: lazy to describe work that will happen elsewhere or later,
immediate to do it here and now.

## The cost of being lazy: you cannot just reach in and edit

```
immediate: edited the network directly, 27 -> 28 layers
           z(1,), y_immediate(1, 10)
lazy     : wrapped the loader with @func.extend, nothing has run yet
           z(1,), y_lazy(1, 10)
```

With the immediate API the network is an ordinary object. With the lazy API there
is no network yet, so the edit has to be described too — `polygraphy.func.extend`
wraps a loader so your function runs on whatever it produces.

## Related

+ `../../API/` — the original example in this directory; it mixes both styles without saying so.
+ `../04-ExtendInterop/` — `func.extend` in depth.
