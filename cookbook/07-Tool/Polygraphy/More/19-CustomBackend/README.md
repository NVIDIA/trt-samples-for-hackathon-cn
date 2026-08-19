# A custom backend for `polygraphy run`

+ Add a runner to `polygraphy run` with an extension module.

+ Steps to run.

```bash
python3 main.py                       # installs extension/, runs the cases, uninstalls it
# or, by hand:
pip install ./extension
polygraphy run model.onnx --trt --cookbook-ref --cookbook-ref-precision float64
```

> `main.py` **installs a package into the current environment** and uninstalls it
> again in a `finally` block. That is not incidental — see the first section.

`extension/` is a real installable package, `polygraphy_cookbook_ref`:

```
extension/
├── pyproject.toml                     # declares the polygraphy.run.plugins entry point
└── polygraphy_cookbook_ref/
    ├── __init__.py                    # registers AddScalar with PluginRefRunner on import
    ├── export.py                      # the entry point: returns argument groups
    ├── backend/runner.py              # CookbookRefRunner: a NumPy interpreter
    └── args/runner.py                 # CookbookRefRunnerArgs: --cookbook-ref
```

## The package must be installed, not just importable

```
python3 -c 'import polygraphy_cookbook_ref' with PYTHONPATH: imported
`polygraphy run --help` with the same PYTHONPATH mentions --cookbook-ref: False
after `pip install ./extension`                              : True
```

`polygraphy run` reads `importlib.metadata.entry_points()` for the group
`polygraphy.run.plugins`, and that comes from installed distribution metadata,
which `sys.path` does not provide. The package imports, the option is missing,
and there is no diagnostic at all — `--cookbook-ref` is simply not a recognised
argument.

The half-state worth knowing: building the package once leaves an `*.egg-info`
directory in the source tree, and that *is* metadata. After `pip uninstall`,
`PYTHONPATH=extension/` keeps working, which looks like PYTHONPATH being enough.
It is the stale build artifact. `main.py` deletes it when uninstalling.

## What one entry point buys

```toml
[project.entry-points."polygraphy.run.plugins"]
cookbook-ref = "polygraphy_cookbook_ref.export:export_argument_groups"
```

The group name is fixed; the entry name is free. The target takes no arguments
and returns argument-group instances, which are parsed after all of Polygraphy's
own.

```
--cookbook-ref        Run inference using Cookbook NumPy Reference.
Cookbook NumPy Reference Inference:
--cookbook-ref-precision {float32,float64}
`polygraphy convert --help` mentions cookbook-ref: False
`polygraphy inspect --help` mentions cookbook-ref: False
```

The scope is `polygraphy run` and nothing else. There is no equivalent hook for
adding a whole subcommand — see
[`../18-WritingACliTool/`](../18-WritingACliTool/README.md).

## The runner is invoked by generated code

`add_to_script_impl` does not construct a runner; it appends lines to a Python
script that `polygraphy run` then executes. `--gen-script -` prints it:

```python
# Loaders
load_onnx = OnnxFromPath('model-cancellation.onnx')
gs_from_onnx = GsFromOnnx(load_onnx)

# Runners
runners = [
    CookbookRefRunner(gs_from_onnx, precision='float64'),
]

# Runner Execution
results = Comparator.run(runners)
```

This is both how you debug an extension and how a CLI invocation becomes the
first draft of a Python script.

## Why write a runner at all

`PluginRefRunner`'s op table has three entries, so `--pluginref` cannot run a
graph containing `Mul` ([`../16-PluginReference/`](../16-PluginReference/README.md)):

```
polygraphy run model-cancellation.onnx --pluginref
  [!] Op: Mul does not have a reference implementation registered!
polygraphy run model-cancellation.onnx --onnxrt --cookbook-ref
  [I] PASSED | All outputs matched | Outputs: ['y']
```

## float64 as the reference

The model is `(x/3 + 1e7) - 1e7`, which is `x/3` on paper. In float32 the `+1e7`
throws away every significant digit, and both ONNX-Runtime and the float32
reference reproduce that loss **exactly**:

```
float32 reference vs onnxrt:
  max_absdiff=0 max_reldiff=0 mean_absdiff=0
float64 reference vs onnxrt:
  max_absdiff=0.29271 max_reldiff=1 mean_absdiff=0.12565
float64 reference vs TensorRT:
  max_absdiff=0.29271 max_reldiff=1 mean_absdiff=0.12565
```

Two backends agreeing to `max_absdiff=0` is not evidence that the answer is
right. Switching one of them to float64 turns the same command into a
numerical-stability check, and the model fails it — `max_reldiff=1` on values of
order 0.3 means nothing survived. TensorRT gives the identical answer to
ONNX-Runtime, which is the useful part: the error is the model's, not the
backend's.

That precision knob is the thing a built-in runner cannot give you, and it is
about 15 lines of extension.

## Importing the module registers an op

`polygraphy_cookbook_ref/__init__.py` calls `register("AddScalar")` at import
time, and `polygraphy run` imports the module to reach the entry point:

```
polygraphy run model-addscalar.onnx --pluginref   (with the extension installed)
  [I] PASSED
the same command after `pip uninstall`
  [!] Op: AddScalar does not have a reference implementation registered!
```

So the extension fixes `--pluginref` without adding any option — which is the only
route, since there is no CLI flag for registering a reference implementation. It
is also a reminder that installing an extension module runs its code; the side
effect here is deliberate and would be just as easy to introduce by accident.

## Prefix your option names

```
subscribing a group that redefines --onnx-outputs -> ArgumentError: argument --onnx-outputs: conflicting option string: --onnx-outputs
```

`polygraphy run` already owns around 200 options, and the collision fires while
the parser is being built — so `polygraphy run --help` breaks for every user of
that environment, not just for invocations that wanted the extension. Prefix
every option with the runner's own name, as `--cookbook-ref-precision` does.
