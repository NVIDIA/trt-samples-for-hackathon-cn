# Writing a command-line tool

+ Build a CLI tool on Polygraphy's `Tool` base class and its argument groups.

+ Steps to run.

```bash
python3 main.py          # the cases below
./plan-size /path/to/model.onnx --version-compatible
```

`plan-size` in this directory is the tool — 50 lines that build an engine and
report the build time and plan size. Everything else here is about what the base
class does and does not do for you.

## What you get for free

```
Minimal       :   7 options from 0 subscription(s)
WithDataLoader:  15 options from 1 subscription(s)
WithTheLot    :  95 options from 9 subscription(s)
```

The empty tool is not empty: `LoggerArgs` is subscribed by the base class, so
`-v`, `--silent`, `--log-file` and friends are there before you write anything.
Every `get_subscriptions_impl` entry after that brings its own options, already
parsed into the argument-group instance and already wired to the loader that
consumes them.

`plan-size` declares exactly one option of its own (`--json`) and accepts every
build flag `polygraphy convert` does:

```bash
./plan-size model-trained.onnx --version-compatible
[I] Built in 13.02 s, plan is 118034692 bytes (118.035 MB)
```

`--version-compatible` appears nowhere in `plan-size`; `TrtConfigArgs` brought
it. That is the same 118 MB measured in
[`../17-VersionCompatibility/`](../17-VersionCompatibility/README.md), from a
tool that knows nothing about it.

## The dependency graph is a docstring

Argument groups depend on each other, and the dependencies live only in prose:

```
OnnxLoadArgs             depends on ['ModelArgs', 'OnnxInferShapesArgs', 'OnnxSaveArgs', 'OnnxFromTfArgs']
TrtConfigArgs            depends on ['DataLoaderArgs', 'ModelArgs']
TrtLoadNetworkArgs       depends on ['ModelArgs', 'TrtLoadPluginsArgs', 'OnnxLoadArgs', 'TrtOnnxFlagArgs', 'TrtConfigArgs']
TrtLoadEngineBytesArgs   depends on ['ModelArgs', 'TrtLoadPluginsArgs', 'TrtLoadNetworkArgs', 'TrtConfigArgs', 'TrtSaveEngineBytesArgs']
```

Nothing validates that you subscribed to them. Miss one and:

```
a tool missing OnnxInferShapesArgs parses its arguments fine: model_file=model-trained.onnx
...and then, at run time: KeyError: <class 'polygraphy.tools.args.backend.onnx.loader.OnnxInferShapesArgs'>
```

Argument parsing succeeds, the tool starts, the model is read, and only then does
`ArgGroups.__getitem__` raise a bare `KeyError` seven frames down with a class
object as the message. On a real model that is several seconds of apparent
progress first. The class name in the `KeyError` is the only hint about what to
add.

## `main()` calls `sys.exit`

`Tool.main()` is four public calls plus an exit:

```python
parser = tool.setup_parser()
args = parser.parse_args([onnx_file, "--trt-min-shapes", "x:[1,1,28,28]"])
tool.parse(args)
status = tool.run(args)
```

Doing it by hand is how you test a tool in-process, and how you pass a fixed
argument list instead of `sys.argv`:

```
parsed without touching sys.argv: min shapes {'x': ([1, 1, 28, 28], [1, 1, 28, 28], [1, 1, 28, 28])}
run() -> 0 (run_impl returned 0)
a run_impl returning None still exits 0: run() -> 0
```

## Two ways to write a tool that will not start

```
neither run nor run_impl -> NotImplementedError: run_impl() must be implemented by child classes
no class docstring     -> AssertionError: No help output was provided for this tool!
```

The base class defines `run` as a wrapper around `run_impl`. Upstream's
`examples/dev/01_writing_cli_tools/gen-data` overrides `run` directly — that
still works, because `main` calls `run`, but it skips the wrapper that logs
module versions and defaults a `None` status to `0`. Write `run_impl`.

The docstring assert means `python3 -O` (which strips asserts) gives you an empty
help output instead of an error.

## Your tool stays standalone

```
tools in polygraphy's registry: ['run', 'convert', 'inspect', 'check', 'surgeon', 'template', 'debug', 'data', 'plugin', 'multi-device']
`polygraphy plan-size` -> exit 2: polygraphy: error: argument tools: invalid choice: 'plan-size'
```

`polygraphy/tools/registry.py` is a flat list of `try_register_tool(...)` calls
evaluated at import: no entry point, no plugin hook, no environment variable. The
only extension point in the whole CLI is `polygraphy.run.plugins`, which adds
runners to `polygraphy run` and nothing else — see
[`../19-CustomBackend/`](../19-CustomBackend/README.md).

So a `Tool` subclass is always its own executable. That is fine, and it is what
upstream's example does; it is just not what "to integrate a tool into
Polygraphy, you will need to add it to the registry" in that example's closing
comment sounds like — it means editing the installed package.
