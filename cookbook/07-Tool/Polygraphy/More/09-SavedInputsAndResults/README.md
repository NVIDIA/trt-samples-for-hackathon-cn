# Saved inputs and results

+ Persist inference inputs and outputs, and bridge the CLI and the Python API.

+ Steps to run.

```bash
python3 main.py
```

## Two formats

| what | type | CLI flag that writes it |
| --- | --- | --- |
| inputs | `List[Dict[str, np.ndarray]]`, one feed dict per iteration | `--save-inputs` |
| outputs | `RunResults`, runner name → list of `IterationResult` | `--save-outputs` |

```python
results = Comparator.run([...], save_inputs_path="inputs.json")
results.save("outputs.json")

inputs  = load_json("inputs.json")      # plain object -> load_json
results = RunResults.load("outputs.json")  # Polygraphy type -> its own load
```

`load_json` is for plain objects; Polygraphy's own types carry `save` / `load`.
The round trip is bit-exact for both outputs of the MNIST model.

```
inputs.json: 4459 B
outputs.json: 1372 B
```

The inputs file is the larger one — it holds a 28×28 image while the outputs hold
10 logits and a class index.

## The bridge: the CLI writes the same files

```
CLI runner name : 'trt-runner-N0-08/27/26-00:52:01'   <- timestamped, do not hard-code it
output keys     : ['y', 'z']
```

`RunResults.load` reads what `polygraphy run --save-outputs` wrote. That is the
practical reason to know the format: produce results with the CLI on one machine,
analyse them in Python somewhere else. The upstream example only shows the
loading half, with files that appear from nowhere.

Note the runner name — the CLI stamps it with a timestamp, so reading CLI output
means taking `list(results.keys())[0]` rather than assuming a name.

## Merging runs that never shared a process

```
CLI and Python got identical inputs without coordinating: True
merged runners: ['python', 'cli']
python vs cli within 1e-5: True
```

`RunResults.add(iterations, runner_name=...)` builds an object holding both runs,
which `Comparator.compare_accuracy` then treats as two runners.

The first line is the one worth noticing: **Polygraphy's default data loader is
deterministic**, so two independent processes generated bit-identical inputs
without being told to. That is what makes the comparison meaningful rather than
vacuous — and it is verified in the case rather than assumed. For a real dataset,
hand the saved file to the CLI with `--load-inputs` instead of relying on it.

## Related

+ `../../Run/` — the CLI side of `--save-inputs` / `--save-outputs`.
+ [`../02-ComparingBackends/`](../02-ComparingBackends/README.md) — what to do with a `RunResults` once you have one.
+ [`../08-ValidatingOnADataset/`](../08-ValidatingOnADataset/README.md) — why `RunResults` is the wrong container for a large dataset.
