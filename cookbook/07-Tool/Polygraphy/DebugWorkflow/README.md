# Debug Workflow

+ Three debugging workflows that Polygraphy and trtexec make possible but do not give you: an error scatter plot, an auditable `debug reduce`, and a bridge for the inputs between the two tools.

+ Steps to run.

```bash
python3 accuracy_scatterplot.py   # is the error spread, or a few outliers?
python3 inputs_to_trtexec.py      # polygraphy --save-inputs  ->  trtexec --loadInputs
python3 autoreduce.py             # keep every reduction step, not just the answer
```

These are **from-scratch reimplementations** of the ideas behind five internal scripts
(`accuracy_scatterplot.py`, `polygraphy_autoreduce.py`, `polygraphy_inputs_to_trtexec.py`,
`accuracy_analyzer.py`, `dump_io_data.py`). Those are `LicenseRef-NvidiaProprietary`; no code was
taken from them. Measured on B200, TensorRT 11.1.0.106, polygraphy 0.50.3.

## `accuracy_scatterplot.py` — one max-abs-diff is not an accuracy report

`max |a - b| = 0.05` cannot distinguish the only two things you want to know apart:

+ error **spread** over every element — reduced precision working as intended, fix is more precision;
+ error **driven by a few outliers** — a saturating value, a badly scaled channel, one op that should
  not have been demoted. Fixable without giving up the speed.

The script builds two synthetic error patterns with the **exact same maximum** and shows they are
nothing alike:

| pattern | max | mean | elements wrong | top-1% share of total error | verdict |
| ------- | --: | ---: | -------------: | --------------------------: | ------- |
| spread | 0.050000 | 2.64e-02 | 80 / 80 | 2.4% | spread |
| outlier | 0.050000 | 6.25e-04 | **1 / 80** | **100.0%** | outlier-driven |

Same headline number, mean differing by 42x. The **top-1% share** — what fraction of the total error
lives in the worst 1% of elements — is the statistic that separates them: it tends to 1% for evenly
spread error and to 100% when one element carries everything. The scatter plot
(`result-accuracy_scatterplot.png`) shows the same thing as a horizontal band versus a few points
far above the rest.

The real TensorRT-vs-ONNX-Runtime comparison on the MNIST model is also run, for reference: 51/80
elements differ, max 7.5e-09, top-1% share 5.3% — textbook spread, which is what a correct fp32
engine should look like.

## `inputs_to_trtexec.py` — the two tools do not speak the same language

`polygraphy run --save-inputs` writes **one JSON** with names, shapes and dtypes.
`trtexec --loadInputs` wants **one raw binary per tensor**, with none of that. Four things must be
right, and the script checks each:

1. **C-contiguous bytes** — `np.ascontiguousarray` before `.tofile`, or a sliced/transposed array is
   written in the wrong order.
2. **Dtype preserved** — recorded in the file name, since the format cannot carry it.
3. **Byte count** — trtexec *does* check this, and says so clearly. But it compares against the
   **binding** size, so forgetting `--shapes` rejects a perfectly good file:
   `Input binding size is: 3136 bytes but the file size is 12544 bytes`.
4. **The shape syntax differs.** Polygraphy writes `x:[4,1,28,28]`, trtexec wants `x:4x1x28x28`.

Point 4 is the expensive one, because of *how* trtexec rejects it:

```txt
trtexec --loadInputs=x:x.4.1.28.28.float32.raw --shapes=x:[4,1,28,28] ...
  -> exit=1, and trtexec prints its entire help text
```

No error message, no mention of which argument — just the help, which reads like "trtexec is broken"
rather than "one argument was malformed". So the script emits **both** arguments in trtexec's syntax,
runs trtexec for real to prove the conversion works, and then re-runs it with Polygraphy's syntax to
put the failure mode in the log.

## `autoreduce.py` — make `debug reduce` auditable

`polygraphy debug reduce` writes exactly one file: the final `reduced.onnx`. Two questions cannot be
answered from it, and both matter:

+ *Did the reduction follow the failure I care about?* Reduction is driven by a pass/fail command,
  and a command that fails for a **second** reason — a typo, a missing file, an OOM — reduces just as
  neatly to something irrelevant.
+ *How did it get there?*

Polygraphy has the mechanism, it is just off by default: `--artifacts polygraphy_debug.onnx` names
the per-iteration file and `--art-dir` collects them; `--save-debug-replay` records the journal.
On `model-unknown.onnx` (5 nodes) that turns one output into a trail:

```txt
original : 5 nodes {'Identity': 3, 'UnknownNode1': 1, 'UnknownNode2': 1}
reduced  : 1 nodes {'UnknownNode1': 1}
kept 3 intermediate model(s):
    .../N0_inputs.onnx : 1 nodes {'UnknownNode1': 1}
    .../N0_outputs.onnx: 3 nodes {'Identity': 2, 'UnknownNode1': 1}
    .../N1_outputs.onnx: 2 nodes {'Identity': 1, 'UnknownNode1': 1}
replay journal: 3 keys -> ['_N0_outputs', '_N1_outputs', '_N0_inputs']
```

**The option is `--save-debug-replay`, not `--save-replay`** — the shorter name fails with
`Unrecognized Options` and no suggestion.

Then the script runs the *same* reduction driven by a deliberately broken check (it asks for a
tensor that does not exist, so every candidate "fails"). On this 5-node model both reductions land
on the same node — and that coincidence **is** the lesson: the reduced model cannot tell you which
failure drove it. Only the recorded per-iteration check output can, which is precisely what
`--artifacts` and `--save-debug-replay` preserve.

## Related

+ [`../Debug/`](../Debug/README.md) — the `debug` subtool's own examples.
+ [`../Run/`](../Run/README.md) — cross-backend comparison, the source of the golden run here.
+ [`../../trtexec/`](../../trtexec/README.md) — `--loadInputs` / `--dumpRawBindingsToFile` and the
  JSON exports on the trtexec side.
+ [`../../OnnxGraphSurgeon/11_mark_output_to_bisect.py`](../../OnnxGraphSurgeon/README.md) — bisecting
  a precision bug by hand when `debug reduce` is too blunt.
