# Progress Monitor

+ Steps to run.

```bash
python3 main.py
```

| Case | What it shows |
| --- | --- |
| `case_list` | `CookbookProgressMonitor` printing each build step as a line |
| `case_animation` | the same phases hand-drawn with terminal escape sequences |
| `case_tqdm` | the same phases as nested `tqdm` bars |
| `case_cancel` | cancelling a build from `step_complete` |

## `case_tqdm` — nested progress bars

Adapted from `TQDMProgressMonitor` in `utilities.py` of
[Stable-Diffusion-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT)
(Apache-2.0), where it is what a user watching a Stable Diffusion engine build actually sees.
`case_animation` hand-draws the same information with escape sequences; this is the version to
reach for when a progress-bar library is already available — one `tqdm` per phase, `position` set
to the nesting depth.

Two details that are not decoration:

+ **`max_indent`.** TensorRT nests phases several levels deep, and a bar per level is unreadable.
  Phases deeper than the limit are **not tracked at all** — note the early `return` in
  `phase_start`, which is why every other callback has to tolerate an unknown phase name.
+ **`leave=True` only for the outermost phase**, so the finished build leaves one line behind
  instead of a dozen.

`tqdm` is not a new dependency: `transformers`, already in `requirements.txt`, requires
`tqdm>=4.60`. The bars are written for a terminal, so in `log-main.py.log` they appear as escape
sequences — that is expected, not a bug.

## `case_cancel` — only `step_complete` can stop a build

`IProgressMonitor` has three callbacks and **only `step_complete` returns a value**. Returning
`False` from it aborts the build:

```txt
[TRT] [E] IBuilder::buildSerializedNetwork: Error Code 7: BuildCanceled (Build canceled by user)
Cancelled after 5 steps: build() returned False, engine_bytes is None
```

This matters because the upstream implementation gets it wrong. Its `phase_start` and
`phase_finish` handlers write

```python
except KeyboardInterrupt:
    _step_result = False      # a bare name: assigns a new local, never the attribute
```

so a Ctrl-C arriving during either callback is swallowed and the build runs to completion. The
version here writes `self._step_result`, which the next `step_complete` then returns. The comment
in the upstream source — *"The phase_start callback cannot directly cancel the build, so request
the cancellation from within step_complete"* — describes exactly the right design; the code just
does not implement it.

Measured on TensorRT 11.3.0.99, B200, 2026-10-12.
