# Build route

+ `IBuilderConfig.all_build_routes` and `IBuilderConfig.build_route`: the builder's internal tuning
  knobs, and the only API that reports TensorRT's **default build strategy**.

```bash
python3 main.py
```

Measured on B200, TensorRT 11.3.0.99.

## `all_build_routes` is a database, not a setting

It is a read-only JSON string — 44 KB of it — describing every knob the internal tuner accepts:

```txt
tuner_version = 2.21.55, 215 knobs

allowed values               knobs
['on', 'off']                  192
['<free-form>']                 14
['0', '1', '2']                  5
['0', '1', '2', '3']             3
['1', '2', '...']                1

of the 192 on/off knobs, 151 default to `on` and 41 to `off`
```

Each entry carries `option`, `allowed_values`, `default_value` and a line of `help`, and **all 215
have help text**:

```txt
-conv_use_long_w            default=on     Convert Conv [N, C, X, 1] to [N, C, 1, X] in CASK params for perf.
-conv_use_long_w:reshapes   default=off    Convert Conv [N, C, X, 1] to [N, C, 1, X] by inserting reshapes.
-reshape_ppg                default=on     Reshape propagation pass.
```

That list of defaults **is** the default build strategy. No other TensorRT API reports it.

## `tuner_version` is a finer fingerprint than the TensorRT version

| TensorRT | tuner | knobs |
| --- | --- | ---: |
| 11.1.0.106 | 2.19.45 | 209 |
| 11.3.0.99 | 2.21.55 | 215 |

Six knobs appeared between those releases while the API shape did not change. When two machines
disagree about a build and their TensorRT versions match, this is the next thing to compare.

Both `build_route` and `all_build_routes` arrived **mid-11.1**: absent in `11.1.0.2`, present in
`11.1.0.106`.

## Accepted is not applied

`build_route` takes a string of `-name=value` tokens. Setting it is accepted, reads back, and the
tuner echoes it into the log at INFO:

```txt
[TuningBuildRoute] Using build route: -conv_use_long_w=off
```

That echo is the only *direct* evidence a route reached the builder. Plan size is an indirect probe,
and a noisy one — five builds of an identical configuration here give two different sizes:

```txt
baseline plan size, 5 identical builds: [34860, 34860, 34476, 34476, 34476]
-> 2 distinct value(s), spread 384 B on a 34,476 B plan
```

So every comparison is made against a measured **band**, not a single number. Against that band:

```txt
route                                plan size   verdict
-conv_use_long_w=off                    34,860   inside the baseline band
-conv_lowering=off                      34,860   inside the baseline band
-reshape_ppg=off                        34,476   inside the baseline band
-transpose_ppg=off                      34,860   inside the baseline band
```

All four were echoed by the tuner, so they all arrived — and none of them moved the plan out of the
noise. Flipping the first 16 on/off knobs away from their defaults gives the same answer: **16
inside the band, 0 outside, 0 failures.**

A knob acts on a pattern that has to be *present*, and a 64×64 MatMul gives most of them nothing to
do. This is the same trap as [`../../02-API/BuilderFlag/`](../../02-API/BuilderFlag/README.md),
where every flag was accepted and only a few changed anything. To tune for real, sweep against
**your** network and compare latency, not plan size.

## Wrong input is handled asymmetrically

```txt
route                         built   echoed by tuner         message
-no_such_knob=on              False   -                       Build route knob '-no_such_knob' is not a valid knob...
garbage                       False   -                       Build route token 'garbage' must match format -name=value...
-conv_use_long_w              False   -                       Build route token '-conv_use_long_w' must match format...
-conv_use_long_w=maybe        True    -conv_use_long_w=maybe
```

The knob **name** is validated. The knob **value** is not.

`-conv_use_long_w=maybe` builds successfully, and the TensorRT logger reports, at INFO, that it is
*using* a value it is about to discard. The only complaint —

```txt
... invalid compilation option, ignored (invalid boolean value "maybe")
```

— comes out on raw **stderr** from the compiler underneath, not through `ILogger` at all. Any
program that routes TensorRT's log to a file never sees it. So a mistyped value leaves the knob at
its default and tells you it worked. Validate a route against `all_build_routes` yourself.

## Related

+ [`../../02-API/BuilderFlag/`](../../02-API/BuilderFlag/README.md) — the same "accepted is not
  applied" measurement, for `BuilderFlag`.
+ [`../BuilderOptimizationLevel/`](../BuilderOptimizationLevel/README.md),
  [`../TacticSource/`](../TacticSource/README.md) — the other two builder-tuning dials, both of
  which *do* change the plan.
+ [`../../07-Tool/trtexec/`](../../07-Tool/trtexec/README.md) — the CLI side. Its `--tuningCache`
  JSON also carries a `build_route` field, but that is a search *over* routes; this page is the API
  that defines them.
