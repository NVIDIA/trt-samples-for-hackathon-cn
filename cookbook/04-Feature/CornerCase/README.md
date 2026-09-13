# Corner Case

+ UINT8, BOOL, NaN and fan-out: the semantics that only show up when the types are unusual.

+ Steps to run.

```bash
python3 main.py
```

`02-API/Layer/*` covers each operation's API. This file is about the questions whose answers are
easy to assume and easy to get wrong, and which the rest of the cookbook did not reach (surveyed
2026-09-04). Measured on B200, TensorRT 11.1.0.106.

## UINT8 is an I/O type, not an arithmetic one

It exists so an engine can accept raw 8-bit image data without a host-side conversion — the first
thing the network must do is `Cast` it. Arithmetic straight on UINT8 is refused **at build time**,
which is the good outcome: a restriction enforced rather than silently producing wrong numbers.

```txt
UINT8 -> Cast -> Add(1): [0 1 2 3 4 5] -> [1. 2. 3. 4. 5. 6.]
UINT8 arithmetic       : build returned None
    invalid weights type of UInt8, permitted types: INT32, float, ...
```

Read the builder's own message rather than the exception that follows it. Letting the wrapper fail
reports `deserialize_cuda_engine(): incompatible function arguments` — the symptom of a `None` plan,
not the reason for it.

## A BOOL element is one **byte**

A comparison produces BOOL and `Select` consumes it. The footprint is the surprise: one byte per
element, not one bit, so a mask costs as much as an INT8 tensor and eight times a real bitset. That
matters when the mask *is* the big tensor. A BOOL also cannot be reduced directly — cast to INT32
to count how many elements passed.

## NaN: `MAX` and `MIN` silently drop it

`ISNAN` and `ISINF` work as expected, and `x == x` is `False` for NaN as IEEE-754 requires. The part
worth knowing is what a reduction does:

| reduction over `[5.0, 6.0, nan, 8.0]` | TensorRT | NumPy |
| --- | --- | --- |
| `SUM` | `nan` | `nan` |
| `MAX` | **8.0** | `nan` |
| `MIN` | **5.0** | `nan` |

**`SUM` propagates the NaN; `MAX` and `MIN` do not.** IEEE-754 permits either for min/max and
TensorRT takes the ignore-NaN option, so a NaN that is obvious in a sum vanishes completely from a
max-pool or a top-k style reduction.

**Do not use a max-reduction as a NaN detector.** Use `ISNAN` followed by a `SUM`. Only the row
containing the NaN is affected either way — a reduction destroys its own row, not the tensor.

## Fan-out costs reads, not copies

One tensor consumed by N layers — the shape of an attention head split, a multi-scale head, or a
residual reused far downstream:

| consumers | network layers | engine bytes |
| --------: | -------------: | -----------: |
| 1 | 2 | 12,468 |
| 2 | 4 | 14,180 |
| 4 | 8 | 17,468 |
| 8 | 16 | 77,332 |
| 16 | 32 | 151,684 |

Layer count grows one `Elementwise` per consumer, as written, but the **producer is not
duplicated** — reading a tensor many times does not materialise it many times. (The engine grows
because there are more consumer layers and their kernels, not because the source tensor was copied.)

## Related

+ [`../../90-Misc/Number/`](../../90-Misc/Number/README.md) — what each TensorRT data type can represent.
+ [`../../02-API/Layer/Unary/`](../../02-API/Layer/Unary/README.md),
  [`../../02-API/Layer/Reduce/`](../../02-API/Layer/Reduce/README.md),
  [`../../02-API/Layer/Select/`](../../02-API/Layer/Select/README.md) — the operations used here.
+ [`../../07-Tool/OnnxGraphSurgeon/11_mark_output_to_bisect.py`](../../07-Tool/OnnxGraphSurgeon/README.md)
  — finding *where* a NaN or an overflow first appears.
