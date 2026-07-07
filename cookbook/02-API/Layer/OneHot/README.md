# OneHot layer

+ OneHot layer. Produces a one-hot encoding of the `indices` input: `onValue` is written at every position selected by an index and `offValue` everywhere else. See `case_simple` in `main.py`.

+ Steps to run.

```bash
python3 main.py
```

+ Input / output data types.

| Item             | Data Type                                                    | Shape / Rank                                            |
| :--------------- | :---------------------------------------------------------- | :------------------------------------------------------ |
| indices          | `int32`                                                     | `[A0, ..., An]`                                         |
| values           | `T`: `int32`, `int64`, `float16`, `float32`, `bfloat16`, `bool` | rank-1 with exactly 2 elements `[offValue, onValue]`  |
| depth            | `int32`                                                     | rank-0 scalar `[d]` (MUST be a build-time constant)     |
| output           | same type `T` as values                                     | `[A0, ..., A_{axis-1}, d, A_{axis+1}, ..., An]`         |

+ Attributes.

| Name   | Description                                                    | Default                            | Range                                  |
| :----- | :------------------------------------------------------------ | :--------------------------------- | :------------------------------------- |
| `axis` | Dimension into which the depth `d` is inserted in the output. | set at construction (positional arg) | `[-rank(indices)-1, rank(indices)]`  |
