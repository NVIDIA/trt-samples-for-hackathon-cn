# Cumulative layer

+ Cumulative layer.

+ Steps to run.

```bash
python3 main.py
```

## Attributes

| Attribute   | Description                                                        | Default |
| ----------- | ------------------------------------------------------------------ | ------- |
| `op`        | Cumulative operation type. Currently supports `SUM` (prefix-sum). | N/A     |
| `exclusive` | Whether the operation is exclusive (vs. inclusive) cumulative.     | `false` |
| `reverse`   | Whether the cumulative operation applies in reverse direction.     | `false` |
