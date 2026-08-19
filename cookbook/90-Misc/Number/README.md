# Number

+ Get the range and layout information of floating data type.

+ Steps to run

```bash
python3 build-number-md.py
python3 build-number-picture.py
```

+ Some output files of typical data types are listed in `output/`, all of them generated (including
  `Integer.md`), so fixes belong in `build-number-md.py` rather than in the `.md`.
+ `mannuscript.md` is the cross-format summary table, checked against `torch` and `ml_dtypes`.
