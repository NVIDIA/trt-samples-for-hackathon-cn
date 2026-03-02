# Engine Inspector

+ Steps to run.

```bash
python3 main.py
```

+ `AddScalarPlugin.so` is compiled from `05-Plugin/BasicExample-V2-deprecated`.

+ Available environment variables

|                        Name                        |  Type  | Default value |                                                                Description                                                                |
| :------------------------------------------------: | :----: | :-----------: | :---------------------------------------------------------------------------------------------------------------------------------------: |
|             TRT_SHIM_OUTPUT_JSON_FILE              | string |      ""       |                                                    Path to save the captured JSON file                                                    |
|             TRT_SHIM_NVINFER_LIB_NAME              | string | libnvinfer.so |                                                     Intercepted TensorRT library name                                                     |
|                 TRT_SHIM_DUMP_API                  |  bool  |     false     |                                         Print enter and exit messages for every API function call                                         |
|               TRT_SHIM_PRINT_WELCOME               |  bool  |     false     |                                          Print Welcome to TensorRT Shim at the start of the run                                           |
|          TRT_SHIM_FORCE_SINGLE_THREAD_API          |  bool  |     false     |                  Lock every API call to enforce single-threaded execution. Ignored when TRT_SHIM_OUTPUT_JSON_FILE is set                  |
|      TRT_SHIM_INLINE_WEIGHTS_LOWER_EQUAL_THAN      |  int   |       8       |                 Inline weights into the JSON instead of a separate .bin when their size (in elements) is ≤ this threshold                 |
| TRT_SHIM_MARK_AS_RANDOM_WEIGHTS_GREATER_EQUAL_THAN |  int   |    max int    |                        Skip saving weights with an element count ≥ this threshold (they will be marked as random)                         |
|          TRT_SHIM_FLUSH_AFTER_EVERY_CALL           |  bool  |     false     |                             Flush captured calls to the file after every API call instead of aggregating them                             |
|             TRT_SHIM_SET_TACTIC_CACHE              | string |      ""       | Path to a tactic-cache file that will be loaded and applied to TensorRT’s IBuilderConfig so tactic selection stays consistent across runs |
