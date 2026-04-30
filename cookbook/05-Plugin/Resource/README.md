# Resource

+ Example of using TensorRT `IPluginResource` to share information between two `PluginV3` layers in one network.
+ `ResourceWriter` writes info into shared resource, then `ResourceReader` reads and prints it.

+ Steps to run.

```bash
make build
python3 main.py
./main.exe
```
