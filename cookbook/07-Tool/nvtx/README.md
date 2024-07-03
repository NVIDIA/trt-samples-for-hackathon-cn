# nvtx

+ Use NVIDIA®Tools Extension SDK to add mark in timeline of Nsight systems.

+ Steps to run.

```bash
nsys profile \
    --force-overwrite=true \
    -o py \
    python3 main.py

make
nsys profile \
    --force-overwrite=true \
    -o cpp \
    ./main.exe
```
