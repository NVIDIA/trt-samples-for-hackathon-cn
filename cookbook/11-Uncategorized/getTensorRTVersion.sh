echo "CUDA version:"
echo $(nvcc --version | grep -o cuda_.*/)
echo "cuDNN version:"
cat /usr/include/x86_64-linux-gnu/cudnn_version_v*.h | grep -Eo "CUDNN_.* \<[0-9]+\>\$"
echo "libnvinfer.so version:"
nm -D /usr/lib/x86_64-linux-gnu/libnvinfer.so | grep -o tensorrt_version_.*
echo "python tensorrt version:"
python -c "import tensorrt as trt; print(trt.__version__)"
