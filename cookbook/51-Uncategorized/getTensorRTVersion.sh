echo "+ CUDA version:"
echo $(nvcc --version | grep -o cuda_.*/)
echo
echo "+ cuDNN version:"
cat /usr/include/x86_64-linux-gnu/cudnn_version_v*.h | grep -Eo "CUDNN_.* \<[0-9]+\>\$"
echo
echo "+ libnvinfer.so version:"
nm -D /usr/lib/x86_64-linux-gnu/libnvinfer.so | grep -o tensorrt_version_.*
echo 
echo "+ python tensorrt version:"
python3 -c "import tensorrt as trt; print(trt.__version__)"
