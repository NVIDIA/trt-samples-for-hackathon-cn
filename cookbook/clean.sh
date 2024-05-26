# common
find . -name "*.d" | xargs rm -rfv
find . -name "*.o" | xargs rm -rfv
find . -name "*.so" | xargs rm -rfv
find . -name "*.exe" | xargs rm -rfv
find . -name "*.trt" | xargs rm -rfv
find . -name "*.cache" | xargs rm -rfv
find . -name "*.tacitc" | xargs rm -rfv
find . -name "*.raw" | xargs rm -rfv
find . -name "*.log" | xargs rm -rfv

find . -name "*.pb" | xargs rm -rfv
find . -name "*.onnx" | xargs rm -rfv
find . -name "*.weight" | xargs rm -rfv

find . -name __pycache__ | xargs rm -rfv

# specific directory
rm -rfv 00-MNISTData/__pycache__/
#rm -rf  00-MNISTData/test/*.jpg
#rm -rf  00-MNISTData/train/*.jpg

rm -rfv 03-BuildEngineByTensorRTAPI/*/__pycache__/
rm -rfv 03-BuildEngineByTensorRTAPI/*/*.npz
rm -rfv 03-BuildEngineByTensorRTAPI/*/*/*.npz

rm -rfv 04-BuildEngineByONNXParser/*/__pycache__/

rm -rfv 04-BuildEngineByONNXParser/Paddlepaddle-ONNX-TensorRT/paddle_model_static_onnx_temp_dir/
rm -rfv 04-BuildEngineByONNXParser/*/*.npz
rm -rfv 04-BuildEngineByONNXParser/*/*/*.npz
rm -rfv 04-BuildEngineByONNXParser/*/*.onnx
rm -rfv 04-BuildEngineByONNXParser/TensorFlow1-Caffe-TensorRT/checkpoint
rm -rfv 04-BuildEngineByONNXParser/TensorFlow1-Caffe-TensorRT/model.*
rm -rfv 04-BuildEngineByONNXParser/TensorFlow1-ONNX-TensorRT-QAT/model/
rm -rfv 04-BuildEngineByONNXParser/TensorFlow1-UFF-TensorRT/model.uff
rm -rfv 04-BuildEngineByONNXParser/TensorFlow2-ONNX-TensorRT/model-*/

rm -rfv 05-Plugin/loadNpz/data.npz

rm -rfv 06-FrameworkTRT/TensorFlow1-TFTRT/TFModel/
rm -rfv 06-FrameworkTRT/TensorFlow1-TFTRT/TRTModel/
rm -rfv 06-FrameworkTRT/TensorFlow2-TFTRT/TFModel/
rm -rfv 06-FrameworkTRT/TensorFlow2-TFTRT/TRTModel/
rm -rfv 06-FrameworkTRT/Torch-TensorRT/model.ts

rm -rfv 07-Tool/Polygraphy/inspectExample/polygraphy_capability_dumps/
rm -rfv 07-Tool/trex/.ipynb_checkpoints/

rm -rfv 08-Advance/*/*.npz

rm -rfv 09*/*/*.onnx
rm -rfv 09*/*/*.trt

rm -rfv 10*/*/*.onnx
rm -rfv 10*/*/*.trt
