# common
find . -name "*.d" | xargs rm -rfv
find . -name "*.o" | xargs rm -rfv
find . -name "*.so" | xargs rm -rfv
find . -name "*.exe" | xargs rm -rfv
find . -name "*.plan" | xargs rm -rfv
find . -name "*.cache" | xargs rm -rfv
find . -name "*.tacitc" | xargs rm -rfv

find . -name "*.pb" | xargs rm -rfv
find . -name "*.onnx" | xargs rm -rfv

find . -name __pycache__ | xargs rm -rfv

# specific directory
rm -rfv 00-MNISTData/__pycache__/
rm -rf  00-MNISTData/test/*.jpg
rm -rf  00-MNISTData/train/*.jpg

rm -rfv 03-APIModel/*/__pycache__/
rm -rfv 03-APIModel/*/*.npz
rm -rfv 03-APIModel/*/*/*.npz

rm -rfv 04-Parser/*/__pycache__/

rm -rfv 04-Parser/Paddlepaddle-ONNX-TensorRT/paddle_model_static_onnx_temp_dir/
rm -rfv 04-Parser/*/*.npz
rm -rfv 04-Parser/*/*/*.npz
rm -rfv 04-Parser/*/*.onnx
rm -rfv 04-Parser/TensorFlow1-Caffe-TensorRT/checkpoint
rm -rfv 04-Parser/TensorFlow1-Caffe-TensorRT/model.*
rm -rfv 04-Parser/TensorFlow1-ONNX-TensorRT-QAT/model/
rm -rfv 04-Parser/TensorFlow1-UFF-TensorRT/model.uff
rm -rfv 04-Parser/TensorFlow2-ONNX-TensorRT/model-*/

rm -rfv 05-Plugin/loadNpz/data.npz

rm -rfv 06-PluginAndParser/TensorFlow2-AddScalar/model/

rm -rfv 07-FrameworkTRT/TensorFlow1-TFTRT/TFModel/
rm -rfv 07-FrameworkTRT/TensorFlow1-TFTRT/TRTModel/
rm -rfv 07-FrameworkTRT/TensorFlow2-TFTRT/TFModel/
rm -rfv 07-FrameworkTRT/TensorFlow2-TFTRT/TRTModel/
rm -rfv 07-FrameworkTRT/Torch-TensorRT/model.ts

rm -rfv 08-Tool/Polygraphy/inspectExample/polygraphy_capability_dumps/
rm -rfv 08-Tool/trex/.ipynb_checkpoints/


rm -rfv 10*/*/*.onnx
rm -rfv 10*/*/*.plan

rm -rfv 11*/*/*.onnx
rm -rfv 11*/*/*.plan

rm -rfv 09-Advance/*/*.npz
