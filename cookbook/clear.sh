rm -rfv 00*/__pycache__/
rm -rfv 00*/test/*.jpg
rm -rfv 00*/train/*.jpg
#rm -rfv 00*/*.gz

rm -rfv 01*/*/*.d
rm -rfv 01*/*/*.o
rm -rfv 01*/*/*.exe
rm -rfv 01*/*/*.plan

rm -rfv 03*/*/__pycache__/
rm -rfv 03*/*/*.plan
rm -rfv 03*/*/*.cache
rm -rfv 03*/*/*.npz

rm -rfv 04*/*/__pycache__/
rm -rfv 04*/*/*.pb
rm -rfv 04*/*/*.pt
rm -rfv 04*/*/*.onnx
rm -rfv 04*/*/*.uff
rm -rfv 04*/*/*.plan
rm -rfv 04*/*/*.cache

rm -rfv 05*/*/*.d
rm -rfv 05*/*/*.o
rm -rfv 05*/*/*.so
rm -rfv 05*/*/*.plan
rm -rfv 05*/*/*.exe
rm -rfv 05*/*/*.npz

rm -rfv 05*/PluginReposity/*/*.d
rm -rfv 05*/PluginReposity/*/*.o
rm -rfv 05*/PluginReposity/*/*.so
rm -rfv 05*/PluginReposity/*/*.plan
rm -rfv 05*/PluginReposity/*/*.exe

rm -rfv 05*/PluginReposity/*/*/*.d
rm -rfv 05*/PluginReposity/*/*/*.o
rm -rfv 05*/PluginReposity/*/*/*.so
rm -rfv 05*/PluginReposity/*/*/*.plan
rm -rfv 05*/PluginReposity/*/*/*.exe

rm -rfv 06*/*/*.pb
rm -rfv 06*/*/*.pt
rm -rfv 06*/*/*.onnx
rm -rfv 06*/*/*.plan
rm -rfv 06*/*/*.d
rm -rfv 06*/*/*.o
rm -rfv 06*/*/*.so

rm -rfv 07*/TFTRT/*.pb
rm -rfv 07*/TFTRT/*Model/
rm -rfv 07*/Torch-TensorRT/*.ts

rm -rfv 08*/*/*.qdrep
rm -rfv 08*/*/*.nsys-rep
rm -rfv 08*/*/*.pb
rm -rfv 08*/*/*.onnx
rm -rfv 08*/*/*.plan
rm -rfv 08*/*/*/*.pb
rm -rfv 08*/*/*/*.onnx
rm -rfv 08*/*/*/*.plan

rm -rfv 09*/*/*.qdrep
rm -rfv 09*/*/*.nsys-rep

rm -rfv 13*/*

# 仅供 Hackathon Github 上传时使用
#rm -rfv 05*/loadDataFromNpz
#rm -rfv 05*/PluginReposity/LayerNormPlugin/*
#rm -rfv 06*/pyTorch-LayerNorm/*

