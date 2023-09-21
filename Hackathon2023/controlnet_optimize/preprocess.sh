echo "preprocess"
cd plugin && mkdir build && cd build && cmake .. && make -j5
cd ../../flash_attention_v2 && make -j12
cd ../ && python3 export_onnx.py