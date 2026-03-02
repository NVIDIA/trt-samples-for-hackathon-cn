echo "preprocess"
cd plugin && mkdir build && cd build && cmake .. && make -j5
cd ../../ && python3 export_onnx.py
