#/bin/bash

set -e
set -x
rm -rf *.onnx *.weight
#clear

# 01-Create a ONNX graph with Onnx Graphsurgeon
python3 01-create_model.py

# 02-Add a new node into the graph
python3 02-add_node.py

# 03-Remove a node in the graph
python3 03-remove_node.py

# 04-Replace a node in the graph
python3 04-replace_node.py

# 05-rint the information of the graph
python3 05-print_graph.py > result-05.log

# 06-Constant-Fold, Cleanup and Toposort of the graph
python3 06-fold.py

# 07-Shape related operation
python3 07-shape_operation_and_simplify.py

# 08-Clip the graph into subgraph
python3 08-isolate_subgraph.py

# 09-Use gs.Graph.register() to create the graph
python3 09-build_model_with_API.py

# 10-Use advance API to work on ONNX files, wili's ONNX tool box.
python3 10-advance_API.py
