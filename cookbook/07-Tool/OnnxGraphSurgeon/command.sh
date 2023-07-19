rm -rf ./*.onnx ./*.log

# 01-Create a ONNX graph with Onnx Graphsurgeon
python3 01-CreateModel.py

# 02-Add a new node into the graph
python3 02-AddNode.py

# 03-Remove a node in the graph
python3 03-RemoveNode.py

# 04-Replace a node in the graph
python3 04-ReplaceNode.py

# 05-rint the information of the graph
python3 05-PrintGraphInformation.py > result-05.log

# 06-Constant-Fold, Cleanup and Toposort of the graph
python3 06-Fold.py

# 07-Shape related operation
python3 07-ShapeOperationAndSimplify.py

# 08-Clip the graph into subgraph
python3 08-IsolateSubgraph.py

# 09-Use gs.Graph.register() to create the graph
python3 09-BuildModelWithAPI.py

# 10-Use advance API to work on ONNX files, wili's ONNX tool box.
python3 10-AdvanceAPI.py
