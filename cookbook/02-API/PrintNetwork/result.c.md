   0->CONVOLUTION,in=1,out=1,(Unnamed Layer* 0) [Convolution]
	Input   0:(-1, 1, 28, 28),FLOAT,inputT0
	Output  0:(-1, 32, 28, 28),FLOAT,(Unnamed Layer* 0) [Convolution]_output
   1->ACTIVATION,in=1,out=1,(Unnamed Layer* 1) [Activation]
	Input   0:(-1, 32, 28, 28),FLOAT,(Unnamed Layer* 0) [Convolution]_output
	Output  0:(-1, 32, 28, 28),FLOAT,(Unnamed Layer* 1) [Activation]_output
   2->POOLING,in=1,out=1,(Unnamed Layer* 2) [Pooling]
	Input   0:(-1, 32, 28, 28),FLOAT,(Unnamed Layer* 1) [Activation]_output
	Output  0:(-1, 32, 14, 14),FLOAT,(Unnamed Layer* 2) [Pooling]_output
   3->CONVOLUTION,in=1,out=1,(Unnamed Layer* 3) [Convolution]
	Input   0:(-1, 32, 14, 14),FLOAT,(Unnamed Layer* 2) [Pooling]_output
	Output  0:(-1, 64, 14, 14),FLOAT,(Unnamed Layer* 3) [Convolution]_output
   4->ACTIVATION,in=1,out=1,(Unnamed Layer* 4) [Activation]
	Input   0:(-1, 64, 14, 14),FLOAT,(Unnamed Layer* 3) [Convolution]_output
	Output  0:(-1, 64, 14, 14),FLOAT,(Unnamed Layer* 4) [Activation]_output
   5->POOLING,in=1,out=1,(Unnamed Layer* 5) [Pooling]
	Input   0:(-1, 64, 14, 14),FLOAT,(Unnamed Layer* 4) [Activation]_output
	Output  0:(-1, 64, 7, 7),FLOAT,(Unnamed Layer* 5) [Pooling]_output
   6->SHUFFLE,in=1,out=1,(Unnamed Layer* 6) [Shuffle]
	Input   0:(-1, 64, 7, 7),FLOAT,(Unnamed Layer* 5) [Pooling]_output
	Output  0:(-1, 3136),FLOAT,(Unnamed Layer* 6) [Shuffle]_output
   7->CONSTANT,in=0,out=1,(Unnamed Layer* 7) [Constant]
	Output  0:(3136, 1024),FLOAT,(Unnamed Layer* 7) [Constant]_output
   8->MATRIX_MULTIPLY,in=2,out=1,(Unnamed Layer* 8) [Matrix Multiply]
	Input   0:(-1, 3136),FLOAT,(Unnamed Layer* 6) [Shuffle]_output
	Input   1:(3136, 1024),FLOAT,(Unnamed Layer* 7) [Constant]_output
	Output  0:(-1, 1024),FLOAT,(Unnamed Layer* 8) [Matrix Multiply]_output
   9->CONSTANT,in=0,out=1,(Unnamed Layer* 9) [Constant]
	Output  0:(1, 1024),FLOAT,(Unnamed Layer* 9) [Constant]_output
  10->ELEMENTWISE,in=2,out=1,(Unnamed Layer* 10) [ElementWise]
	Input   0:(-1, 1024),FLOAT,(Unnamed Layer* 8) [Matrix Multiply]_output
	Input   1:(1, 1024),FLOAT,(Unnamed Layer* 9) [Constant]_output
	Output  0:(-1, 1024),FLOAT,(Unnamed Layer* 10) [ElementWise]_output
  11->ACTIVATION,in=1,out=1,(Unnamed Layer* 11) [Activation]
	Input   0:(-1, 1024),FLOAT,(Unnamed Layer* 10) [ElementWise]_output
	Output  0:(-1, 1024),FLOAT,(Unnamed Layer* 11) [Activation]_output
  12->CONSTANT,in=0,out=1,(Unnamed Layer* 12) [Constant]
	Output  0:(1024, 10),FLOAT,(Unnamed Layer* 12) [Constant]_output
  13->MATRIX_MULTIPLY,in=2,out=1,(Unnamed Layer* 13) [Matrix Multiply]
	Input   0:(-1, 1024),FLOAT,(Unnamed Layer* 11) [Activation]_output
	Input   1:(1024, 10),FLOAT,(Unnamed Layer* 12) [Constant]_output
	Output  0:(-1, 10),FLOAT,(Unnamed Layer* 13) [Matrix Multiply]_output
  14->CONSTANT,in=0,out=1,(Unnamed Layer* 14) [Constant]
	Output  0:(1, 10),FLOAT,(Unnamed Layer* 14) [Constant]_output
  15->ELEMENTWISE,in=2,out=1,(Unnamed Layer* 15) [ElementWise]
	Input   0:(-1, 10),FLOAT,(Unnamed Layer* 13) [Matrix Multiply]_output
	Input   1:(1, 10),FLOAT,(Unnamed Layer* 14) [Constant]_output
	Output  0:(-1, 10),FLOAT,(Unnamed Layer* 15) [ElementWise]_output
  16->SOFTMAX,in=1,out=1,(Unnamed Layer* 16) [Softmax]
	Input   0:(-1, 10),FLOAT,(Unnamed Layer* 15) [ElementWise]_output
	Output  0:(-1, 10),FLOAT,(Unnamed Layer* 16) [Softmax]_output
  17->TOPK,in=1,out=2,(Unnamed Layer* 17) [TopK]
	Input   0:(-1, 10),FLOAT,(Unnamed Layer* 16) [Softmax]_output
	Output  0:(-1, 1),FLOAT,(Unnamed Layer* 17) [TopK]_output_1
	Output  1:(-1, 1),INT32,(Unnamed Layer* 17) [TopK]_output_2
Succeeded building serialized engine!
