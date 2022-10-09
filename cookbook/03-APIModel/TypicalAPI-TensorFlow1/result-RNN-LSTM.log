
tf.keras.layers.LSTM 或 tf.keras.layers.LSTMCell + tf.keras.layers.RNN
Weight:
lstm/kernel:0 (7, 20)
lstm/recurrent_kernel:0 (5, 20)
lstm/bias:0 (20,)
check h1: True 3.7252903e-09 4.3136657e-07
check c1: True 7.450581e-09 4.717342e-07
check AllOutput: True 7.450581e-09 4.3136657e-07

tf.keras.layers.CuDNNLSTM -----------------------------------------
Weight:
lstm/kernel:0 (7, 20)
lstm/recurrent_kernel:0 (5, 20)
lstm/bias:0 (20,)
cu_dnnlstm/kernel:0 (7, 20)
cu_dnnlstm/recurrent_kernel:0 (5, 20)
cu_dnnlstm/bias:0 (40,)
check h1: True 4.4703484e-08 8.8860327e-07
check c1: True 8.940697e-08 8.876627e-07
check AllOutput: True 4.4703484e-08 1.4713461e-06

[tf.nn.rnn_cell.BasicLSTMCell / tf.contrib.rnn.BasicLSTMCell] + [tf.nn.static_rnn / tf.nn.dynamic_rnn]
Weight:
lstm/kernel:0 (7, 20)
lstm/recurrent_kernel:0 (5, 20)
lstm/bias:0 (20,)
cu_dnnlstm/kernel:0 (7, 20)
cu_dnnlstm/recurrent_kernel:0 (5, 20)
cu_dnnlstm/bias:0 (40,)
rnn/basic_lstm_cell/kernel:0 (12, 20)
rnn/basic_lstm_cell/bias:0 (20,)
check h1: True 2.9802322e-08 2.7105062e-07
check c1: True 1.1920929e-07 1.207336e-07
check AllOutput: True 5.9604645e-08 2.0149196e-06

[tf.nn.rnn_cell.LSTMCell / tf.contrib.rnn.LSTMCell] + [tf.nn.static_rnn / tf.nn.dynamic_rnn]
Weight:
lstm/kernel:0 (7, 20)
lstm/recurrent_kernel:0 (5, 20)
lstm/bias:0 (20,)
cu_dnnlstm/kernel:0 (7, 20)
cu_dnnlstm/recurrent_kernel:0 (5, 20)
cu_dnnlstm/bias:0 (40,)
rnn/basic_lstm_cell/kernel:0 (12, 20)
rnn/basic_lstm_cell/bias:0 (20,)
rnn/lstm_cell/kernel:0 (12, 20)
rnn/lstm_cell/bias:0 (20,)
rnn/lstm_cell/projection/kernel:0 (5, 5)
check h1: True 5.9604645e-08 1.5783212e-07
check c1: True 1.4901161e-08 7.548726e-08
check AllOutput: True 7.4505806e-08 6.081728e-07

tf.contrib.cudnn_rnn.CudnnLSTM ------------------------------------
Weight:
lstm/kernel:0 (7, 20)
lstm/recurrent_kernel:0 (5, 20)
lstm/bias:0 (20,)
cu_dnnlstm/kernel:0 (7, 20)
cu_dnnlstm/recurrent_kernel:0 (5, 20)
cu_dnnlstm/bias:0 (40,)
rnn/basic_lstm_cell/kernel:0 (12, 20)
rnn/basic_lstm_cell/bias:0 (20,)
rnn/lstm_cell/kernel:0 (12, 20)
rnn/lstm_cell/bias:0 (20,)
rnn/lstm_cell/projection/kernel:0 (5, 5)
cudnn_lstm/opaque_kernel:0 (280,)
check h1: True 1.4901161e-08 1.895603e-07
check c1: True 2.9802322e-08 2.2503906e-07
check AllOutput: True 2.9802322e-08 3.9563696e-07

[tf.contrib.rnn.LSTMBlockCell / tf.contrib.rnn.LSTMBlockFusedCell] + [tf.nn.static_rnn / tf.nn.dynamic_rnn]
Weight:
lstm/kernel:0 (7, 20)
lstm/recurrent_kernel:0 (5, 20)
lstm/bias:0 (20,)
cu_dnnlstm/kernel:0 (7, 20)
cu_dnnlstm/recurrent_kernel:0 (5, 20)
cu_dnnlstm/bias:0 (40,)
rnn/basic_lstm_cell/kernel:0 (12, 20)
rnn/basic_lstm_cell/bias:0 (20,)
rnn/lstm_cell/kernel:0 (12, 20)
rnn/lstm_cell/bias:0 (20,)
rnn/lstm_cell/projection/kernel:0 (5, 5)
cudnn_lstm/opaque_kernel:0 (280,)
rnn/tf-contrib-rnn-LSTMBlockCell-LSTM/kernel:0 (12, 20)
rnn/tf-contrib-rnn-LSTMBlockCell-LSTM/bias:0 (20,)
check h1: True 2.9802322e-08 1.267692e-07
check c1: True 2.9802322e-08 9.9891636e-08
check AllOutput: True 2.9802322e-08 4.6843606e-06

test finish!
