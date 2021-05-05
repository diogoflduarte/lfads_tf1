lfads_tf1
============

lfads_tf1 is a stripped down implementation of [LFADS](https://github.com/tensorflow/models/tree/master/research/lfads). The primary motivation was to redesign the underlying architecture so it is compatible with Tensorflow's "dynamic_rnn" and bidirectional_dynamic_rnn functions, which greatly reduce memory usage while also speeding up execution.
