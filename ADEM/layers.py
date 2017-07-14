import tensoflow as tf


def tf_multi_lstms(input_, mask,
               batch_size, state_size,
               keep_prob, num_layers, scope, init_state=None):

    def basic_lstm():
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            state_size, forget_bias=1.0, state_is_tuple=True)
        if keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=keep_prob)
        return lstm_cell

    multi_lstms = tf.contrib.rnn.MultiRNNCell(
        [basic_lstm() for _ in range(num_layers)], state_is_tuple=True)

    if init_state is None:
        init_state = multi_lstms.zero_state(batch_size, tf.float32)

    with tf.variable_scope(scope):
        hidden_output, hidden_state = tf.nn.dynamic_rnn(
            cell=multi_lstms,
            inputs=input_,
            sequence_length=mask,
            initial_state=init_state,
            dtype=tf.float32,
        )
    return hidden_output, hidden_state
