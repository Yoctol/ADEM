import tensorflow as tf


def multi_lstms(input_with_embedding, mask,
                batch_size, state_size,
                keep_prob, num_layers, scope_name='lstm',
                forget_bias=1.0, activation=tf.tanh,
                init_state=None):
    # input_with_embedding is batch_size * time_step * embedding_size

    def basic_lstm():
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=state_size, forget_bias=forget_bias,
            activation=activation, state_is_tuple=True)
        if keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=keep_prob)
        return lstm_cell

    with tf.variable_scope(scope_name):
        multi_lstms = tf.contrib.rnn.MultiRNNCell(
            [basic_lstm() for _ in range(num_layers)], state_is_tuple=True)

        if init_state is None:
            init_state = multi_lstms.zero_state(batch_size, tf.float32)

        hidden_outputs, hidden_states = tf.nn.dynamic_rnn(
            cell=multi_lstms,
            inputs=input_with_embedding,
            sequence_length=mask,
            initial_state=init_state,
            dtype=tf.float32,
        )

        #state num_layer * m, c state * batch_size (nested tuple)
        #output [batch_size, time_steps, state_size]
    return hidden_outputs, hidden_states
