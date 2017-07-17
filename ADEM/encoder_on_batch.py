import tensorflow as tf

from .rnn_encoder import *


def encoder_on_batch(batch_with_embedding, batch_mask,
                     encoder_name, encoder_params):

    encoder_func = globals()[encoder_name]
    batch_size = batch_with_embedding.get_shape().as_list()[0]
    output_dim = encoder_params['context_level_state_size']
    idx = tf.constant(0)
    output = tf.zeros([1, output_dim], dtype=tf.float32)
    condition_func = lambda idx, output: idx < batch_size

    def body_func(idx, output):
        encoder_params['input_with_embedding'] = batch_with_embedding[idx]
        encoder_params['mask'] = batch_mask[idx]
        encoder_result = encoder_func(**encoder_params)
        return [idx + 1, tf.concat([output, encoder_result], axis=0)]

    result = tf.while_loop(
        condition_func, body_func,
        loop_vars=[idx, output],
        shape_invariants=[idx.get_shape(), tf.TensorShape([None, output_dim])])

    return tf.slice(result[1], [1, 0], [batch_size, output_dim])
