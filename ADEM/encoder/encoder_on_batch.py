import tensorflow as tf

from .rnn_encoder import *


def check_encoder_format(encoder):
    if not isinstance(encoder, dict):
        raise TypeError('encoder should be a dict')
    for key, type_ in zip(['name', 'params'], [str, dict]):
        if key not in encoder:
            raise KeyError('encoder should have {}'.format(key))
        if not isinstance(encoder[key], type_):
            raise TypeError(
                'encoder {} should have type {}, now receive {}'.format(
                    key, type_, type(encoder[key])))


def get_encoder(encoder):
    encoder_func = globals()[encoder['name']]
    if encoder_func is None:
        raise ImportError('module {} not found !!!'.format(encoder['name']))
    return encoder_func


def encoder_on_batch(batch_with_embedding, batch_mask,
                     encoder, output_dim, scope_name=None):
    check_encoder_format(encoder)
    encoder_func = get_encoder(encoder)
    batch_size = tf.shape(batch_with_embedding, )[0]
    idx = tf.constant(0)
    output = tf.zeros([1, output_dim], dtype=tf.float32)
    condition_func = lambda idx, output: idx < batch_size

    def body_func(idx, output):
        encoder['params']['input_with_embedding'] = batch_with_embedding[idx]
        encoder['params']['mask'] = batch_mask[idx]
        encoder['params']['scope_name'] = scope_name
        encoder_result = encoder_func(**encoder['params'])
        return [idx + 1, tf.concat([output, encoder_result], axis=0)]

    _, batch_output = tf.while_loop(
        condition_func, body_func,
        loop_vars=[idx, output],
        shape_invariants=[idx.get_shape(), tf.TensorShape([None, output_dim])])

    return tf.slice(batch_output, [1, 0], [batch_size, output_dim])
