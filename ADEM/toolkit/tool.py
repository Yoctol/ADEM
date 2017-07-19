import tensorflow as tf


def cast_to_float32(tensor_list):
    for num, tensor in enumerate(tensor_list):
        tensor_list[num] = tf.cast(tensor, tf.float32)
    return tensor_list


def get_last_effective_result(input_, mask):
    [input_] = cast_to_float32([input_])
    input_shape = tf.shape(input_, )
    zero_one_embedding = tf.concat([tf.zeros([1, input_shape[2]]),
                                    tf.ones([1, input_shape[2]])], axis=0)
    last_word_idx = tf.one_hot(
        indices=mask - 1, depth=input_shape[1], dtype=tf.int32)
    last_word_idx_with_embedding = tf.nn.embedding_lookup(
        zero_one_embedding, last_word_idx)
    return tf.reduce_sum(input_ * last_word_idx_with_embedding, axis=1)
