import tensorflow as tf


def cast_to_float32(tensor_list):
    for num, tensor in enumerate(tensor_list):
        tensor_list[num] = tf.cast(tensor, tf.float32)
    return tensor_list
