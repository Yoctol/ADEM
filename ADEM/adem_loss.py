import tensorflow as tf

from .tool import cast_to_float32

def matrix_l1_norm(matrix):
    matrix = tf.cast(matrix, tf.float32)
    abs_matrix = tf.abs(matrix)
    row_max = tf.reduce_max(abs_matrix, axis=1)
    return tf.reduce_sum(row_max)


def compute_adem_l1_loss(human_score, model_score, M, N):
    [human_score, model_score] = cast_to_float32([human_score, model_score])
    loss = tf.reduce_sum(tf.square(human_score - model_score))
    regularization = matrix_l1_norm(M) + matrix_l1_norm(N)
    gamma = tf.constant(0.3, name='gamma')
    return loss + (gamma * regularization)


def tf_static_adem_l1_loss(human_score, model_score, M, N):
    hs_shape = human_score.get_shape().as_list()
    ms_shape = model_score.get_shape().as_list()
    with tf.control_dependencies(
        [tf.assert_equal(len(hs_shape), 1, message='score should be 1D.'),
         tf.assert_equal(len(ms_shape), 1, message='score should be 1D.'),
         tf.assert_equal(hs_shape, ms_shape,
                         message='human and model scores should have an equal amount.')]):
        return compute_adem_l1_loss(human_score, model_score, M, N)
