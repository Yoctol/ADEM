import tensorflow as tf


def adem_score(context, model_response, reference_response):
    rr_size, rr_dim = reference_response.get_shape().as_list()
    mr_size, mr_dim = model_response.get_shape().as_list()
    ct_size, ct_dim = context.get_shape().as_list()
    with tf.control_dependencies(
        [tf.assert_equal(rr_size, mr_size, message='responses size not equal'),
         tf.assert_equal(ct_size, mr_size, message='context response size not equal')]):
        with tf.variable_scope('score'):
            M = tf.Variable(name='M', initial_value=tf.random_normal(
                shape=[mr_dim, ct_dim]), dtype=tf.float32)
            N = tf.Variable(name='N', initial_value=tf.random_normal(
                shape=[mr_dim, rr_dim]), dtype=tf.float32)
            alpha = tf.Variable(
                name='alpha', initial_value=tf.random_uniform([1], maxval=5.0))
            beta = tf.Variable(
                name='beta', initial_value=tf.random_uniform([1], maxval=5.0))
            score = (tf.reduce_sum((context * tf.matmul(model_response, M)), axis=1) +
                     tf.reduce_sum((reference_response * tf.matmul(model_response, N)), axis=1) -
                     alpha) / beta
        return score, M, N


def matrix_l1_norm(matrix):
    abs_matrix = tf.abs(matrix)
    row_max = tf.reduce_max(abs_matrix, axis=1)
    return tf.reduce_sum(row_max)


def adem_l1_loss(human_score, model_score, M, N):
    # TODO human_score.shape == model_score.shape
    loss = tf.reduce_sum(tf.square(human_score - model_score))
    regularization = matrix_l1_norm(M) + matrix_l1_norm(N)
    gamma = tf.constant(0.3, name='gamma')
    return loss + (gamma * regularization)
