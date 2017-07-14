import tensorflow as tf


def adem_score(context, model_response, reference_response):
    context = tf.cast(context, tf.float32)
    model_response = tf.cast(model_response, tf.float32)
    reference_response = tf.cast(reference_response, tf.float32)
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
    matrix = tf.cast(matrix, tf.float32)
    abs_matrix = tf.abs(matrix)
    row_max = tf.reduce_max(abs_matrix, axis=1)
    return tf.reduce_sum(row_max)


def adem_l1_loss(human_score, model_score, M, N):
    human_score = tf.cast(human_score, tf.float32)
    model_score = tf.cast(model_score, tf.float32)
    hs_shape = human_score.get_shape().as_list()
    ms_shape = model_score.get_shape().as_list()
    with tf.control_dependencies(
        [tf.assert_equal(len(hs_shape), 1, message='score should be 1D.'),
         tf.assert_equal(len(ms_shape), 1, message='score should be 1D.'),
         tf.assert_equal(hs_shape, ms_shape,
                         message='human and model scores should have an equal amount.')]):
        loss = tf.reduce_sum(tf.square(human_score - model_score))
        regularization = matrix_l1_norm(M) + matrix_l1_norm(N)
        gamma = tf.constant(0.3, name='gamma')
    return loss + (gamma * regularization)
