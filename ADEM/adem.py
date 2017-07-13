import tensorflow as tf


def adem_score(context, model_response, reference_response):
    # TODO assert context.shape[0] == model_response.shape[0] ==
    # reference_response.shape[0]
    ct = tf.shape(context)[1]
    mr = tf.shape(model_response)[1]
    rr = tf.shape(reference_response)[1]
    transposed_model_response = tf.transpose(model_response)
    with tf.variable_scope('score'):
        M = tf.get_variable(name='M', shape=[ct, mr], dtype=tf.float32)
        N = tf.get_variable(name='N', shape=[rr, mr], dtype=tf.float32)
        alpha = tf.get_variable(name='alpha',
                                shape=[1], initializer=tf.random_uniform(5.0))
        beta = tf.get_variable(name='beta',
                               shape=[1], initializer=tf.random_uniform(5.0))

        score = (tf.matmul(context, tf.matmul(M, transposed_model_response)) +
                 tf.matmul(reference_response, tf.matmul(N, transposed_model_response)) -
                 alpha) / beta

    return score, M, N


def matrix_l1_norm(matrix):
    abs_matrix = tf.abs(matrix)
    row_max = tf.reduce_max(abs_matrix, axis=1)
    return tf.reduce_sum(row_max)


def l1_loss(human_score, model_score, M, N):
    # TODO human_score.shape == model_score.shape
    loss = tf.reduce_sum(tf.square(human_score - model_score))
    regularization = matrix_l1_norm(M) + matrix_l1_norm(N)
    gamma = tf.constant(0.3, name='gamma's)
    return loss + (gamma * regularization)
