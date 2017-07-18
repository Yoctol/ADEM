import tensorflow as tf

from ADEM.adem_score import *


class AdemScoreTest(tf.test.TestCase):

    def setUp(self):
        self.batch_size = 10
        self.context_dim = 200
        self.model_response_dim = 100
        self.reference_response_dim = 120
        self.context = tf.random_normal(
            shape=[self.batch_size, self.context_dim], name='context')
        self.model_response = tf.random_normal(
            shape=[self.batch_size, self.model_response_dim],
            name='model_response')
        self.reference_response = tf.random_normal(
            shape=[self.batch_size, self.reference_response_dim],
            name='reference_response')

    def test_compute_adem_score(self):
        with self.test_session() as sess:
            score, M, N = compute_adem_score(
                self.context, self.model_response, self.reference_response,
                mr_dim=self.model_response_dim,
                ct_dim=self.context_dim,
                rr_dim=self.reference_response_dim)
            sess.run(tf.global_variables_initializer())
            self.assertEqual(M.eval().shape, (self.model_response_dim,
                                              self.context_dim))
            self.assertEqual(N.eval().shape, (self.model_response_dim,
                                              self.reference_response_dim))
            self.assertEqual(score.eval().shape, (self.batch_size,))

    def test_tf_static_adem_score(self):
        with self.test_session() as sess:
            score, M, N = tf_static_adem_score(
                self.context, self.model_response, self.reference_response)
            sess.run(tf.global_variables_initializer())
            self.assertEqual(M.eval().shape, (self.model_response_dim,
                                              self.context_dim))
            self.assertEqual(N.eval().shape, (self.model_response_dim,
                                              self.reference_response_dim))
            self.assertEqual(score.eval().shape, (self.batch_size,))

    def test_tf_dynamic_adem_score(self):
        with self.assertRaises(TypeError):
            wrong_type_shape_info = ['a', 'b', 'c']
            tf_dynamic_adem_score(
                self.context, self.model_response, self.reference_response,
                wrong_type_shape_info)
        with self.assertRaises(KeyError):
            wrong_content_shape_info = {'rr_dim': 10, 'mr_dim': 20, 'a': 100}
            tf_dynamic_adem_score(
                self.context, self.model_response, self.reference_response,
                wrong_content_shape_info)

        with self.test_session() as sess:
            shape_info = {'mr_dim': self.model_response_dim,
                          'rr_dim': self.reference_response_dim,
                          'ct_dim': self.context_dim}
            score, M, N = tf_dynamic_adem_score(
                self.context, self.model_response, self.reference_response, shape_info)
            sess.run(tf.global_variables_initializer())
            self.assertEqual(M.eval().shape, (self.model_response_dim,
                                              self.context_dim))
            self.assertEqual(N.eval().shape, (self.model_response_dim,
                                              self.reference_response_dim))
            self.assertEqual(score.eval().shape, (self.batch_size,))

    # def test_matrix_l1_norm(self):
    #     with self.test_session() as sess:
    #         matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
    #         l1_norm = matrix_l1_norm(matrix)
    #         self.assertEqual(l1_norm.eval(), 9)

    # def test_adem_l1_loss(self):
    #     with self.test_session() as sess:
    #         human_score = tf.constant([1, 2, 3])
    #         model_score = tf.constant([1, 4, 3])
    #         M = tf.random_normal(shape=[100, 200])
    #         N = tf.random_normal(shape=[100, 120])
    #         loss_val = adem_l1_loss(human_score, model_score, M, N)
    #         self.assertEqual(loss_val.eval().dtype.name, 'float32')
