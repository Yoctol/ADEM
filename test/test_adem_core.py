import tensorflow as tf

from ADEM.adem_core import *


class AdemCoreTest(tf.test.TestCase):

    def test_adem_score(self):
        with self.test_session() as sess:
            context = tf.random_normal(shape=[10, 200], name='context')
            model_response = tf.random_normal(
                shape=[10, 100], name='model_response')
            reference_response = tf.random_normal(
                shape=[10, 120], name='reference_response')
            score, M, N = adem_score(
                context, model_response, reference_response)
            sess.run(tf.global_variables_initializer())
            self.assertEqual(M.eval().shape, (100, 200))
            self.assertEqual(N.eval().shape, (100, 120))
            self.assertEqual(score.eval().shape, (10,))

    def test_matrix_l1_norm(self):
        with self.test_session() as sess:
            matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
            l1_norm = matrix_l1_norm(matrix)
            self.assertEqual(l1_norm.eval(), 9)

    def test_adem_l1_loss(self):
        with self.test_session() as sess:
            human_score = tf.constant([1, 2, 3])
            model_score = tf.constant([1, 4, 3])
            M = tf.random_normal(shape=[100, 200])
            N = tf.random_normal(shape=[100, 120])
            loss_val = adem_l1_loss(human_score, model_score, M, N)
            self.assertEqual(loss_val.eval().dtype.name, 'float32')
