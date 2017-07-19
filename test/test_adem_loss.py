import tensorflow as tf

from ADEM.adem.adem_loss import *


class AdemLossTest(tf.test.TestCase):

    def test_matrix_l1_norm(self):
        with self.test_session() as sess:
            matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
            l1_norm = matrix_l1_norm(matrix)
            self.assertEqual(l1_norm.eval(), 9)

    def test_compute_adem_l1_loss(self):
        with self.test_session() as sess:
            human_score = tf.constant([1, 2, 3])
            model_score = tf.constant([1, 4, 3])
            M = tf.random_normal(shape=[100, 200])
            N = tf.random_normal(shape=[100, 120])
            loss_val = compute_adem_l1_loss(human_score, model_score, M, N)
            self.assertEqual(loss_val.eval().dtype.name, 'float32')

    def test_tf_static_adem_l1_loss(self):
        with self.test_session() as sess:
            human_score = tf.constant([1, 2, 3])
            model_score = tf.constant([1, 4, 3])
            M = tf.random_normal(shape=[100, 200])
            N = tf.random_normal(shape=[100, 120])
            loss_val = tf_static_adem_l1_loss(human_score, model_score, M, N)
            self.assertEqual(loss_val.eval().dtype.name, 'float32')
        # TODO test assertion
