import tensorflow as tf

from ADEM.toolkit.layers import *


class LayersTest(tf.test.TestCase):

    def test_multi_lstms(self):
        with self.test_session() as sess:
            input_with_embedding = tf.constant(
                [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                 [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15], [4, 1, 2, 6, 7], [0, 0, 0, 0, 0]],
                 [[10, 20, 30, 40, 50], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
            input_with_embedding = tf.cast(input_with_embedding, tf.float32)
            hidden_outputs, hidden_states = multi_lstms(
                input_with_embedding=input_with_embedding,
                mask=[2, 3, 1], batch_size=3, state_size=100,
                keep_prob=0.9, num_layers=5, scope_name='lstm',
                forget_bias=1.0, activation=tf.tanh, init_state=None)
            sess.run(tf.global_variables_initializer())
            self.assertEqual((3, 4, 100), hidden_outputs.eval().shape)
            self.assertEqual(5, len(hidden_states))
            self.assertEqual(2, len(hidden_states[0]))
            self.assertEqual((3, 100), hidden_states[0][0].eval().shape)
