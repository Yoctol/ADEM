import numpy as np
import tensorflow as tf

from ADEM.rnn_encoder import *


class RNNEncoderTest(tf.test.TestCase):

    def test_lstm_context_encoder(self):
        with self.test_session() as sess:
            input_with_embedding = tf.constant(
                [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                 [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15],
                     [4, 1, 2, 6, 7], [0, 0, 0, 0, 0]],
                 [[10, 20, 30, 40, 50], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
            input_with_embedding = tf.cast(input_with_embedding, tf.float32)
            context_vector = lstm_context_encoder(
                input_with_embedding=input_with_embedding,
                mask=np.array([2, 3, 1]),
                utterence_level_state_size=50,
                utterence_level_keep_proba=0.9,
                utterence_level_num_layers=2,
                context_level_state_size=100,
                context_level_keep_proba=0.8,
                context_level_num_layers=3)
            sess.run(tf.global_variables_initializer())
            self.assertEqual((1, 100), context_vector.eval().shape)
