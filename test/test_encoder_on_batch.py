import numpy as np
import tensorflow as tf

from ADEM.encoder_on_batch import *


class EncoderOnBatchTest(tf.test.TestCase):

    def setUp(self):
        self.batch_context_with_embedding = tf.constant(
            [[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
              [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15],
                [4, 1, 2, 6, 7], [0, 0, 0, 0, 0]],
                [[2, 3, 4, 6, 8], [0, 0, 0, 0, 0], [
                    0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
                [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                 [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15],
                  [4, 1, 2, 6, 7], [0, 0, 0, 0, 0]],
                 [[10, 20, 30, 40, 50], [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                 [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 1, 2, 3]]]]
        )
        self.batch_context_mask = tf.constant([[2, 3, 1, 0], [2, 3, 1, 4]])

        self.batch_context_with_embedding = tf.cast(
            self.batch_context_with_embedding, tf.float32)

    def test_encoder_on_batch(self):
        with self.test_session() as sess:
            batch_context_vectors = encoder_on_batch(
                batch_with_embedding=self.batch_context_with_embedding,
                batch_mask=self.batch_context_mask,
                encoder={'name': 'lstm_context_encoder',
                         'params': {'utterence_level_state_size': 50,
                                    'utterence_level_keep_proba': 0.8,
                                    'utterence_level_num_layers': 2,
                                    'context_level_state_size': 100,
                                    'context_level_keep_proba': 0.9,
                                    'context_level_num_layers': 3}},
                output_dim=100)

            sess.run(tf.global_variables_initializer())
            self.assertEqual((2, 100), batch_context_vectors.eval().shape)

    def test_check_encoder_format(self):
        with self.assertRaises(TypeError):
            check_encoder_format(['encodera', 'encoderb'])

        with self.assertRaises(KeyError):
            check_encoder_format({'name': 'lstm_context_encoder'})

        with self.assertRaises(KeyError):
            check_encoder_format({'params': 'lstm_context_encoder'})

        with self.assertRaises(TypeError):
            check_encoder_format({'name': ['a', 'b', 'c'],
                                  'params': {'a': 1, 'b': 2}})


    def test_get_encoder(self):
        with self.assertRaises(KeyError):
            input_ = {'name': 'abencoder'}
            get_encoder(input_)
