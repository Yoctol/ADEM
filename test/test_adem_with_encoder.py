import numpy as np
import tensorflow as tf

from ADEM.adem_with_encoder import *


class AdemWithEncoderTest(tf.test.TestCase):

    def setUp(self):
        self.encoder_name = 'lstm_context_encoder'
        self.encoder_params = {'utterence_level_state_size': 50,
                               'utterence_level_keep_proba': 0.8,
                               'utterence_level_num_layers': 2,
                               'context_level_state_size': 100,
                               'context_level_keep_proba': 0.9,
                               'context_level_num_layers': 3}
        self.vocab_size = 10
        self.embedding_size = 30
        self.embedding_trainable = True
        self.learning_rate = 0.1
        self.max_grad_norm = 5

    def test_adem_with_encoder(self):
        with self.test_session() as sess:
            context_place, model_response_place, reference_response_place, \
                context_mask_place, model_response_mask_place, reference_response_mask_place,\
                human_score_place, new_lr_place, train_op, \
                loss, model_score, lr_update = adem_with_encoder(
                    self.learning_rate, self.vocab_size, self.embedding_size, self.embedding_trainable,
                    self.encoder_name, self.encoder_params, self.max_grad_norm)

            sess.run(tf.global_variables_initializer())
            prediction, loss_val, _ = sess.run(
                [model_score, loss, train_op],
                feed_dict={context_place: np.array([[[1, 2, 3], [4, 5, 0]],
                                                    [[3, 0, 0], [0, 0, 0]]]),
                           model_response_place: np.array([[[1, 1, 2]], [[1, 1, 0]]]),
                           reference_response_place: np.array([[[1, 1, 2], [0, 0, 0]],
                                                               [[1, 1, 0], [2, 0, 0]]]),
                           human_score_place: np.array([1., 2.]),
                           context_mask_place: np.array([[3, 2], [1, 0]]),
                           model_response_mask_place: np.array([[3], [2]]),
                           reference_response_mask_place: np.array([[3, 0], [2, 1]])})
            self.assertEqual(2, len(prediction))
