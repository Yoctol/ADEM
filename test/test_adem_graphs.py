import numpy as np
import tensorflow as tf

from ADEM.adem_graphs import *


class AdemWithEncoderGraphTest(tf.test.TestCase):

    def setUp(self):
        self.context_encoder = {
            'name': 'lstm_context_encoder',
            'params': {'utterence_level_state_size': 50,
                       'utterence_level_keep_proba': 0.8,
                       'utterence_level_num_layers': 2,
                       'context_level_state_size': 100,
                       'context_level_keep_proba': 0.9,
                       'context_level_num_layers': 3}}
        self.model_response_encoder = {
            'name': 'lstm_context_encoder',
            'params': {'utterence_level_state_size': 100,
                       'utterence_level_keep_proba': 0.8,
                       'utterence_level_num_layers': 1,
                       'context_level_state_size': 150,
                       'context_level_keep_proba': 0.9,
                       'context_level_num_layers': 1}}
        self.reference_response_encoder = {
            'name': 'lstm_context_encoder',
            'params': {'utterence_level_state_size': 10,
                       'utterence_level_keep_proba': 0.7,
                       'utterence_level_num_layers': 2,
                       'context_level_state_size': 50,
                       'context_level_keep_proba': 0.9,
                       'context_level_num_layers': 1}}
        self.vocab_size = 10
        self.embedding_size = 30
        self.learn_embedding = True
        self.learning_rate = 0.1
        self.max_grad_norm = 5

    def test_adem_with_encoder_graph(self):
        tf.reset_default_graph()
        with self.test_session() as sess:
            context_place, model_response_place, reference_response_place, \
                context_mask_place, model_response_mask_place, reference_response_mask_place,\
                human_score_place, new_lr_place, train_op, \
                loss, model_score, lr_update = adem_with_encoder_graph(
                    self.learning_rate, self.vocab_size, self.embedding_size, self.learn_embedding,
                    self.context_encoder, self.model_response_encoder,
                    self.reference_response_encoder, self.max_grad_norm)

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
