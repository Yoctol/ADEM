import numpy as np
import tensorflow as tf

from ADEM.model_adem_with_encoder import *


class ModelAdemWithEncoderTest(tf.test.TestCase):

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
        self.embedding_lut_path = None
        self.vocab_size = 10
        self.embedding_size = 30
        self.learn_embedding = True
        self.learning_rate = 0.1
        self.max_grad_norm = 5

    def test_adem_with_encoder(self):
        tf.reset_default_graph()
        with self.test_session() as sess:
            model = ADEMWithEncoder(
                self.vocab_size, self.embedding_size,
                self.context_encoder, self.model_response_encoder,
                self.reference_response_encoder, self.embedding_lut_path,
                self.learn_embedding, self.learning_rate, self.max_grad_norm)

            sess.run(tf.global_variables_initializer())

            for _ in range(10):
                prediction, loss = model.train_on_single_batch(
                    sess,
                    context=np.array(
                        [[[1, 2, 3], [4, 5, 0]], [[3, 0, 0], [0, 0, 0]]]),
                    model_response=np.array([[[1, 1, 2]], [[1, 1, 0]]]),
                    reference_response=np.array(
                        [[[1, 1, 2], [0, 0, 0]], [[1, 1, 0], [2, 0, 0]]]),
                    human_score=np.array([1., 2.]),
                    context_mask=np.array([[3, 2], [1, 0]]),
                    model_response_mask=np.array([[3], [2]]),
                    reference_response_mask=np.array([[3, 0], [2, 1]]))

                self.assertEqual(2, len(prediction))
