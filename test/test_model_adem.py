import numpy as np
import tensorflow as tf

from ADEM.model_adem import *


class ModelAdemTest(tf.test.TestCase):

    def setUp(self):

        self.learning_rate = 0.1
        self.max_grad_norm = 5

        self.context_dim = 5
        self.model_response_dim = 3
        self.reference_response_dim = 2

    def test_adem_with_encoder(self):
        tf.reset_default_graph()
        with self.test_session() as sess:
            model = ADEM(
                self.context_dim, self.model_response_dim, self.reference_response_dim,
                self.learning_rate, self.max_grad_norm)

            sess.run(tf.global_variables_initializer())

            for _ in range(10):
                prediction, loss = model.train_on_single_batch(
                    sess,
                    context=np.array(
                        [[1, 2, 3, 5, 4], [4, 5, 0, 1, 2]]),
                    model_response=np.array([[1, 1, 2], [1, 1, 0]]),
                    reference_response=np.array(
                        [[1, 1], [2, 0]]),
                    human_score=np.array([1., 2.]))
                self.assertEqual(2, len(prediction))
