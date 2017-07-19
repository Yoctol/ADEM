import tensorflow as tf

from .toolkit.embedding import load_embedding_from_pickle
from .adem_graphs import adem_graph


class ADEM(object):

    def __init__(self, context_dim, model_response_dim, reference_response_dim,
                 learning_rate=0.1, max_grad_norm=5, **kwargs):
        self.context_dim = context_dim
        self.model_response_dim = model_response_dim
        self.reference_response_dim = reference_response_dim

        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

    ### build graph ###
        self.context_place, self.model_response_place, \
            self.reference_response_place, \
            self.human_score_place, self.new_lr_place, \
            self.train_op, self.loss, self.model_score, self.lr_update = self.build_graph()

    def build_graph(self):
        return adem_graph(
            context_dim=self.context_dim,
            model_response_dim=self.model_response_dim,
            reference_response_dim=self.reference_response_dim,
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
        )

    def train_on_single_batch(
            self, train_session,
            context, model_response, reference_response,
            human_score, learning_rate=None):

        assert(context.shape[0] == model_response.shape[
            0] == reference_response.shape[0])
        assert(context.shape[0] == len(human_score))
        assert(context.shape[1] == self.context_dim)
        assert(model_response.shape[1] == self.model_response_dim)
        assert(reference_response.shape[1] == self.reference_response_dim)

        if learning_rate != None:
            self.assign_learning_rate(train_session, learning_rate)

        prediction, loss_val, _ = train_session.run(
            [self.model_score, self.loss, self.train_op],
            feed_dict={
                self.context_place: context,
                self.model_response_place: model_response,
                self.reference_response_place: reference_response,
                self.human_score_place: human_score
            })
        return prediction, loss_val

    def predict_on_single_batch(
            self, predict_session,
            context, model_response, reference_response):
        assert(context.shape[0] == model_response.shape[
            0] == reference_response.shape[0])
        assert(context.shape[1] == self.context_dim)
        assert(model_response.shape[1] == self.model_response_dim)
        assert(reference_response.shape[1] == self.reference_response_dim)

        prediction, loss_val, _ = predict_session.run(
            [self.model_score, self.loss],
            feed_dict={
                self.context_place: context,
                self.model_response_place: model_response,
                self.reference_response_place: reference_response})
        return prediction, loss_val

    def assign_learning_rate(self, train_session, learning_rate):
        # control learning rate outside of model
        print('...Assign new learning rate = {}'.format(learning_rate))
        train_session.run(self.lr_update,
                          feed_dict={self.new_lr_place: learning_rate})
