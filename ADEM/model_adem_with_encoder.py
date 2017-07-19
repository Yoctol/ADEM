import tensorflow as tf
from .embedding import load_embedding_from_pickle
from .adem_with_encoder_graph import adem_with_encoder_graph


class ADEMWithEncoder(object):

    def __init__(self, vocab_size, embedding_size,
                 context_encoder, model_response_encoder=None,
                 reference_response_encoder=None,
                 embedding_lut_path=None, learn_embedding=False,
                 learning_rate=0.1, max_grad_norm=5, **kwargs):
        # embedding
        if embedding_lut_path is not None:
            self.init_embedding = load_embedding_from_pickle(
                embedding_lut_path)
            self.vocab_size, self.embed_size = self.init_embedding.shape
        else:
            self.init_embedding = None
            self.vocab_size = vocab_size
            self.embedding_size = embedding_size

        self.learn_embedding = learn_embedding

        # encoder
        self.context_encoder = context_encoder

        if model_response_encoder is None:
            self.model_response_encoder = self.context_encoder
        else:
            self.model_response_encoder = model_response_encoder

        if reference_response_encoder is None:
            self.reference_response_encoder = self.context_encoder
        else:
            self.reference_response_encoder = reference_response_encoder

        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        ### build graph ###
        self.context_place, self.model_response_place, self.reference_response_place, \
            self.context_mask_place, self.model_response_mask_place, \
            self.reference_response_mask_place,\
            self.human_score_place, self.new_lr_place, self.train_op, \
            self.loss, self.model_score, self.lr_update = self.build_graph()

    def build_graph(self):
        return adem_with_encoder_graph(
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            learn_embedding=self.learn_embedding,
            context_encoder=self.context_encoder,
            model_response_encoder=self.model_response_encoder,
            reference_response_encoder=self.reference_response_encoder,
            init_embedding=self.init_embedding)

    def train_on_single_batch(
            self, train_session,
            context, model_response, reference_response,
            context_mask, model_response_mask, reference_response_mask,
            human_score, learning_rate=None):
        if learning_rate != None:
            self.assign_learning_rate(train_session, learning_rate)

        prediction, loss_val, _ = train_session.run(
            [self.model_score, self.loss, self.train_op],
            feed_dict={
                self.context_place: context,
                self.model_response_place: model_response,
                self.reference_response_place: reference_response,
                self.context_mask_place: context_mask,
                self.model_response_mask_place: model_response_mask,
                self.reference_response_mask_place: reference_response_mask,
                self.human_score_place: human_score
            })
        return prediction, loss_val

    def predict_on_single_batch(
            self, predict_session,
            context, model_response, reference_response,
            context_mask, model_response_mask, reference_response_mask):
        prediction, loss_val, _ = predict_session.run(
            [self.model_score, self.loss],
            feed_dict={
                self.context_place: context,
                self.model_response_place: model_response,
                self.reference_response_place: reference_response,
                self.context_mask_place: context_mask,
                self.model_response_mask_place: model_response_mask,
                self.reference_response_mask_place: reference_response_mask})
        return prediction, loss_val

    def assign_learning_rate(self, train_session, learning_rate):
        # control learning rate outside of model
        print('...Assign new learning rate = {}'.format(learning_rate))
        train_session.run(self.lr_update,
                          feed_dict={self.new_lr_place: learning_rate})
