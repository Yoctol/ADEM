import numpy as np
import tensorflow as tf

from .layers import *
from .tool import get_last_effective_result


def lstm_context_encoder(
        input_with_embedding, mask,
        utterence_level_state_size,
        utterence_level_keep_proba,
        utterence_level_num_layers,
        context_level_state_size,
        context_level_keep_proba,
        context_level_num_layers,
        utterence_level_forget_bias=1.0,
        utterence_level_activation=tf.tanh,
        context_level_forget_bias=1.0,
        context_level_activation=tf.tanh,
        scope_name=None):

    if scope_name is None:
        scope_name = 'lstm_context_encoder'

    with tf.variable_scope(scope_name, reuse=None):
        # context = several utterences (1 or more)
        # two level encoder
        input_shape = tf.shape(input_with_embedding, )

        # utterence level encoder
        utterence_outputs, utterence_states = multi_lstms(
            input_with_embedding=input_with_embedding,
            mask=mask,
            forget_bias=utterence_level_forget_bias,
            activation=utterence_level_activation,
            batch_size=input_shape[0],
            state_size=utterence_level_state_size,
            keep_prob=utterence_level_keep_proba,
            num_layers=utterence_level_num_layers,
            scope_name='utterence_level_lstm',
            init_state=None)

        final_utterence_output = get_last_effective_result(utterence_outputs, mask)

        # context level encoder
        utt_output_shape = tf.shape(final_utterence_output, )
        context_input = tf.reshape(
            final_utterence_output,
            shape=tf.concat([[1], utt_output_shape], axis=0))

        context_mask = tf.count_nonzero([mask], axis=1)

        context_outputs, context_states = multi_lstms(
            input_with_embedding=context_input,
            mask=context_mask,
            forget_bias=context_level_forget_bias,
            activation=context_level_activation,
            batch_size=1,
            state_size=context_level_state_size,
            keep_prob=context_level_keep_proba,
            num_layers=context_level_num_layers,
            scope_name='context_level_lstm',
            init_state=None)

        final_context_output = get_last_effective_result(
            context_outputs, context_mask)

    return final_context_output
