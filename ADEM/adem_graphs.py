import functools

import tensorflow as tf

from .encoder.encoder_on_batch import encoder_on_batch
from .toolkit.embedding import lookup_embedding
from .adem.adem_score import tf_dynamic_adem_score
from .adem.adem_loss import compute_adem_l1_loss


def get_vector_representation(tokens, mask, scope_name,
                              vocab_size, embedding_size,
                              learn_embedding, init_embedding,
                              encoder, output_dim, reuse_embedding=None):
    token_with_embedding = lookup_embedding(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        input_place=tokens,
        embedding_trainable=learn_embedding,
        init_embedding=init_embedding,
        reuse_embedding=reuse_embedding)
    output_vectors = encoder_on_batch(
        batch_with_embedding=token_with_embedding,
        batch_mask=mask,
        encoder=encoder,
        output_dim=output_dim,
        scope_name=scope_name)
    return output_vectors


def adem(context_vector, model_response_vector, reference_response_vector,
         context_dim, model_response_dim, reference_response_dim,
         human_score_place, lr, max_grad_norm):
    model_score, M, N = tf_dynamic_adem_score(
        context=context_vector,
        model_response=model_response_vector,
        reference_response=reference_response_vector,
        shape_info={'batch_size': None,
                    'ct_dim': context_dim,
                    'mr_dim': model_response_dim,
                    'rr_dim': reference_response_dim})

    loss = compute_adem_l1_loss(human_score_place, model_score, M, N)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss, tvars), max_grad_norm)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step()
    )
    return train_op, loss, model_score


def adem_with_encoder_graph(
        learning_rate, vocab_size, embedding_size, learn_embedding,
        context_encoder, model_response_encoder, reference_response_encoder,
        max_grad_norm, init_embedding=None):

    with tf.name_scope('input_placeholder'):
        context_place = tf.placeholder(
            dtype=tf.int32, shape=[None, None, None], name='context_place')
        model_response_place = tf.placeholder(
            dtype=tf.int32, shape=[None, None, None], name='model_response_place')
        reference_response_place = tf.placeholder(
            dtype=tf.int32, shape=[None, None, None], name='reference_response_place')
        human_score_place = tf.placeholder(
            dtype=tf.float32, shape=[None], name='human_score_place')

        context_mask_place = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name='context_mask_place')
        model_response_mask_place = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name='model_response_mask_place')
        reference_response_mask_place = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name='reference_response_mask_place')

    lr = tf.Variable(
        initial_value=tf.constant(
            learning_rate, dtype=tf.float32,
            shape=[], name='learning_rate'),
        trainable=False,
        dtype=tf.float32,
        name='learning_rate')

    get_vector_representation_simple = functools.partial(
        get_vector_representation, vocab_size=vocab_size,
        embedding_size=embedding_size,
        learn_embedding=learn_embedding,
        init_embedding=init_embedding)

    context_vector = get_vector_representation_simple(
        context_place, context_mask_place,
        encoder=context_encoder,
        output_dim=context_encoder['params']['context_level_state_size'],
        scope_name='context_encoder')

    model_response_vector = get_vector_representation_simple(
        model_response_place, model_response_mask_place,
        encoder=model_response_encoder,
        output_dim=model_response_encoder[
            'params']['context_level_state_size'],
        scope_name='model_response_encoder', reuse_embedding=True)

    reference_response_vector = get_vector_representation_simple(
        reference_response_place, reference_response_mask_place,
        encoder=reference_response_encoder,
        output_dim=reference_response_encoder[
            'params']['context_level_state_size'],
        scope_name='reference_response_encoder', reuse_embedding=True)

    train_op, loss, model_score = adem(
        context_vector, model_response_vector, reference_response_vector,
        context_encoder['params']['context_level_state_size'],
        model_response_encoder['params']['context_level_state_size'],
        reference_response_encoder['params']['context_level_state_size'],
        human_score_place, lr, max_grad_norm)

    with tf.name_scope('new_lr'):
        new_lr_place = tf.placeholder(
            tf.float32, shape=[], name='new_learning_rate')
        lr_update = tf.assign(lr, new_lr_place)

    return context_place, model_response_place, reference_response_place, \
        context_mask_place, model_response_mask_place, reference_response_mask_place,\
        human_score_place, new_lr_place, train_op, loss, model_score, lr_update


def adem_graph(context_dim, model_response_dim, reference_response_dim,
               learning_rate=0.1, max_grad_norm=5):
    with tf.name_scope('input_placeholder'):
        context_place = tf.placeholder(
            dtype=tf.int32, shape=[None, context_dim], name='context_place')
        model_response_place = tf.placeholder(
            dtype=tf.int32, shape=[None, model_response_dim], name='model_response_place')
        reference_response_place = tf.placeholder(
            dtype=tf.int32, shape=[None, reference_response_dim], name='reference_response_place')
        human_score_place = tf.placeholder(
            dtype=tf.float32, shape=[None], name='human_score_place')

    lr = tf.Variable(
        initial_value=tf.constant(
            learning_rate, dtype=tf.float32,
            shape=[], name='learning_rate'),
        trainable=False,
        dtype=tf.float32,
        name='learning_rate')

    train_op, loss, model_score = adem(
        context_place, model_response_place, reference_response_place,
        context_dim, model_response_dim, reference_response_dim,
        human_score_place, lr, max_grad_norm)

    with tf.name_scope('new_lr'):
        new_lr_place = tf.placeholder(
            tf.float32, shape=[], name='new_learning_rate')
        lr_update = tf.assign(lr, new_lr_place)

    return context_place, model_response_place, reference_response_place, \
        human_score_place, new_lr_place, train_op, loss, model_score, lr_update
