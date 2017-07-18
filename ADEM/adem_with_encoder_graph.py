import functools

import tensorflow as tf

from .encoder_on_batch import encoder_on_batch
from .embedding import lookup_embedding
from .adem_score import tf_dynamic_adem_score
from .adem_loss import compute_adem_l1_loss


def get_vector_representation(tokens, mask, scope_name,
                              vocab_size, embedding_size,
                              embedding_trainable, init_embedding,
                              encoder_name, encoder_params,
                              reuse_embedding=None):
    token_with_embedding = lookup_embedding(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        input_place=tokens,
        embedding_trainable=embedding_trainable,
        init_embedding=init_embedding,
        reuse_embedding=reuse_embedding)
    output_vectors = encoder_on_batch(
        batch_with_embedding=token_with_embedding,
        batch_mask=mask,
        encoder_name=encoder_name,
        encoder_params=encoder_params,
        scope_name=scope_name)
    return output_vectors


def adem_with_encoder(learning_rate, vocab_size, embedding_size, embedding_trainable,
                      encoder_name, encoder_params, max_grad_norm,
                      init_embedding=None):
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
        embedding_trainable=embedding_trainable,
        init_embedding=init_embedding,
        encoder_name=encoder_name,
        encoder_params=encoder_params)

    context_vector = get_vector_representation_simple(
        context_place, context_mask_place, scope_name='context_encoder')

    model_response_vector = get_vector_representation_simple(
        model_response_place, model_response_mask_place,
        scope_name='model_response_encoder', reuse_embedding=True)

    reference_response_vector = get_vector_representation_simple(
        reference_response_place, reference_response_mask_place,
        scope_name='reference_response_encoder', reuse_embedding=True)

    model_score, M, N = tf_dynamic_adem_score(
        context=context_vector,
        model_response=model_response_vector,
        reference_response=reference_response_vector,
        shape_info={'batch_size': None,
                    'ct_dim': encoder_params['context_level_state_size'],
                    'mr_dim': encoder_params['context_level_state_size'],
                    'rr_dim': encoder_params['context_level_state_size']})

    loss = compute_adem_l1_loss(human_score_place, model_score, M, N)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss, tvars), max_grad_norm)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step()
    )

    with tf.name_scope('new_lr'):
        new_lr_place = tf.placeholder(
            tf.float32, shape=[], name='new_learning_rate')
        lr_update = tf.assign(lr, new_lr_place)

    return context_place, model_response_place, reference_response_place, \
        context_mask_place, model_response_mask_place, reference_response_mask_place,\
        human_score_place, new_lr_place, train_op, loss, model_score, lr_update
