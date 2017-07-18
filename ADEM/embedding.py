import errno
import os
import pickle

import numpy as np
import tensorflow as tf


def load_embedding_from_pickle(embedding_lookup_table_path):
    if not os.path.isfile(embedding_lookup_table_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), embedding_lookup_table_path)

    print('...Loading word embedding from {}'.format(embedding_lookup_table_path))
    with open(embedding_lookup_table_path, 'rb') as fileop:
        lookup_table = pickle.load(fileop)
    return lookup_table.astype(np.float32)


def lookup_embedding(vocab_size, embedding_size, input_place,
                     embedding_trainable=True, init_embedding=None,
                     reuse_embedding=None):

    with tf.device("/cpu:0"):
        with tf.variable_scope('embedding', reuse=reuse_embedding):
            if init_embedding is None:
                embedding = tf.get_variable(
                    name='embedding',
                    shape=[vocab_size, embedding_size],
                    trainable=embedding_trainable,
                    dtype=tf.float32)
            else:
                print('Using pretrained word embedding')
                init_embedding = init_embedding.astype(np.float32)
                embedding = tf.get_variable(
                    name='embedding',
                    initializer=init_embedding,
                    trainable=embedding_trainable,
                    dtype=tf.float32)
        input_with_embedding = tf.nn.embedding_lookup(embedding, input_place)
    return input_with_embedding
