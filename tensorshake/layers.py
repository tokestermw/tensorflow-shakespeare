""" Custom layers.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt
from sonnet.python.modules import basic


class Embedding(snt.Embed):
    """ Embedding layer that builds the matrix at __init__().
    
    So we can set the embedding layer and use it elsewhere 
    (e.g. tied weights for the output layer, or using embedding matrix at inference)
    """

    def __init__(self,
                 vocab_size=None,
                 embed_dim=None,
                 existing_vocab=None,
                 initializers=None,
                 partitioners=None,
                 regularizers=None,
                 trainable=True,
                 name="embedding"
                 ):
        super(Embedding, self).__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            existing_vocab=existing_vocab,
            initializers=initializers,
            partitioners=partitioners,
            regularizers=regularizers,
            trainable=trainable,
            name=name
        )

        with self._enter_variable_scope():
            if self._existing_vocab is None:
                if self.EMBEDDINGS not in self._initializers:
                    self._initializers[self.EMBEDDINGS] = basic.create_linear_initializer(
                        self._vocab_size)
                self._embeddings = tf.get_variable(
                    "embed",
                    shape=[self._vocab_size, self._embed_dim],
                    dtype=tf.float32,
                    initializer=self._initializers[self.EMBEDDINGS],
                    partitioner=self._partitioners.get(self.EMBEDDINGS, None),
                    regularizer=self._regularizers.get(self.EMBEDDINGS, None),
                    trainable=self._trainable)
            else:
                self._embeddings = tf.get_variable(
                    "embed",
                    dtype=tf.float32,
                    initializer=self._existing_vocab,
                    regularizer=self._regularizers.get(self.EMBEDDINGS, None),
                    trainable=self._trainable)

    def _build(self, ids):
        return tf.nn.embedding_lookup(
            self._embeddings, ids, name="embedding_lookup")

    @property
    def embeddings(self):
        # self._ensure_is_connected()
        return self._embeddings



