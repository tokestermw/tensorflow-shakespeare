from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.layers import core as layers_core

import data as shake_data


def preprocess_encoder(word_ids):
    word_ids = tf.convert_to_tensor(word_ids, tf.int32)
    sequence_length = tf.reduce_sum(
        tf.cast(tf.not_equal(word_ids, 0), tf.int32), axis=1)
    return word_ids, sequence_length


def preprocess_decoder(word_ids):
    if word_ids is None:
        return None, None, None

    word_ids = tf.convert_to_tensor(word_ids, tf.int32)
    sequence_length = tf.reduce_sum(
        tf.cast(tf.not_equal(word_ids, 0), tf.int32), axis=1)

    source_word_ids = word_ids[:, :-1]
    target_word_ids = word_ids[:, 1:]
    sequence_length -= 1
    return source_word_ids, target_word_ids, sequence_length


def encoder_function(inputs,
            vocab_size, 
            embedding_size=128,
            rnn_hidden_size=128,
            dropout_rnn=1.0,
            num_rnn_layers=1,  # TODO: doesn't work with dynamic_decode
            trainable=True,
            ):
    
    with tf.device("/cpu:0"):
        word_ids, sequence_length = preprocess_encoder(inputs)

        embedding_matrix = tf.get_variable(
            "embedding", shape=[vocab_size, embedding_size],
            dtype=tf.float32)

        embedding_inputs = tf.nn.embedding_lookup(
            embedding_matrix, word_ids, name="embedding_lookup")

    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        rnn_hidden_size, layer_norm=True, dropout_keep_prob=dropout_rnn)

    rnn_outputs, final_state = tf.nn.dynamic_rnn(
        cell=rnn_cell,
        inputs=embedding_inputs,
        sequence_length=sequence_length,
        time_major=False, dtype=tf.float32)

    return rnn_outputs, final_state, sequence_length


def decoder_function(encoder_outputs,
            encoder_final_state,
            encoder_sequence_length,
            decoder_inputs,
            vocab_size,
            embedding_size=128,
            rnn_hidden_size=128,
            attention_hidden_size=128,
            dropout_rnn=1.0,
            ):
    is_inference = decoder_inputs is None
    batch_size = tf.shape(encoder_outputs)[0]

    with tf.device("/cpu:0"):
        embedding_matrix = tf.get_variable(
            "embedding", shape=[vocab_size, embedding_size],
            dtype=tf.float32)

        source_word_ids, target_word_ids, sequence_length = \
            preprocess_decoder(decoder_inputs)

        if not is_inference:
            embedding_inputs = tf.nn.embedding_lookup(
                embedding_matrix, source_word_ids, name="embedding_lookup")

    rnn_cell =  tf.contrib.rnn.LayerNormBasicLSTMCell(
        rnn_hidden_size, layer_norm=True, dropout_keep_prob=dropout_rnn)

    output_layer = layers_core.Dense(
        vocab_size, use_bias=False, trainable=True, name="output_projections")

    if is_inference:
        helper = seq2seq.GreedyEmbeddingHelper(
            embedding_matrix, 
            start_tokens=tf.tile([2], [batch_size]), end_token=3)
    else:
        # TODO: add scheduled sampling option
        helper = seq2seq.TrainingHelper(
            embedding_inputs, sequence_length)

    attention_mechanism = seq2seq.LuongAttention(
        num_units=attention_hidden_size,
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length)

    rnn_cell = seq2seq.AttentionWrapper(
        rnn_cell,
        attention_mechanism,
        attention_layer_size=attention_hidden_size,
        alignment_history=False,
        cell_input_fn=None,
        output_attention=True,  # Luong style attention mechanism
        initial_cell_state=None)

    initial_state = rnn_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    initial_state = initial_state.clone(cell_state=encoder_final_state)

    decoder = seq2seq.BasicDecoder(
        cell=rnn_cell,
        helper=helper,
        initial_state=initial_state,
        output_layer=output_layer)

    final_outputs, final_state, final_sequence_length = \
        seq2seq.dynamic_decode(decoder, maximum_iterations=shake_data.MAXLEN)
    return final_outputs.rnn_output, final_sequence_length, target_word_ids


def cost_function(decoder_outputs, decoder_sequence_length, target_word_ids):
    # so sequence_length matches
    chop_id = tf.reduce_max(decoder_sequence_length)
    mask = tf.cast(tf.sequence_mask(decoder_sequence_length), tf.float32)
    cost = seq2seq.sequence_loss(
        decoder_outputs, target_word_ids[:, :chop_id], mask)
    return cost


class Seq2Seq:
    def __init__(self, source_vocab_size, target_vocab_size, config):
        self._source_vocab_size = source_vocab_size
        self._target_vocab_size = target_vocab_size
        self._config = config

        self._global_step = tf.contrib.framework.get_or_create_global_step()
        self._increment_op = tf.assign(self._global_step, self._global_step + 1)

        self._encoder_function = tf.make_template(
            "encoder_function", encoder_function, create_scope_now_=True)
        self._decoder_function = tf.make_template(
            "decoder_function", decoder_function, create_scope_now_=True)

    def __call__(self, source_inputs, target_inputs):

        encoder_outputs, encoder_final_state, encoder_sequence_length = \
            self._encoder_function(
                source_inputs, self._source_vocab_size,
                embedding_size=self.config.embedding_size,
                rnn_hidden_size=self.config.rnn_hidden_size,
                dropout_rnn=self.config.dropout_rnn \
                    if self.config.is_inference else 1.0,
                num_rnn_layers=self.config.num_rnn_layers,
                trainable=self.config.trainable)

        decoder_outputs, decoder_sequence_length, target_word_ids = \
            self._decoder_function(
                encoder_outputs, encoder_final_state, encoder_sequence_length,
                target_inputs, self._target_vocab_size,
                embedding_size=self.config.embedding_size,
                rnn_hidden_size=self.config.rnn_hidden_size,
                attention_hidden_size=self.config.attention_hidden_size,
                dropout_rnn=self.config.dropout_rnn \
                    if self.config.is_inference else 1.0)

        if self.config.is_inference:
            return encoder_outputs, decoder_outputs, None

        cost = cost_function(
            decoder_outputs, decoder_sequence_length, target_word_ids)

        return encoder_outputs, decoder_outputs, cost

    @property
    def config(self):
        return self._config
    
    @property
    def global_step(self):
        return self._global_step

    @property
    def increment_op(self):
        return self._increment_op
    
    @property
    def encoder_outputs(self):
        return self._encoder_outputs

    @property
    def decoder_outputs(self):
        return self._decoder_outputs
    
    @property
    def encoder_final_state(self):
        return self._encoder_final_state

    @property
    def cost(self):
        return self._cost
    

if __name__ == "__main__":
    _test()
