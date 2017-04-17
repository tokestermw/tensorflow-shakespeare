from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import sonnet as snt

from tensorflow.python.layers import core as layers_core

import layers as shake_layers

MAXLEN = 100


class Encoder(snt.AbstractModule):
    def __init__(self, vocab_size,
                 embedding_dim=128,
                 rnn_hidden_dim=128,
                 rnn_type="lstm",
                 num_rnn_layers=1,
                 is_bidi=False,
                 is_skip_connections=False,
                 reverse_sequence=False,
                 name="encoder"):
        super(Encoder, self).__init__(name=name)

        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._rnn_hidden_dim = rnn_hidden_dim

        self._rnn_type = rnn_type
        self._num_rnn_layers = num_rnn_layers
        self._is_bidi = is_bidi
        self._is_skip_connections = is_skip_connections
        self._reverse_sequence = reverse_sequence

        with self._enter_variable_scope():
            # self._embedding_layer = snt.Embed(
            #     vocab_size, embedding_dim, existing_vocab=None)
            self._embedding_layer = shake_layers.Embedding(
                vocab_size, embedding_dim, existing_vocab=None)

            rnn_cell = lambda i: snt.LSTM(rnn_hidden_dim, name="lstm_{}".format(i))
            rnn_layers = [rnn_cell(i) for i in range(num_rnn_layers)]
            self._cell = snt.DeepRNN(rnn_layers, skip_connections=is_skip_connections)

    def _build(self, encoder_inputs, sequence_length):
        batch_size = tf.shape(encoder_inputs)[0]
        initial_state = self._cell.initial_state(batch_size)

        if self._reverse_sequence:
            encoder_inputs = tf.reverse_sequence(encoder_inputs, sequence_length, seq_axis=1)

        batch_embedding_layer = snt.BatchApply(self._embedding_layer)
        embedding_outputs = batch_embedding_layer(encoder_inputs)

        if self._is_bidi:
            rnn_outputs, (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self._cell, cell_bw=self._cell,
                inputs=embedding_outputs,
                time_major=False,
                initial_state_fw=initial_state,
                initial_state_bw=initial_state,
                sequence_length=sequence_length)

            rnn_outputs = tf.concat(rnn_outputs, axis=2)

            final_state_fw = final_state_fw[-1]
            final_state_bw = final_state_bw[-1]
            # TODO: add average or concat option?
            final_state = tuple([tf.add(*i) for i in zip(final_state_fw, final_state_bw)])

        else:
            rnn_outputs, final_state = tf.nn.dynamic_rnn(
                cell=self._cell,
                inputs=embedding_outputs,
                time_major=False,
                initial_state=initial_state,
                sequence_length=sequence_length)

            final_state = final_state[-1]  # last RNN layer output

        # TODO: output projection layer?
        return rnn_outputs, final_state


class Decoder(snt.AbstractModule):
    def __init__(self, vocab_size,
                 embedding_dim=128,
                 rnn_hidden_dim=128,
                 attention_hidden_dims=128,
                 rnn_type="lstm",
                 add_attention=False,
                 attention_type="luong",
                 name="decoder"):
        super(Decoder, self).__init__(name=name)

        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._rnn_hidden_dim = rnn_hidden_dim

        self._add_attention = add_attention
        self._attention_hidden_dims = attention_hidden_dims
        self._attention_type = attention_type

        with self._enter_variable_scope():
            # self._embedding_layer = snt.Embed(
            #     vocab_size, embedding_dim, existing_vocab=None)
            self._embedding_layer = shake_layers.Embedding(
                vocab_size, embedding_dim, existing_vocab=None)

            if self._add_attention:
                if attention_type == "luong":
                    self._create_attention_mechanism = seq2seq.LuongAttention
                elif attention_type == "bahdanaeu":
                    self._create_attention_mechanism = seq2seq.BahdanauAttention
                else:
                    raise ValueError("Wrong attention_type.")

            self._cell = snt.LSTM(rnn_hidden_dim, name="decoder_lstm")

            # TODO: tied weights
            # self._output_layer = snt.Linear(vocab_size, name="output_projection")
            self._output_layer = layers_core.Dense(
                vocab_size, use_bias=False, trainable=True, name="output_projection")

    def _build(self, encoder_outputs, encoder_final_state, encoder_sequence_length,
               decoder_inputs=None, decoder_sequence_length=None):
        is_inference = decoder_inputs is None and decoder_sequence_length is None

        batch_size = tf.shape(encoder_outputs)[0]

        if is_inference:
            # TODO: if only doing inference, this hasn't gone instantiated.
            embedding_matrix = self._embedding_layer.embeddings
            helper = seq2seq.GreedyEmbeddingHelper(
                embedding_matrix, start_tokens=tf.tile([2], [batch_size]), end_token=3)
        else:
            # TODO: add scheduled sampling option
            batch_embedding_layer = snt.BatchApply(self._embedding_layer)
            embedding_outputs = batch_embedding_layer(decoder_inputs)

            helper = seq2seq.TrainingHelper(embedding_outputs, decoder_sequence_length)

        cell = self._cell

        if self._add_attention:
            attention_mechanism = self._create_attention_mechanism(
                num_units=self._attention_hidden_dims,
                memory=encoder_outputs,
                memory_sequence_length=encoder_sequence_length)

            cell = seq2seq.DynamicAttentionWrapper(
                cell,
                attention_mechanism,
                attention_size=self._attention_hidden_dims,
                output_attention=True if self._attention_type == "luong" else False)

        # -- same as below but only works for training not inference
        # final_outputs, final_state = tf.nn.dynamic_rnn(
        #         cell=self._cell,
        #         inputs=embedding_outputs,
        #         time_major=False,
        #         dtype=tf.float32,
        #         initial_state=cell.initial_state(batch_size),
        #         initial_state=encoder_final_state,
        # sequence_length=decoder_sequence_length)

        if self._add_attention:
            _initial_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
            initial_state = seq2seq.DynamicAttentionWrapperState(
                cell_state=encoder_final_state, attention=_initial_state.attention)
        else:
            initial_state = encoder_final_state

        # TODO: add beam search decoder
        decoder = seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=initial_state,
            # .rnn_output will include calculations of the output layer if output_layer is not None
            output_layer=self._output_layer)

        final_outputs, final_state = seq2seq.dynamic_decode(decoder, maximum_iterations=MAXLEN)
        return final_outputs.rnn_output


class Seq2Seq(snt.AbstractModule):
    def __init__(self, encoder, decoder,
                 name="seq2seq"):
        super(Seq2Seq, self).__init__(name=name)

        if decoder._add_attention:
            assert not encoder._reverse_sequence, "Don't reverse sequence when adding attention."

        with self._enter_variable_scope():
            self._encoder = encoder
            self._decoder = decoder

            self._global_step = get_or_create_global_step()
            self._increment_global_step = tf.assign(self._global_step, self._global_step + 1)

    @staticmethod
    def preprocess_encoder(word_ids):
        word_ids = tf.convert_to_tensor(word_ids, tf.int32)
        sequence_length = tf.reduce_sum(tf.cast(tf.not_equal(word_ids, 0), tf.int32), axis=1)
        return word_ids, sequence_length

    @staticmethod
    def preprocess_decoder(word_ids):
        if word_ids is None:
            return None, None, None

        word_ids = tf.convert_to_tensor(word_ids, tf.int32)
        sequence_length = tf.reduce_sum(tf.cast(tf.not_equal(word_ids, 0), tf.int32), axis=1)

        source_word_ids = word_ids[:, :-1]
        target_word_ids = word_ids[:, 1:]
        sequence_length -= 1
        return source_word_ids, target_word_ids, sequence_length

    def _build(self, encoder_word_ids, decoder_word_ids=None):
        is_inference = decoder_word_ids is None

        with tf.device("/cpu:0"):
            encoder_inputs, encoder_sequence_length = self.preprocess_encoder(
                encoder_word_ids)
            decoder_source_inputs, decoder_target_inputs, decoder_sequence_length = self.preprocess_decoder(
                decoder_word_ids)

        encoder_outputs, encoder_final_state = self._encoder(
            encoder_inputs, encoder_sequence_length)

        decoder_outputs = self._decoder(
            encoder_outputs, encoder_final_state, encoder_sequence_length,
            decoder_source_inputs, decoder_sequence_length)

        return decoder_outputs, decoder_target_inputs

    def cost(self, decoder_outputs, decoder_inputs):
        logits = tf.reshape(decoder_outputs, (-1, self._decoder._vocab_size))
        labels = tf.reshape(decoder_inputs, (-1,))

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy = tf.reshape(cross_entropy, tf.shape(decoder_inputs))

        mask = tf.cast(tf.not_equal(decoder_inputs, tf.zeros_like(decoder_inputs)), tf.float32)
        sequence_length = tf.reduce_sum(mask, axis=1)

        loss = tf.reduce_sum(cross_entropy * mask, axis=1) / sequence_length
        cost = tf.reduce_mean(loss)
        return cost

    # TODO: add other stuff like tokenizer, vectorizer?
    @staticmethod
    def generate(decoder_outputs):
        sampled_word_ids = tf.argmax(decoder_outputs, axis=-1)
        return sampled_word_ids


def _test():
    source = tf.constant([[0, 1, 2, 3], [0, 1, 2, 3]])
    target = tf.constant([[1, 1, 2, 3, 3], [1, 1, 2, 3, 3]])

    rev_vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    encoder = Encoder(10, num_rnn_layers=2)
    decoder = Decoder(10, add_attention=True)

    seq2seq_model = Seq2Seq(encoder, decoder)
    train_outputs, train_targets = seq2seq_model(source, target)
    cost = seq2seq_model.cost(train_outputs, train_targets)
    inference_outputs, _ = seq2seq_model(source)
    sampled_word_ids = seq2seq_model.generate(inference_outputs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        c = sess.run(cost)
        print(c)
        s = sess.run(sampled_word_ids)
        print(s)


if __name__ == "__main__":
    _test()
