from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import sonnet as snt


class Encoder(snt.AbstractModule):
    def __init__(self, vocab_size, 
                 embedding_dim=128, 
                 rnn_hidden_dim=128, 
                 rnn_type="lstm", 
                 num_rnn_layers=1, 
                 is_bidi=True, 
                 is_skip_connections=False, 
                 add_attention=True,
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
        self._add_attention = add_attention
        self._reverse_sequence = reverse_sequence

        with self._enter_variable_scope():
            self._embedding_layer = snt.Embed(
                vocab_size, embedding_dim, existing_vocab=None)

            rnn_cell = lambda i: snt.LSTM(rnn_hidden_dim, name="lstm_{}".format(i))
            rnn_layers = [rnn_cell(i) for i in range(num_rnn_layers)]
            self._cell = snt.DeepRNN(rnn_layers, skip_connections=is_skip_connections)

    def _build(self, encoder_inputs, sequence_length):
        batch_size = tf.shape(encoder_inputs)[0]
        initial_state = self._cell.initial_state(batch_size)

        if self._reverse_sequence:
            word_ids = tf.reverse_sequence(word_ids, sequence_length, seq_axis=1)

        batch_embedding_layer = snt.BatchApply(self._embedding_layer)
        embedding_outputs = batch_embedding_layer(encoder_inputs)

        if self._is_bidi:
            rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(          
                    cell_fw=self._cell, cell_bw=self._cell,
                    inputs=embedding_outputs,
                    time_major=False,
                    initial_state_fw=initial_state,
                    initial_state_bw=initial_state,
                    sequence_length=sequence_length)
            rnn_outputs = tf.concat(rnn_outputs, axis=2)
        else:
            rnn_outputs, final_state = tf.nn.dynamic_rnn(          
                cell=self._cell,
                inputs=embedding_outputs,
                time_major=False,
                initial_state=initial_state,
                sequence_length=sequence_length)

        final_state = final_state[-1]  # last RNN layer output
        if isinstance(final_state, tuple):
            final_state = final_state[0]  # assuming h is first dimension

        if self._add_attention:
            encoder_outputs = rnn_outputs
        else:
            encoder_outputs = final_state
        return encoder_outputs


class Decoder(snt.AbstractModule):
    def __init__(self, vocab_size, 
                 embedding_dim=128, 
                 rnn_hidden_dim=128, 
                 output_hidden_dim=128,
                 attention_hidden_dims=128, 
                 rnn_type="lstm", 
                 add_attention=True, 
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
            self._embedding_layer = snt.Embed(
                vocab_size, embedding_dim, existing_vocab=None)

            if self._add_attention:
                if attention_type == "luong":
                    self._create_attention_mechanism = seq2seq.LuongAttention
                elif attention_type == "bahdanaeu":
                    self._create_attention_mechanism = seq2seq.BahdanauAttention

            self._cell = snt.LSTM(rnn_hidden_dim, name="decoder_lstm")

            # TODO: tied weights
            self._output_layer = snt.Linear(vocab_size, name="output_projection")

    def _build(self, encoder_outputs, encoder_sequence_length, decoder_inputs, decoder_sequence_length):
        # TODO: use final_outputs if not using attention
        batch_size = tf.shape(encoder_outputs)[0]

        batch_embedding_layer = snt.BatchApply(self._embedding_layer)
        embedding_outputs = batch_embedding_layer(decoder_inputs)

        # TODO: use GreedyEmbeddingsHelper for inference
        helper = seq2seq.TrainingHelper(embedding_outputs, decoder_sequence_length)

        cell = self._cell

        if self._add_attention:
            # TODO: does reversing sequence matter?
            attention_mechanism = self._create_attention_mechanism(
                num_units=self._attention_hidden_dims,
                memory=encoder_outputs,
                memory_sequence_length=encoder_sequence_length)

            cell = seq2seq.DynamicAttentionWrapper(
                cell,
                attention_mechanism,
                attention_size=self._attention_hidden_dims,
                output_attention=True if self._attention_type == "luong" else False)

        decoder = seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=cell.zero_state(
                dtype=tf.float32, batch_size=batch_size))

        final_outputs, final_state = seq2seq.dynamic_decode(decoder)

        batch_output_layer = snt.BatchApply(self._output_layer)
        output_logits = batch_output_layer(final_outputs.rnn_output)

        decoder_outputs = output_logits
        return decoder_outputs


class Seq2Seq(snt.AbstractModule):
    def __init__(self, encoder, decoder, 
                 mode="train",
                 name="seq2seq"):
        super(Seq2Seq, self).__init__(name=name)
        self._mode = mode

        assert encoder._add_attention == decoder._add_attention, "Attention option needs to sync."
        if decoder._add_attention:
            assert not encoder._reverse_sequence, "Don't reverse sequence when adding attention."

        with self._enter_variable_scope():
            self._encoder = encoder
            self._decoder = decoder

            self._global_step = get_or_create_global_step()

    @staticmethod
    def preprocess_encoder(word_ids):
        word_ids = tf.convert_to_tensor(word_ids, tf.int32)
        sequence_length = tf.reduce_sum(tf.cast(tf.not_equal(word_ids, 0), tf.int32), axis=1)
        return word_ids, sequence_length

    @staticmethod
    def preprocess_decoder(word_ids):
        word_ids = tf.convert_to_tensor(word_ids, tf.int32)
        sequence_length = tf.reduce_sum(tf.cast(tf.not_equal(word_ids, 0), tf.int32), axis=1)

        source_word_ids = word_ids[:, :-1]
        target_word_ids = word_ids[:, 1:]
        sequence_length -= 1
        return source_word_ids, target_word_ids, sequence_length

    def _build(self, encoder_word_ids, decoder_word_ids):
        with tf.device("/cpu:0"):
            encoder_inputs, encoder_sequence_length = self.preprocess_encoder(encoder_word_ids)
            decoder_source_inputs, decoder_target_inputs, decoder_sequence_length = self.preprocess_decoder(decoder_word_ids)

        encoder_outputs = self._encoder(encoder_inputs, encoder_sequence_length)
        decoder_outputs = self._decoder(encoder_outputs, encoder_sequence_length, decoder_source_inputs, decoder_sequence_length)
        # TODO: how to do inference?

        cost = self.cost(decoder_outputs, decoder_target_inputs)
        return cost

    def _check_shape(self):
        pass

    # TODO: integrate into _build?
    def cost(self, decoder_outputs, decoder_inputs):
        logits = tf.reshape(decoder_outputs, (-1, self._decoder._vocab_size))
        labels = tf.reshape(decoder_inputs, (-1, )) 

        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        # TODO: sequence_length
        loss = tf.reduce_mean(xent)
        return loss


def _test():
    source = tf.constant([[0,1,2,3], [0,1,2,3]])
    source_sequence_length = [5, 2]
    target = tf.constant([[1,1,2,3,3], [1,1,2,3,3]])
    target_sequence_length = [3, 4]

    encoder = Encoder(10, num_rnn_layers=2)
    decoder = Decoder(10, add_attention=True)

    # encoder_outputs = encoder(source, sequence_length=source_sequence_length)
    # decoder_outputs = decoder(encoder_outputs, source_sequence_length, target, target_sequence_length)

    seq2seq_model = Seq2Seq(encoder, decoder)
    cost = seq2seq_model(source, target)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        c = sess.run(cost)
        print(c)


if __name__ == "__main__":
    _test()