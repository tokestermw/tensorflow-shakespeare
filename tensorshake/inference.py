from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import data as shake_data
import model as shake_model
import train as shake_train

FLAGS = shake_train.FLAGS


# TODO: move this to train
def main(_argv):
    if FLAGS.pretrained_embeddings_path is None:
        source_word2idx, source_idx2word = shake_data.build_vocab(FLAGS.source_train)
    else:
        source_word2idx, source_idx2word, source_embedding_matrix = \
            shake_data.build_vocab_with_embeddings(FLAGS.pretrained_embeddings_path)

    target_word2idx, target_idx2word = shake_data.build_vocab(FLAGS.target_train)

    source_vocab_size = len(source_word2idx)
    target_vocab_size = len(target_word2idx)

    tf.logging.info("source vocab size: %i target vocab size: %i", source_vocab_size, target_vocab_size)

    encoder = shake_model.Encoder(
        source_vocab_size,
        num_rnn_layers=FLAGS.num_rnn_layers,
        is_bidi=FLAGS.is_bidi,
        reverse_sequence=FLAGS.reverse_sequence,
        embedding_matrix=source_embedding_matrix,
        use_batch_norm=FLAGS.use_batch_norm,
        use_sentence_projection=FLAGS.use_sentence_projection,
        is_train=False,
        )
    decoder = shake_model.Decoder(
        target_vocab_size,
        add_attention=FLAGS.add_attention)
    seq2seq_model = shake_model.Seq2Seq(encoder, decoder)

    input_ph = tf.placeholder(tf.int32, shape=(None, None))
    decoder_outputs_inference, _ = seq2seq_model(input_ph)
    sampled_word_ids = seq2seq_model.generate(decoder_outputs_inference)

    saver = tf.train.Saver()

    scaffold = tf.train.Scaffold(
        init_op=None,
        init_feed_dict=None,
        init_fn=None,
        ready_op=None,
        ready_for_local_init_op=None,
        local_init_op=None,
        summary_op=None,
        saver=saver)

    _sample_text = shake_train._sample_text

    with tf.train.SingularMonitoredSession(scaffold=scaffold, checkpoint_dir=FLAGS.model_dir) as sess:
        while True:
            text = raw_input("Enter text: ")
            sampled_text = _sample_text(sess, sampled_word_ids, input_ph, text, source_word2idx, target_idx2word)
            print(sampled_text)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

