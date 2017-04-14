from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import tensorflow as tf

import data as shake_data
import model as shake_model

flags = tf.flags

flags.DEFINE_string("model_dir", "./tmp", (
    "Model directory."))
flags.DEFINE_string("source_train", shake_data.default_modern_train_path, (
    ""))
flags.DEFINE_string("source_dev", shake_data.default_modern_dev_path, (
    ""))
flags.DEFINE_string("target_train", shake_data.default_original_train_path, (
    ""))
flags.DEFINE_string("target_dev", shake_data.default_original_dev_path, (
    ""))

FLAGS = flags.FLAGS


# TODO: randomize the data
def get_input_queues(source_path, source_vocab, target_path, target_vocab,
                     batch_size=32, num_threads=8, maxlen=None):

    source_ph = tf.placeholder(tf.int32, shape=[None, maxlen])  # [B, T]
    target_ph = tf.placeholder(tf.int32, shape=[None, maxlen])  # [B, T]

    queue = tf.PaddingFIFOQueue(shapes=[[maxlen, ], [maxlen, ]], dtypes=[tf.int32, tf.int32], capacity=5000,)

    enqueue_op = queue.enqueue_many([source_ph, target_ph])
    def enqueue_data(sess):
        while True:
            for source, target in shake_data.data_iterator(source_path, source_vocab, target_path, target_vocab):
                sess.run(enqueue_op, feed_dict={source_ph: source, target_ph: target})

    dequeue_op = queue.dequeue_many(batch_size)
    dequeue_batch = dequeue_op
    # dequeue_batch = tf.train.batch([dequeue_op], batch_size=batch_size, num_threads=num_threads, capacity=1000, 
    #     dynamic_pad=True, name="batch_and_pad")

    return enqueue_data, dequeue_batch


def start_threads(thread_fn, args, n_threads=1):
    assert n_threads == 1, "Having multiple threads causes duplicate data in the queue."

    threads = []
    for n in range(n_threads):
        t = threading.Thread(target=thread_fn, args=args)
        t.daemon = True  # thread will close when parent quits
        t.start()
        threads.append(t)

    # time.sleep(1)  # enqueue a bunch before dequeue
    return threads


def main(_argv):
    source_word2idx, source_idx2word = shake_data.build_vocab(FLAGS.source_train)
    target_word2idx, target_idx2word = shake_data.build_vocab(FLAGS.target_train)    

    encoder = shake_model.Encoder(len(source_word2idx))
    decoder = shake_model.Decoder(len(target_word2idx))
    seq2seq_model = shake_model.Seq2Seq(encoder, decoder)

    enqueue_data, dequeue_batch = get_input_queues(
        FLAGS.source_train, source_word2idx, FLAGS.target_train, target_word2idx)

    cost = seq2seq_model(*dequeue_batch)
    # train_op

    with tf.train.SingularMonitoredSession(checkpoint_dir=FLAGS.model_dir) as sess:
        start_threads(enqueue_data, (sess, ))

        # while True:
        #     a, b = sess.run(dequeue_batch)
        #     print(a.shape, b.shape)
        #     print(a)
        #     print(b)

        while True:
            c = sess.run(cost)
            print(c)


if __name__ == "__main__":
    tf.app.run()