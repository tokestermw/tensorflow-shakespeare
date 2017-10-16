from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
import threading
import json

import tensorflow as tf

import utils as shake_utils
import data as shake_data
import model as shake_model


flags = tf.flags

flags.DEFINE_string("model_dir", "./tmp", (
    "Model directory."))
flags.DEFINE_boolean("is_inference", False, (
    ""))

flags.DEFINE_string("source_train", shake_data.default_modern_train_path, (
    ""))
flags.DEFINE_string("source_dev", shake_data.default_modern_dev_path, (
    ""))
flags.DEFINE_string("target_train", shake_data.default_original_train_path, (
    ""))
flags.DEFINE_string("target_dev", shake_data.default_original_dev_path, (
    ""))
flags.DEFINE_string("pretrained_embeddings_path", shake_data.default_glove_6B_50d_path, (
    ""))

flags.DEFINE_integer("logging_frequency", 25, (
    ""))

flags.DEFINE_integer("batch_size", 32, (
    ""))
flags.DEFINE_integer("epoch_size", 2, (
    ""))

flags.DEFINE_string("optimizer_type", "adam", (
    ""))
flags.DEFINE_float("learning_rate", 1e-4, (
    ""))
flags.DEFINE_float("max_grads", 5.0, (
    ""))

flags.DEFINE_integer("embedding_size", 128, (
    ""))
flags.DEFINE_integer("rnn_hidden_size", 128, (
    ""))
flags.DEFINE_integer("attention_hidden_size", 128, (
    ""))
flags.DEFINE_float("dropout_rnn", 0.6, (
    ""))
flags.DEFINE_integer("num_rnn_layers", 1, (
    ""))
flags.DEFINE_boolean("trainable", True, (
    ""))

FLAGS = flags.FLAGS


def set_train_op(loss, tvars):
    # TODO set all optimizers so it's available in the graph (doesn't work, need to setup train_op)
    optimizer_dict = {
        "sgd": tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate),
        "rmsprop": tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate),
        "adam": tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    }

    optimizer = optimizer_dict[FLAGS.optimizer_type]

    gradients = optimizer.compute_gradients(loss, var_list=tvars)
    clipped_gradients = [(grad if grad is None else tf.clip_by_norm(grad, FLAGS.max_grads), var)
                         for grad, var in gradients]

    train_op = optimizer.apply_gradients(clipped_gradients)
    return train_op


def main(_argv):
    dataset = shake_data.ParallelDataset(
        FLAGS.source_train, FLAGS.target_train,
        batch_size=FLAGS.batch_size, epoch_size=FLAGS.epoch_size)

    config = FLAGS.__flags

    model = shake_model.Seq2Seq(
        dataset.source_vocab_size, dataset.target_vocab_size, FLAGS)
    _, _, cost = model(
        dataset.source_inputs, dataset.target_inputs)

    # TODO: an annoying warning, hack to remove
    # https://github.com/tensorflow/tensorflow/issues/9939#issuecomment-303717350
    del tf.get_collection_ref('LAYER_NAME_UIDS')[0]

    tvars = tf.trainable_variables()
    train_op = set_train_op(cost, tvars)

    saver = tf.train.Saver()

    summary_op = tf.summary.merge_all()

    hooks = [
        tf.train.LoggingTensorHook({
            "cost": cost,
            "global_step": model.global_step
        }, every_n_secs=60),
        tf.train.StepCounterHook(every_n_steps=100),
        tf.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.model_dir,
            save_steps=100,
            saver=saver),
    ]

    scaffold = tf.train.Scaffold(
        init_op=None,
        init_feed_dict=None,
        init_fn=None,
        ready_op=None,
        ready_for_local_init_op=None,
        local_init_op=None,
        summary_op=summary_op,
        saver=saver)

    with tf.train.SingularMonitoredSession(
        hooks=hooks, checkpoint_dir=FLAGS.model_dir, scaffold=scaffold) as sess:

        shake_utils.save_config(config, FLAGS.model_dir)
        
        while True:
            _, step = sess.run(
                [train_op, model.increment_op])


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
