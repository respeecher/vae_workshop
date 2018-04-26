import tensorflow as tf
import numpy as np
import time
import json
import os, sys
from random import shuffle
import argparse
import librosa

from utils.train_utils import save
from inverter import Inverter


parser = argparse.ArgumentParser()
parser.add_argument('logdir', type=str,
                    help='Directory to store checkpoints and summaries')
parser.add_argument('train_dir', type=str,
                    help='Directory with training data')
parser.add_argument('test_dir', type=str,
                    help='Directory with test data')
args = parser.parse_args()


inv = Inverter()
batch_size = 32
num_epochs = 1000


def image_summary(tensor, name):
    im_sum = tf.summary.image(
        name, tf.expand_dims(tensor[:1], -1))
    return im_sum


def get_random_spectrogram(folder):
    fnames = os.listdir(folder)
    shuffle(fnames)
    for f in fnames:
        path = os.path.join(folder, f)
        feats = np.load(path)
        yield (feats['logmel'], feats['logspec'])


def create_dataset(folder, batch_size):
    ds = tf.data.Dataset.from_generator(
            lambda: get_random_spectrogram(folder),
            (tf.float32, tf.float32),
            ((None, inv.n_mel), (None, inv.n_spec)))

    # Yes, what you see is hardcoded silence paddings. Because it's like 3am
    # and I still have tons of debugging to do
    ds = ds.apply(
            tf.contrib.data.padded_batch_and_drop_remainder(
                batch_size,
                padded_shapes=((None, inv.n_mel), (None, inv.n_spec)),
                padding_values=(0.0001, 0.0001)))
    return ds.prefetch(256)

with inv.graph.as_default():
    loss = inv.loss()

    optimizer = tf.train.AdamOptimizer()
    grads = optimizer.compute_gradients(loss, tf.trainable_variables())
    with tf.name_scope('gradient_clipping'):
        clipped_grads = [
            (tf.clip_by_norm(grad, 1e-3), var)
            for grad, var in grads
            if grad is not None
        ]
    updates = optimizer.apply_gradients(clipped_grads)

    writer = tf.summary.FileWriter(args.logdir)
    writer.add_graph(inv.graph)

    with tf.name_scope('summaries'):
        loss_sum = tf.summary.scalar('spec_loss', loss)
        loss_test_sum = tf.summary.scalar('spec_test_loss', loss)

        spec_sum = image_summary(inv.spec, 'spec')
        spec_pred_sum = image_summary(inv.spec_pred, 'spec_pred')

        summaries = [loss_sum]

        summaries_test = [loss_test_sum,
                          spec_pred_sum,
                          spec_sum]


        summaries_train_op = tf.summary.merge(summaries)
        summaries_test_op = tf.summary.merge(summaries_test)


    train_ds = create_dataset(args.train_dir, batch_size)
    test_ds = create_dataset(args.test_dir, batch_size).repeat()

    train_iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                                     train_ds.output_shapes)
    next_train_batch = train_iterator.get_next()
    next_test_batch = test_ds.make_one_shot_iterator().get_next()

    training_init_op = train_iterator.make_initializer(train_ds)

    inputs_train = {inv.mel: next_train_batch[0],
                    inv.spec: next_train_batch[1]}
    inputs_test = {inv.mel: next_test_batch[0],
                   inv.spec: next_test_batch[1]}

    inv.init_and_load(args.logdir)


for epoch in range(num_epochs):
    inv.sess.run(training_init_op)
    while True:
        inv.sess.run(inv.inc_global_i)
        i_ = inv.sess.run(inv.global_i)
        try:
            if i_ % 10 == 0:
                # validation performance
                prefix = 'Valid '
                inputs = inv.sess.run(inputs_test)
                summaries, loss_ = inv.sess.run(
                    [summaries_test_op, loss], inputs)
            else:
                # training step
                prefix = 'Train '
                inputs = inv.sess.run(inputs_train)
                summaries, loss_, _ = inv.sess.run(
                    [summaries_train_op, loss, updates], inputs)

            print('{} Epoch: {}, Step: {}, Loss: {:.3f}.'
                  .format(prefix, epoch, i_, loss_))

            writer.add_summary(summaries, i_)

            if i_ % 1000 == 0:
                inv.save(args.logdir, i_)

        except tf.errors.OutOfRangeError:
            break
        except KeyboardInterrupt:
            inv.save(args.logdir, i_)
            inv.sess.close()
            sys.exit()
