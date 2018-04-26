import tensorflow as tf
import numpy as np
import os
import sys
from random import shuffle
import argparse

from utils.train_utils import *
from model import create_vae_with_elbo_loss

config = tf.ConfigProto(device_count = {'GPU': 1})

parser = argparse.ArgumentParser()
parser.add_argument('logdir', type=str,
                    help='Directory to store checkpoints and summaries')
parser.add_argument('train_dir', type=str,
                    help='Directory with training data')
parser.add_argument('test_dir', type=str,
                    help='Directory with test data')
args = parser.parse_args()


batch_size = 128
segment_length = 20
segment_channels = 80

encoder_units = 256
decoder_units = 256

num_epochs = 1200

num_latents = 32
beta = 8.0


segment = tf.placeholder(
        tf.float32, shape=(batch_size, segment_length, segment_channels))

total_loss, outputs = create_vae_with_elbo_loss(
        segment, segment_channels,
        encoder_units, decoder_units, num_latents,
        beta)
outs = outputs['encoder_outs']
z_prior = outputs['z_prior']
z = outputs['z_posterior_sample']
segment_pred = outputs['x_reconst_mean']

optimizer = tf.train.AdamOptimizer()
updates = optimizer.minimize(total_loss)


loss_sum = tf.summary.scalar("total_loss", total_loss)
random_segment_sum = plot_segments(segment_pred, "random_segments")
reconst_segment_sum = plot_segments(
        tf.concat((segment, segment_pred), axis=2),
        "reconstructed_segments")
outs = tf.expand_dims(outs[:1], -1) 
outs_sum = tf.summary.image("encoder_outs", outs)

train_ds = create_batched_dataset(args.train_dir, batch_size, segment_channels,
                                  segment_length, shuffle=False)
test_ds = create_batched_dataset(args.test_dir, batch_size, segment_channels,
                                 segment_length, shuffle=False).repeat()

train_iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                                 train_ds.output_shapes)
next_train_batch = train_iterator.get_next()
next_test_batch = test_ds.make_one_shot_iterator().get_next()

training_init_op = train_iterator.make_initializer(train_ds)

global_i = tf.get_variable("global_i", dtype=tf.int32, trainable=False,
                           initializer=0)
inc_global_i = tf.assign_add(global_i, 1)

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
load(saver, sess, args.logdir)
summary_writer = tf.summary.FileWriter(args.logdir, sess.graph)

test_loss_ = np.inf
train_loss_ = np.inf
for epoch in range(num_epochs):
    # Initialize an iterator over the training dataset.
    sess.run(training_init_op)
    while True:
        sess.run(inc_global_i)
        i_ = sess.run(global_i)
        try:
            if i_ % 100 != 0:
                train_seg_ = sess.run(next_train_batch)
                train_loss_, _ = sess.run(
                        [total_loss, updates], {segment: train_seg_})
            else:
                test_seg_, z_prior_ = sess.run(
                        [next_test_batch, z_prior])
                test_loss_, loss_sum_, reconst_segment_sum_, outs_sum_ = (
                        sess.run([total_loss, loss_sum,
                                  reconst_segment_sum, outs_sum],
                                 {segment: test_seg_}))
                random_segment_sum_ = sess.run(
                        random_segment_sum, {z: z_prior_})
                summary_writer.add_summary(loss_sum_, i_)
                summary_writer.add_summary(reconst_segment_sum_, i_)
                summary_writer.add_summary(random_segment_sum_, i_)
                summary_writer.add_summary(outs_sum_, i_)

            if i_ % 10000 == 0:
                save(saver, sess, args.logdir, i_)

            print("Epoch: {}, step: {},  train_loss: {:.2f}, test_loss: {:.2f}"
                  .format(epoch, i_, train_loss_, test_loss_))

        except tf.errors.OutOfRangeError:
            break
        except KeyboardInterrupt:
            save(saver, sess, args.logdir, i_)
            sess.close()
            sys.exit()
