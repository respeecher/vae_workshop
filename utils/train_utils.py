import tensorflow as tf
from random import shuffle
import numpy as np
import os


def find_files(directory, pattern, relative=False):
    '''Recursively finds all files matching the pattern.

    Args:
        relative (string): whether to return a path relative to the directory passed
    '''
    files = []
    for root, dirnames, filenames in os.walk(directory, followlinks=True):
        for filename in fnmatch.filter(filenames, pattern):
            fname = os.path.join(root, filename)
            if relative:
                fname = os.path.relpath(fname, directory)
            files.append(fname)
    # Finding no files is usually an error we want to catch as early as possible.
    # If it is sometimes useful we can add an explicit option to allow it.
    assert len(files) > 0, 'find_files found no files'
    return files


def get_random_spectrogram(folder):
    fnames = os.listdir(folder)
    shuffle(fnames)
    for f in fnames:
        yield np.load(os.path.join(folder, f))['logmel']


def split_into_segments(ds, segment_length, segment_channels):
    def split(spectrogram):
        spec_length = tf.shape(spectrogram)[0]
        segments = tf.reshape(
                spectrogram[:spec_length - spec_length % segment_length],
                (-1, segment_length, segment_channels))
        return tf.data.Dataset.from_tensor_slices(segments)
    return ds.flat_map(split)


def create_batched_dataset(folder, batch_size, segment_channels,
                           segment_length, shuffle=True):
    ds = tf.data.Dataset.from_generator(
            lambda: get_random_spectrogram(folder),
            tf.float32, (None, segment_channels))
    segments = split_into_segments(ds, segment_length, segment_channels)
    if shuffle:
        segments = segments.shuffle(100000)

    ds = segments.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
    return ds.prefetch(100000)


def plot_segments(segment_batch, name, N=20):
    part = tf.reshape(
            segment_batch[:N, :, :], (-1, tf.shape(segment_batch)[2]))
    img_mat = tf.transpose(part)[::-1, :]
    img_mat = tf.expand_dims(img_mat, 0)
    img_mat = tf.expand_dims(img_mat, 3)
    return tf.summary.image(name, img_mat)


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir))
    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="", flush=True)

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


