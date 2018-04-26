import tensorflow as tf
import tflearn
import numpy as np
import os
import sys
import librosa

from utils.audio_utils import (preemphasis, inv_preemphasis, griffin_lim,
                                 log_magnitude_postproc)
from utils.train_utils import load, save
from .cbhg import CBHG
import json

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


def invert_spectrogram(spec, magnitude_emphasis, preemphasis, n_fft, hop_size):
    stftm_pred = log_magnitude_postproc(spec.T, magnitude_emphasis)

    w_test = griffin_lim(
        stftm_pred,
        n_fft=n_fft,
        hop_size=hop_size)
    w_test = inv_preemphasis(w_test, preemphasis)

    return w_test


class Inverter():
    def __init__(self, n_mel=80, n_fft=1024, hop_size=160):
        self.n_fft = n_fft
        self.n_mel = n_mel
        self.hop_size = hop_size
        self.n_spec = n_fft // 2 + 1

        with open('inverter/params.json') as f:
            params = json.load(f)

        params["input_channels"] = self.n_mel
        params["output_fc_channels"] = self.n_spec

        with tf.Graph().as_default() as graph:
            with tf.device('/cpu:0'):
                self.graph = graph
                self.sess = tf.Session()
                tflearn.is_training(True, self.sess)

                self.global_i = tf.get_variable("global_i", dtype=tf.int32,
                                                trainable=False,
                                                initializer=0)
                self.inc_global_i = tf.assign_add(self.global_i, 1)


                self.mel = tf.placeholder(
                        tf.float32, (None, None, self.n_mel),
                        "mel") 
                self.spec = tf.placeholder(
                        tf.float32, (None, None, self.n_spec),
                        "spec") 

                cbhg = CBHG(**params)

                self.spec_pred, _ = cbhg.create_network(self.mel, None)

                # workaround for tflearn moving averages
                # using union to add missing moving vars to list
                all_var_list = list(set(tf.trainable_variables()) |
                                    set(tf.model_variables()))
                self.var_list = all_var_list

    def init_and_load(self, logdir):
        with self.graph.as_default():
            # workaround for tflearn moving averages
            # using union to add missing moving vars to list
            all_var_list = list(set(tf.trainable_variables()) |
                                set(tf.model_variables()))
            self.var_list = all_var_list

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            load(self.saver, self.sess, logdir)

    def save(self, logdir, i):
        with self.graph.as_default():
            save(self.saver, self.sess, logdir, i)

    def mel_to_wave(self, source):
        mel_ = np.stack([source], axis=0)
        inputs = {self.mel: mel_}
        spec_pred_, = self.sess.run([self.spec_pred], inputs)

        w = invert_spectrogram(spec_pred_[0], 1.2, .97,
                               self.n_fft, self.hop_size)
        w = w / np.abs(w).max()
        return w

    def loss(self):
        with tf.name_scope('mel_loss'):
            return tf.reduce_mean(tf.abs(self.spec - self.spec_pred))
