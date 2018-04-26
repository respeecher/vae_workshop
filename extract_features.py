import numpy as np
import librosa
import os
import random
import re
import sys
import argparse

from scipy import signal

from utils.audio_utils import preemphasis

EPS = 1e-10

description = (
        """Extract spectral features from audio files. The script will search
        for audio files and compute spectral features for each file (unless
        filters are specified) and preserve the same folder structure as in the
        audio directory.""")
parser = argparse.ArgumentParser(description=description)
parser.add_argument('audio_dir', type=str,
                    help='Directory with audio files')
parser.add_argument('target_dir', type=str,
                    help='Place to save feature files')

parser.add_argument('--dry_run', action='store_true',
                    help="Just find the files and don't write anything")
parser.add_argument('--filter_speakers', type=lambda s: s.split(','),
                    help='Comma-separated list of speakers to filter '
                         'filenames',
                    default='')
parser.add_argument('--sample_rate', type=int,
                    help='The input audio will be interpolated to this sample '
                         'rate.',
                    default=16000)
parser.add_argument('--hop_size', type=int,
                    help='Hop size for the STFT algorithm (in samples)',
                    default=160)
parser.add_argument('--n_fft', type=int,
                    help='Length of the window for the STFT algorithm (in '
                         'samples). The number of the resulting FFT '
                         'coefficients will be equal to (n_fft / 2) - 1.',
                    default=1024)
parser.add_argument('--n_filterbank', type=int,
                    help='Size of the mel filterbank.',
                    default=80)
parser.add_argument('--preemphasis', type=float,
                    help='Preemphasis coefficient.',
                    default=0.97)
parser.add_argument('--clip_before_log', type=float,
                    help='Clip filterbank powers to this value from below, '
                         'before taking logarithms.',
                    default=0.0001)
parser.add_argument('--trim_silence_db', type=float,
                    help='The threshold (in decibels) below reference to '
                         'consider as silence and trim it.',
                    default=20)

args = parser.parse_args()


def get_features(
        fname,
        sample_rate,
        n_fft,
        n_mel,
        hop_size,
        preemph,
        clip_value,
        trim_silence_db):

    au, _ = librosa.load(fname, sample_rate, dtype=np.float64)

    au, _ = librosa.effects.trim(au, trim_silence_db, np.max, 256, 128)

    au = librosa.util.normalize(au)

    if preemph:
        au = preemphasis(au, preemph)

    spec = librosa.stft(
            au,
            n_fft=n_fft,
            hop_length=hop_size)
    spec = np.abs(spec.T)**2
    mel_filters = librosa.filters.mel(sample_rate, n_fft, n_mel,
                                      fmin=125, fmax=7600)

    mel_spec = np.dot(mel_filters, spec.T).T
    mel_spec = np.clip(mel_spec, clip_value, None)

    spec = np.clip(spec, clip_value, None)

    return [arr.astype(np.float32) for arr in [au, spec, mel_spec]]

if args.dry_run:
    print("WARNING: working in dry-run mode")


fnames = []

for root, dirs, files in os.walk(args.audio_dir, followlinks=True):
    for f in files:
        full_path = os.path.join(root, f)
        passes = bool(re.search(r'\.(ogg|wav)$', f))
        passes = passes and any(
                [name in full_path for name in args.filter_speakers])
        if passes:
            fnames.append(os.path.join(root, f))

print("Found {} files to process".format(len(fnames)))

if os.path.exists(args.target_dir):
    raise OSError(
        "the target directory '{}' already exists!".format(args.target_dir))

if not args.dry_run:
    os.makedirs(args.target_dir)

n_files = len(fnames)

for i, fname in enumerate(fnames):
    au, spec, mel_spec = get_features(fname,
                                      args.sample_rate,
                                      args.n_fft,
                                      args.n_filterbank,
                                      args.hop_size,
                                      args.preemphasis,
                                      args.clip_before_log,
                                      args.trim_silence_db)

    feat_file = (re.sub(r'\.(ogg|wav)$', '.feat', fname)
                 .replace(os.path.normpath(args.audio_dir),
                          os.path.normpath(args.target_dir)))

    progress = i / n_files * 100
    print('{} -> {} [{:.1f}%]\r'.format(
        fname, feat_file, progress), end='', flush=True)


    if not args.dry_run:
        os.makedirs(os.path.dirname(feat_file), exist_ok=True)

        np.savez(feat_file,
                 au=au,
                 logspec=np.log(spec + EPS),
                 logmel=np.log(mel_spec + EPS))

print('\ndone')
