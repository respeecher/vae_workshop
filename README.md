# beta-VAE on spectrograms
Code to replicate results from the VAE workshop held by Dmytro Bielievtsv and Grant Reaber from Respeecher at ODSC Conference 2018, Kyiv. The code might be pretty messed up, but hopefully can be helpful for educational purposes.

# Requirements
* tensorflow >= 1.5
* tflearn (I'm sorry)
* librosa
* scipy

# Usage
## 1) Feature extraction
Prepare a set of audio files in some folder (say, `audio/`) and split it into two subfolders `audio/train` and `audio/test`. The files can be in either `.wav` or `.ogg` format. Extract features as using `extract_features.py`:
```bash
$ cd vae_workshop
$ python extract_features.py ../audio/ ../audio_feat/
```
The command above will extract both compressed mel and full linear log-magnitude spectrograms and store them as `.npz` files in `audio_feat/`. See `python extract_features.py -h` for more details.

## 2) Train the VAE
Run the VAE training script:
```bash
$ python train_beta_vae.py checkpoints/vae/ ../audio_feat/train/ ../audio_feat/test/
```
On a 1080Ti it takes about 1.5 hours to train the network for 300k iterations. While it is training, it is interesting to take a look at the learning dynamics using tensorboard:
```bash
$ tensorboard --logdir checkpoints/
```

## 3) Train the mel spectrogram inverter
Mel-spectrogram is usually about 100 frames per second at 80 dimensions per frame. This is a very compressed representation, so to get nicely sounding waveforms one can use a separately trained neural network (a CBHG block, as in Wang, Yuxuan et al. "Tacotron: Towards End-to-End Speech Synthesis.", 2017) to convert from e.g. an 80-dim melspec representation to a 513-dim power spectrum, which can then be converted back to the waveform using the Griffin-Lim approximation (Daniel Griffin and Jae Lim. Signal estimation from modified short-time Fourier transform, 1984).

Run the training script:
```bash
$ python train_inverter.py checkpoints/inverter/ ../audio_feat/train/ ../audio_feat/test/
```

With a 2-hour single speaker dataset, it takes about 40k iterations to reach reasonable quality. This model is much larger, so it can take several hours to train.

## 4) Run the notebook and test the models
```bash
$ python jupyter-notebook
```

# TODO
* Get rid of tflearn
