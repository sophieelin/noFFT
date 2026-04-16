# %% [markdown]
# # Chroma and Chromagrams
# 
# Alexandre R.J. Francois

# %%
import numpy as np
from matplotlib import colormaps as mcm
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

from noFFT_utils import log_frequencies, alphas_heuristic, resonate_wrapper

# %%
#https://librosa.org/doc/main/recordings.html


# y, sr = sf.read(librosa.ex("brahms"), frames=100000)
filepath = "../xylo_scale.wav"
y, sr = sf.read(filepath)
# y, sr = sf.read(librosa.ex("sweetwaltz"))
# y, sr = sf.read(librosa.ex("libri1"))
# y, sr = sf.read(librosa.ex("libri2"))
# y, sr = sf.read(librosa.ex("libri3"))
# y, sr = sf.read(librosa.ex("robin"))

librosa.display.waveshow(y, sr=sr)
print(y.shape)
print(y.dtype)

# float_y = np.array(y, dtype=np.float32)
# print(float_y.dtype)

# %%
fmin = 32.70
n_freqs = 84
freqs_per_octave = 12
frequencies = log_frequencies(fmin=fmin, n_freqs=n_freqs, freqs_per_octave=freqs_per_octave)
alphas = alphas_heuristic(frequencies, sr=sr, k=1)
hop_length = 1
dhl = 512 # hop length for display

print(frequencies)

# %%
R = resonate_wrapper(y=y, sr=sr, frequencies=frequencies, alphas=alphas, hop_length=hop_length, output_type='powers')
print(R.shape, R.dtype)

R_db = librosa.power_to_db(R.T, ref=np.max)

librosa.display.specshow(
    R_db[:, ::dhl],
    sr=sr,
    fmin = frequencies[0],
    hop_length=dhl,
    y_axis="cqt_hz",
    x_axis="time",
)


# %%
# chromagram

# print(R.shape)
# print(R)

K = R.copy()
# print(K.shape)
# print(K)

# print(K[0].shape)
# print(K[10].shape)

numChroma = 12
numOctaves = int(K.shape[-1] / numChroma)
# print(numChroma, numOctaves)

C = K.reshape(K.shape[0], numOctaves, numChroma)
# print(C.shape)

C = C.sum(axis=1).T
# print(D.shape, D)

C = librosa.util.normalize(C, norm=np.inf, axis=-2)

# print(C)
# print(freqs[0:12])

# Single spectrogram
fig, ax = plt.subplots(figsize=(8, 2), dpi=100)
librosa.display.specshow(
    C[:,::dhl],
    sr=sr,
    hop_length=dhl,
    fmin = frequencies[0],
    bins_per_octave=12,
    y_axis="chroma",
    x_axis="s",
    ax=ax,
)
ax.set(title="Resonate Chromagram")

# %%
#my changes 
n_bins = 128           # total number of notes/bins
hop_length = dhl     
fmin = librosa.midi_to_hz(0) # lowest note
print(f"minimum note: ", fmin)

R = np.abs(librosa.cqt(
    y,
    sr=sr,
    hop_length=hop_length,
    fmin=fmin,
    n_bins=n_bins,
    bins_per_octave=12  # 12 notes in an octave
))

print(R.shape)
# Plot
fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
librosa.display.specshow(
    C,
    sr=sr,
    hop_length=hop_length,
    y_axis="linear",  # each bin is a note, not chroma
    x_axis="s",
    ax=ax
)
ax.set(title="128-Note")
plt.show()

# per sample 

# %%
plt.figure(figsize=(12, 3))
librosa.display.specshow(
    R, 
    x_axis=None,       # to use this as sample and not as time
    y_axis='linear',   # each row is one note, rahter than a chroma
    hop_length=hop_length,
)
plt.xlabel("Sample")
plt.ylabel("Note in hz")
plt.title("128-Note CQT (per sample)")
plt.colorbar()
plt.show()

# %%
print(R)
print(R.shape)

# %%
distance = np.linalg.norm(R[:,1:]-R[:,:-1], axis = 0)
plt.plot(distance)

# %%
# Librosa Chroma

# librosa.feature.chroma_stft(*, y=None, sr=22050, S=None, norm=inf, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', tuning=None, n_chroma=12, **kwargs)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=dhl)
print(chroma_stft.shape)

# librosa.feature.chroma_cqt(*, y=None, sr=22050, C=None, hop_length=512, fmin=None, norm=inf, threshold=0.0, tuning=None, n_chroma=12, n_octaves=7, window=None, bins_per_octave=36, cqt_mode='full')
chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=dhl, bins_per_octave=12)
print(chroma_cq.shape)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 5), dpi=100)
librosa.display.specshow(chroma_stft, hop_length=dhl, y_axis='chroma', x_axis='s', ax=ax[0])
ax[0].set(title='chroma_stft')
ax[0].label_outer()
img = librosa.display.specshow(chroma_cq, hop_length=dhl, y_axis='chroma', x_axis='s', ax=ax[1])
ax[1].set(title='chroma_cqt')
fig.colorbar(img, ax=ax)


# %%
# Resonate Chroma

# S = np.abs(librosa.stft(y))**2
# S = np.abs(librosa.stft(y, n_fft=4096))**2
# chroma_stft = librosa.feature.chroma_stft(S=S, sr=sr, n_chroma=12, n_fft=4096)

chromaR = librosa.feature.chroma_cqt(C=R.T, sr=sr, bins_per_octave=12)

print(sr, R.T.shape, chromaR.shape)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 5), dpi=100)
# img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax[0])
# img = librosa.display.specshow(librosa.power_to_db(R, ref=np.max), y_axis='log', x_axis='time', ax=ax[0])
librosa.display.specshow(
    librosa.power_to_db(R.T[:,::dhl], ref=np.max),
    sr=sr,
    hop_length=dhl,
    y_axis="cqt_hz",
    x_axis="s",
    ax=ax[0]
)
fig.colorbar(img, ax=[ax[0]])
ax[0].label_outer()

img = librosa.display.specshow(
    chromaR[:,::dhl],
    sr=sr,
    hop_length=dhl,
    y_axis='chroma',
    x_axis='s',
    ax=ax[1])
fig.colorbar(img, ax=[ax[1]])


# %%
fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(8, 8), dpi=300)
librosa.display.specshow(chroma_stft, hop_length=dhl, y_axis='chroma', x_axis='s', ax=ax[0])
ax[0].set(title='Librosa chroma_stft')
ax[0].label_outer()
img = librosa.display.specshow(chroma_cq, hop_length=dhl, y_axis='chroma', x_axis='s', ax=ax[1])
ax[1].set(title='Librosa chroma_cqt')
ax[1].label_outer()
img = librosa.display.specshow(
    chromaR[:,::dhl],
    sr=sr,
    hop_length=dhl,
    y_axis='chroma',
    x_axis='s',
    ax=ax[2])
ax[2].set(title='Resonate chroma')
fig.colorbar(img, ax=ax)



