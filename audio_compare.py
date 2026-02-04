import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft
from scipy.interpolate import interp1d
from scipy.spatial.distance import cosine
import librosa

audio1, fs1 = librosa.load("reference.wav.ogg", sr=None, mono=True)
audio2, fs2 = librosa.load("pattern.wav.ogg", sr=None, mono=True)

# Convert stereo to mono
if audio1.ndim > 1:
    audio1 = np.mean(audio1, axis=1)

if audio2.ndim > 1:
    audio2 = np.mean(audio2, axis=1)

# Normalize audio
audio1 = audio1 / np.max(np.abs(audio1))
audio2 = audio2 / np.max(np.abs(audio2))

# FFT
fft1 = np.abs(fft(audio1))
fft2 = np.abs(fft(audio2))

# Interpolation to same length
min_len = min(len(fft1), len(fft2))

x1 = np.linspace(0, 1, len(fft1))
x2 = np.linspace(0, 1, len(fft2))
x_new = np.linspace(0, 1, min_len)

fft1_interp = interp1d(x1, fft1, fill_value="extrapolate")(x_new)
fft2_interp = interp1d(x2, fft2, fill_value="extrapolate")(x_new)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(fft1_interp, label="Reference")
plt.plot(fft2_interp, label="Pattern")
plt.xlabel("Frequency Bins")
plt.ylabel("Magnitude")
plt.title("Frequency Domain Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# Similarity score
similarity = 1 - cosine(fft1_interp, fft2_interp)
print("Similarity Score:", similarity)

