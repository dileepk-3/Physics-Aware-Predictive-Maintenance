import numpy as np
import pywt

def wavelet_features(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=3)

    features = []
    for c in coeffs:
        energy = np.sum(c**2)
        mean_abs = np.mean(np.abs(c))
        std = np.std(c)
        total = energy + 1e-8
        prob = c**2 / total
        entropy = -np.sum(prob * np.log(np.abs(prob) + 1e-8))

        features += [energy, mean_abs, std, entropy]

    return features

WAVELET_FEATURE_NAMES = []
for i in range(4):
    level = f"L{i}"
    WAVELET_FEATURE_NAMES += [
        f"Wavelet_{level}_Energy",
        f"Wavelet_{level}_MeanAbs",
        f"Wavelet_{level}_Std",
        f"Wavelet_{level}_Entropy"
    ]