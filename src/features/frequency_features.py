import numpy as np

FS = 12000

def frequency_features(signal):
    x = signal - np.mean(signal)
    N = len(x)
    fft_vals = np.abs(np.fft.rfft(x)) / N
    freqs = np.fft.rfftfreq(N, 1/FS)

    mask = (freqs > 5) & (freqs < 2000)
    f = freqs[mask]
    v = fft_vals[mask]

    total = np.sum(v) + 1e-8
    prob = v / total

    dominant_freq = f[np.argmax(v)]
    spectral_centroid = np.sum(f * v) / total
    spectral_energy = np.sum(v**2)
    spectral_entropy = -np.sum(prob * np.log(prob + 1e-8))
    low_band_ratio = np.sum(v[f < 100]) / total
    mid_band_ratio = np.sum(v[(f >= 100) & (f < 500)]) / total
    high_band_ratio = np.sum(v[f >= 500]) / total
    peak_amplitude = np.max(v)

    return [
        dominant_freq,
        spectral_centroid,
        spectral_energy,
        spectral_entropy,
        low_band_ratio,
        mid_band_ratio,
        high_band_ratio,
        peak_amplitude
    ]

FREQ_FEATURE_NAMES = [
    "Dominant_Freq",
    "Spectral_Centroid",
    "Spectral_Energy",
    "Spectral_Entropy",
    "Low_Band_Ratio",
    "Mid_Band_Ratio",
    "High_Band_Ratio",
    "Peak_Amplitude"
]