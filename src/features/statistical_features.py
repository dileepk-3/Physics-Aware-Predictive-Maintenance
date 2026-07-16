import numpy as np

def statistical_features(signal):
    x = signal - np.mean(signal)
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    std = np.std(x) + 1e-8
    mean = np.mean(x)
    kurtosis = np.mean((x - mean)**4) / (std**4)
    skewness = np.mean((x - mean)**3) / (std**3)
    crest_factor = peak / (rms + 1e-8)
    peak_to_peak = np.max(x) - np.min(x)
    impulse_factor = peak / (np.mean(np.abs(x)) + 1e-8)
    shape_factor = rms / (np.mean(np.abs(x)) + 1e-8)
    margin_factor = peak / ((np.mean(np.sqrt(np.abs(x))) + 1e-8)**2)

    return [
        rms,
        peak,
        std,
        mean,
        kurtosis,
        skewness,
        crest_factor,
        peak_to_peak,
        impulse_factor,
        shape_factor,
        margin_factor
    ]

STAT_FEATURE_NAMES = [
    "RMS", "Peak", "Std", "Mean",
    "Kurtosis", "Skewness", "Crest_Factor",
    "Peak_to_Peak", "Impulse_Factor",
    "Shape_Factor", "Margin_Factor"
]