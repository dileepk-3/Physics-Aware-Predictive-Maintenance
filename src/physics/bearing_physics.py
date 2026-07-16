import numpy as np

# ============================================
# CWRU BEARING SPECIFICATIONS
# ============================================
BEARING_SPECS = {
    "CWRU": {
        "pitch_diameter": 1.748,
        "ball_diameter": 0.3126,
        "num_balls": 9,
        "contact_angle": 0
    }
}

# ============================================
# COMPUTE FAULT FREQUENCIES
# ============================================
def compute_fault_frequencies(shaft_rpm, specs=None):
    if specs is None:
        specs = BEARING_SPECS["CWRU"]

    shaft_hz = shaft_rpm / 60.0
    d = specs["ball_diameter"]
    D = specs["pitch_diameter"]
    n = specs["num_balls"]
    angle = np.radians(specs["contact_angle"])
    ratio = (d / D) * np.cos(angle)

    BPFO = (n / 2) * shaft_hz * (1 - ratio)
    BPFI = (n / 2) * shaft_hz * (1 + ratio)
    BSF  = (D / (2 * d)) * shaft_hz * (1 - ratio**2)
    FTF  = (shaft_hz / 2) * (1 - ratio)

    return {
        "BPFO": round(BPFO, 2),
        "BPFI": round(BPFI, 2),
        "BSF":  round(BSF, 2),
        "FTF":  round(FTF, 2),
        "shaft_hz": round(shaft_hz, 2)
    }

# ============================================
# ENERGY AROUND A FREQUENCY
# ============================================
def energy_around_freq(fft_vals, freqs, target_freq, bandwidth=20):
    mask = (freqs >= target_freq - bandwidth) & \
           (freqs <= target_freq + bandwidth)
    return np.sum(fft_vals[mask]**2)

# ============================================
# PART B - HARMONIC ANALYSIS
# ============================================
def harmonic_analysis(fft_vals, freqs, fault_freq, n_harmonics=3, bandwidth=10):
    """
    Checks energy at harmonics of a fault frequency.
    e.g. BPFO, 2xBPFO, 3xBPFO
    Returns list of energies at each harmonic.
    """
    harmonic_energies = []
    for h in range(1, n_harmonics + 1):
        target = fault_freq * h
        energy = energy_around_freq(fft_vals, freqs, target, bandwidth)
        harmonic_energies.append(energy)
    return harmonic_energies

def total_harmonic_energy(fft_vals, freqs, fault_freq, n_harmonics=3):
    energies = harmonic_analysis(fft_vals, freqs, fault_freq, n_harmonics)
    return sum(energies)

# ============================================
# PART B - SIDEBAND ANALYSIS
# ============================================
def sideband_analysis(fft_vals, freqs, fault_freq, shaft_hz, n_sidebands=3, bandwidth=5):
    """
    Sidebands appear at fault_freq +/- k * shaft_hz.
    These indicate modulation — a strong sign of real fault.
    Returns total sideband energy.
    """
    sideband_energies = []
    for k in range(1, n_sidebands + 1):
        upper = fault_freq + k * shaft_hz
        lower = fault_freq - k * shaft_hz
        if lower > 0:
            e_lower = energy_around_freq(fft_vals, freqs, lower, bandwidth)
            sideband_energies.append(e_lower)
        e_upper = energy_around_freq(fft_vals, freqs, upper, bandwidth)
        sideband_energies.append(e_upper)
    return sum(sideband_energies)

# ============================================
# PART C - PHYSICS FEATURES
# ============================================
def physics_features(signal, shaft_rpm=1797, fs=12000):
    fault_freqs = compute_fault_frequencies(shaft_rpm)

    x = signal - np.mean(signal)
    N = len(x)
    fft_vals = np.abs(np.fft.rfft(x)) / N
    freqs = np.fft.rfftfreq(N, 1/fs)

    shaft_hz = fault_freqs["shaft_hz"]

    # Basic energy around fault frequencies
    bpfo_energy = energy_around_freq(fft_vals, freqs, fault_freqs["BPFO"])
    bpfi_energy = energy_around_freq(fft_vals, freqs, fault_freqs["BPFI"])
    bsf_energy  = energy_around_freq(fft_vals, freqs, fault_freqs["BSF"])
    ftf_energy  = energy_around_freq(fft_vals, freqs, fault_freqs["FTF"])

    total_energy = np.sum(fft_vals**2) + 1e-8
    fault_energy_ratio = (bpfo_energy + bpfi_energy + bsf_energy) / total_energy

    bpfo_peak = fft_vals[np.argmin(np.abs(freqs - fault_freqs["BPFO"]))]
    bpfi_peak = fft_vals[np.argmin(np.abs(freqs - fault_freqs["BPFI"]))]

    # Harmonic features
    bpfo_harmonic = total_harmonic_energy(fft_vals, freqs, fault_freqs["BPFO"])
    bpfi_harmonic = total_harmonic_energy(fft_vals, freqs, fault_freqs["BPFI"])
    bsf_harmonic  = total_harmonic_energy(fft_vals, freqs, fault_freqs["BSF"])

    # Sideband features
    bpfo_sideband = sideband_analysis(fft_vals, freqs, fault_freqs["BPFO"], shaft_hz)
    bpfi_sideband = sideband_analysis(fft_vals, freqs, fault_freqs["BPFI"], shaft_hz)

    return [
        bpfo_energy,
        bpfi_energy,
        bsf_energy,
        ftf_energy,
        fault_energy_ratio,
        bpfo_peak,
        bpfi_peak,
        bpfo_harmonic,
        bpfi_harmonic,
        bsf_harmonic,
        bpfo_sideband,
        bpfi_sideband
    ]

PHYSICS_FEATURE_NAMES = [
    "BPFO_Energy",
    "BPFI_Energy",
    "BSF_Energy",
    "FTF_Energy",
    "Fault_Energy_Ratio",
    "BPFO_Peak",
    "BPFI_Peak",
    "BPFO_Harmonic_Energy",
    "BPFI_Harmonic_Energy",
    "BSF_Harmonic_Energy",
    "BPFO_Sideband_Energy",
    "BPFI_Sideband_Energy"
]

# ============================================
# PART D - PHYSICS VALIDATION
# ============================================
def physics_validation(signal, predicted_class, shaft_rpm=1797, fs=12000):
    fault_freqs = compute_fault_frequencies(shaft_rpm)
    shaft_hz = fault_freqs["shaft_hz"]

    x = signal - np.mean(signal)
    N = len(x)
    fft_vals = np.abs(np.fft.rfft(x)) / N
    freqs = np.fft.rfftfreq(N, 1/fs)

    total = np.sum(fft_vals**2) + 1e-8

    if predicted_class == 0:
        return {
            "confidence": 1.0,
            "message": "Healthy - no fault frequency expected"
        }

    class_to_freq = {
        1: fault_freqs["BSF"],
        2: fault_freqs["BPFI"],
        3: fault_freqs["BPFO"]
    }

    target_freq = class_to_freq.get(predicted_class)
    if target_freq is None:
        return {"confidence": 0.0, "message": "Unknown class"}

    # Check fundamental + harmonics + sidebands
    fund_energy    = energy_around_freq(fft_vals, freqs, target_freq)
    harmonic_e     = total_harmonic_energy(fft_vals, freqs, target_freq)
    sideband_e     = sideband_analysis(fft_vals, freqs, target_freq, shaft_hz)

    total_fault_e  = fund_energy + harmonic_e + sideband_e
    confidence     = min(total_fault_e / total * 1000, 1.0)

    return {
        "target_freq":      target_freq,
        "fundamental_energy": fund_energy,
        "harmonic_energy":  harmonic_e,
        "sideband_energy":  sideband_e,
        "confidence":       round(confidence, 4),
        "message":          f"Physics confidence: {round(confidence*100, 2)}%"
    }

# ============================================
# HARMONIC + SIDEBAND PLOT DATA
# ============================================
def get_harmonic_sideband_markers(fault_freqs, n_harmonics=3, n_sidebands=2):
    """
    Returns all harmonic and sideband frequencies
    for plotting on FFT.
    """
    markers = {}
    shaft_hz = fault_freqs["shaft_hz"]

    for fname in ["BPFO", "BPFI", "BSF"]:
        ff = fault_freqs[fname]
        markers[f"{fname}_1x"] = ff
        for h in range(2, n_harmonics + 1):
            markers[f"{fname}_{h}x"] = ff * h
        for k in range(1, n_sidebands + 1):
            markers[f"{fname}+{k}x_shaft"] = ff + k * shaft_hz
            if ff - k * shaft_hz > 0:
                markers[f"{fname}-{k}x_shaft"] = ff - k * shaft_hz

    return markers