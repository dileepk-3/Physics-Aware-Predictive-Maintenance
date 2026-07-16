import os
import numpy as np
import scipy.io as sio

# ============================================
# CONSTANTS
# ============================================
FS = 12000  # Sampling frequency (Hz)

CLASS_MAP = {
    "Healthy": 0,
    "Ball":    1,
    "Inner":   2,
    "Outer":   3
}

CLASS_NAMES = {
    0: "Healthy",
    1: "Ball Fault",
    2: "Inner Race Fault",
    3: "Outer Race Fault"
}

# ============================================
# FIND SIGNAL KEY
# ============================================
def find_de_key(data):
    for key in data.keys():
        if "DE_time" in key:
            return key
    raise KeyError("No DE_time key found in this file")

# ============================================
# LOAD SINGLE FILE
# ============================================
def load_signal(file_path, key=None):
    data = sio.loadmat(file_path)
    if key is None:
        key = find_de_key(data)
    return data[key].flatten()

# ============================================
# SPLIT SIGNAL INTO SEGMENTS
# ============================================
def split_signal(signal, segment_size=2048, step=2048):
    segments = []
    for i in range(0, len(signal) - segment_size, step):
        segments.append(signal[i:i + segment_size])
    return segments

# ============================================
# LOAD FULL DATASET
# ============================================
def load_dataset(data_dir="data/CWRU"):
    X, y, file_names = [], [], []

    for class_name, label in CLASS_MAP.items():
        folder = os.path.join(data_dir, class_name)

        if not os.path.exists(folder):
            print(f"  Warning: folder not found -> {folder}")
            continue

        files = sorted([f for f in os.listdir(folder) if f.endswith(".mat")])
        print(f"Loading [{class_name}]: {len(files)} files found")

        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                signal = load_signal(fpath)
                segments = split_signal(signal)
                for seg in segments:
                    X.append(seg)
                    y.append(label)
                    file_names.append(fname)
            except Exception as e:
                print(f"  Skipping {fname}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset ready!")
    print(f"Segments : {X.shape[0]}")
    print(f"Segment size : {X.shape[1]}")
    print(f"Class distribution : {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, file_names