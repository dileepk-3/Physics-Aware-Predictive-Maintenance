import numpy as np
from src.features.statistical_features import statistical_features, STAT_FEATURE_NAMES
from src.features.frequency_features import frequency_features, FREQ_FEATURE_NAMES
from src.features.wavelet_features import wavelet_features, WAVELET_FEATURE_NAMES
from src.physics.bearing_physics import physics_features, PHYSICS_FEATURE_NAMES

# Combined feature names
FEATURE_NAMES = (STAT_FEATURE_NAMES + 
                 FREQ_FEATURE_NAMES + 
                 WAVELET_FEATURE_NAMES + 
                 PHYSICS_FEATURE_NAMES)

def extract_features(segment, shaft_rpm=1797):
    stat  = statistical_features(segment)
    freq  = frequency_features(segment)
    wave  = wavelet_features(segment)
    phys  = physics_features(segment, shaft_rpm=shaft_rpm)
    return stat + freq + wave + phys

def build_feature_matrix(X_raw, shaft_rpm=1797):
    print(f"Extracting features from {len(X_raw)} segments...")
    X_feat = []
    for i, seg in enumerate(X_raw):
        X_feat.append(extract_features(seg, shaft_rpm))
        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{len(X_raw)}")
    X_feat = np.array(X_feat)
    print(f"Feature matrix shape: {X_feat.shape}")
    print(f"Total features: {X_feat.shape[1]}")
    return X_feat