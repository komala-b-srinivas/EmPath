import numpy as np

def extract_features(signal):
    # Feature extraction logic
    features = {}
    features['mean'] = np.mean(signal)
    features['std_dev'] = np.std(signal)
    features['max'] = np.max(signal)
    features['min'] = np.min(signal)
    # Additional features can be added
    return features
