import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import pywt

class BiosignalPipeline:
    def __init__(self, signal):
        self.signal = signal

    # Time-Domain Features
    def time_domain_features(self):
        mean = np.mean(self.signal)
        variance = np.var(self.signal)
        skewness = pd.Series(self.signal).skew()
        kurtosis = pd.Series(self.signal).kurtosis()
        features = {'mean': mean, 'variance': variance, 'skewness': skewness, 'kurtosis': kurtosis}
        return features

    # Frequency-Domain Features
    def frequency_domain_features(self):
        fft_result = np.fft.fft(self.signal)
        magnitude = np.abs(fft_result)
        power_spectral_density = magnitude ** 2
        features = {'fft': fft_result, 'magnitude': magnitude, 'power_spectral_density': power_spectral_density}
        return features

    # Wavelet Features
    def wavelet_features(self):
        coeffs = pywt.wavedec(self.signal, 'db4')
        features = {f'wavelet_coeff_{i}': coeff for i, coeff in enumerate(coeffs)}
        return features

    # ECG-specific Features
    def ecg_features(self):
        peaks, _ = find_peaks(self.signal, height=0)
        rr_intervals = np.diff(peaks)
        features = {'num_peaks': len(peaks), 'mean_rr_interval': np.mean(rr_intervals) if len(rr_intervals) > 0 else 0}
        return features

    # EDA-specific Features
    def eda_features(self):
        peaks, _ = find_peaks(-self.signal, height=0)
        features = {'num_sweeps': len(peaks)}
        return features

    # EMG-specific Features
    def emg_features(self):
        energy = np.sum(self.signal ** 2)
        features = {'energy': energy}
        return features

    # Full feature extraction combining all
    def extract_features(self):
        features = {}
        features.update(self.time_domain_features())
        features.update(self.frequency_domain_features())
        features.update(self.wavelet_features())
        features.update(self.ecg_features())
        features.update(self.eda_features())
        features.update(self.emg_features())
        return features
