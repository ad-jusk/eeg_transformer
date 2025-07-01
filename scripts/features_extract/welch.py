import mne
import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def extract_welch_features(data: np.ndarray, sfreq: float = 250):

    freqs_of_interest = np.arange(7, 32, 2)  # 7, 9 , 11, ..., 31 Hz

    n_trials, n_channels, _ = data.shape
    features = []

    for trial in range(n_trials):
        trial_features = []
        for ch in range(n_channels):
            f, Pxx = welch(data[trial, ch], fs=sfreq, nperseg=sfreq * 2, window="hamming")  # 2 second windows
            # For each freq of interest, find nearest frequency in f and take corresponding power
            power_vals = []
            for target_f in freqs_of_interest:
                idx = np.argmin(np.abs(f - target_f))
                power_vals.append(Pxx[idx])
            trial_features.extend(power_vals)
        features.append(trial_features)
    f = np.array(features)
    return f
