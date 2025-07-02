import numpy as np
from scipy.signal import welch, detrend
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def extract_welch_features(
    data: np.ndarray, sfreq: float = 250, apply_log: bool = True, normalize: bool = True
) -> np.ndarray:
    """
    Extracts log bandpower features using Welch's method.

    Parameters:
        data       : ndarray of shape (n_trials, n_channels, n_times)
        sfreq      : Sampling frequency
        bands      : List of (fmin, fmax) tuples. Defaults to mu/beta split.
        apply_log  : Whether to apply log10 transform
        normalize  : Whether to z-score features per trial

    Returns:
        features   : ndarray of shape (n_trials, n_channels * n_bands)
    """

    bands = [
        (7, 9),
        (9, 11),
        (11, 13),
        (13, 15),
        (15, 18),
        (18, 21),
        (21, 26),
        (26, 30),
    ]
    n_trials, n_channels, n_times = data.shape
    features = []

    for trial in range(n_trials):
        trial_features = []
        for ch in range(n_channels):
            signal = detrend(data[trial, ch])
            freqs, Pxx = welch(signal, fs=sfreq, nperseg=min(256, n_times), window="hamming")

            for fmin, fmax in bands:
                band_mask = (freqs >= fmin) & (freqs <= fmax)
                if not np.any(band_mask):
                    trial_features.append(0.0)
                    continue
                band_power = np.mean(Pxx[band_mask])
                if apply_log:
                    band_power = np.log10(band_power + 1e-10)
                trial_features.append(band_power)

        trial_features = np.array(trial_features)
        if normalize:
            trial_features = (trial_features - np.mean(trial_features)) / (np.std(trial_features) + 1e-10)
        features.append(trial_features)

    features_array = np.array(features)
    return features_array
