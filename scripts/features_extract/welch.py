import numpy as np
from scipy.signal import welch
from eeg_logger import logger


def extract_welch_features(
    bands: list[tuple[int, int]],
    data: np.ndarray,
    sfreq: float = 250,
    apply_log: bool = True,
) -> np.ndarray:
    """
    Extracts frequency-domain features from EEG trials using Welch's method.

    This function computes bandpower features from EEG signals by estimating
    the power spectral density (PSD) for each channel using Welch's method.
    The power within predefined frequency bands is averaged and optionally
    log-transformed.

    Parameters:
        bands: list[tuple[int, int]]
            bands for Welch method

        data : np.ndarray
            EEG data array of shape (n_trials, n_channels, n_times),
            where:
              - n_trials   = number of epochs/trials
              - n_channels = number of EEG channels
              - n_times    = number of time samples per trial

        sfreq : float, default=250
            Sampling frequency of the EEG signal in Hz. Used to compute
            the frequency resolution for Welch's method.

        apply_log : bool, default=True
            If True, applies a logarithmic transform (log1p) to the computed
            bandpower to reduce skewness and stabilize variance. Useful when
            features span several orders of magnitude.

    Returns:
        features : np.ndarray
            Feature matrix of shape (n_trials, n_channels * n_bands),
            where each feature corresponds to the average power in a specific
            frequency band for a particular channel.
    """

    n_trials, n_channels, n_times = data.shape
    features = []

    for trial in range(n_trials):
        trial_features = []
        for ch in range(n_channels):
            freqs, Pxx = welch(data[trial, ch], fs=sfreq, nperseg=256, window="hamming")
            for fmin, fmax in bands:
                band_mask = (freqs >= fmin) & (freqs <= fmax)
                band_power = np.mean(Pxx[band_mask])
                if apply_log:
                    band_power = np.log1p(band_power)
                trial_features.append(band_power)

        features.append(trial_features)
    features = np.array(features)
    logger.info(f"Welch features extracted, vector's shape: {features.shape}")

    return features
