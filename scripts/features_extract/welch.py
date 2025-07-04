import numpy as np
from scipy.signal import welch, detrend


def extract_welch_features(
    data: np.ndarray,
    sfreq: float = 250,
    apply_log: bool = True,
) -> np.ndarray:
    """
    Extracts frequency-domain features from EEG trials using Welch's method.

    This function computes bandpower features from EEG signals by estimating
    the power spectral density (PSD) for each channel using Welch's method.
    The power within predefined frequency bands is averaged and optionally
    log-transformed. These features are commonly used in motor imagery
    classification and other brain-computer interface (BCI) tasks.

    Parameters:
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

    Notes:
        - Welch's method is applied using a Hamming window with a maximum
          segment length of 256 samples or the full trial duration (whichever is smaller).
        - The bandpower is computed across the following 8 frequency bands:
            (7-9), (9-11), (11-13), (13-15), (15-18), (18-21), (21-26), (26-30) Hz
        - Features are returned unnormalized. Consider normalization if required by your model.
        - Detrending is applied to each channel before PSD estimation to remove linear trends.
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
            freqs, Pxx = welch(signal, fs=sfreq, nperseg=min(256, n_times), window="hamming", scaling="spectrum")

            for fmin, fmax in bands:
                band_mask = (freqs >= fmin) & (freqs <= fmax)
                if not np.any(band_mask):
                    trial_features.append(0.0)
                    continue
                band_power = np.mean(Pxx[band_mask])
                if apply_log:
                    band_power = np.log1p(band_power)
                trial_features.append(band_power)

        features.append(trial_features)

    return np.array(features)
