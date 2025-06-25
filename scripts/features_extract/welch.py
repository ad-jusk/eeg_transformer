import mne
import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler


def extract_welch_features(fif_file, fmin=7, fmax=31, n_fft=256, normalize=True):
    """
    Extract Welch PSD features from epoched EEG data stored in a .fif file.

    Args:
        fif_file (str): Path to the .fif file containing MNE Epochs.
        fmin (float): Minimum frequency to consider (e.g. 8 Hz).
        fmax (float): Maximum frequency to consider (e.g. 30 Hz).
        n_fft (int): Length of FFT window.
        normalize (bool): Whether to normalize features (default: True).

    Returns:
        X (np.ndarray): Normalized feature matrix of shape (n_channels * n_freqs, n_epochs)
        y (np.ndarray): Label array of shape (n_epochs, 1)
    """
    epochs = mne.read_epochs(fif_file, preload=True, verbose=False)

    # Compute PSD using Welch method
    psd = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft)

    # Extract the PSD data and frequency bins
    psds = psd.get_data()  # shape: (n_epochs, n_channels, n_freqs)

    # Flatten features to (n_epochs, n_channels * n_freqs)
    X = psds.reshape(psds.shape[0], -1)

    # Normalize features per column (across all epochs)
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Transpose to (n_features, n_epochs)
    X = X.T

    # Get labels as shape (n_epochs, 1)
    y = epochs.events[:, -1]
    y = np.array([0 if label == 7 else 1 for label in y]).reshape(-1, 1)

    return X, y


def extract_welch_features_paper(fif_file, fmin=7, fmax=31, window_type="hann"):
    """
    Extract Welch PSD features from EEG epochs using paper-specific configuration:
    - Welch method with 50% overlap
    - Hanning window of length N/2
    - Frequency range: 7 - 31 Hz
    - Binned every 2 Hz (one feature per bin per channel)

    Args:
        fif_file (str): Path to .fif file with MNE Epochs.
        fmin (float): Minimum frequency for PSD (default: 7 Hz).
        fmax (float): Maximum frequency for PSD (default: 31 Hz).
        window_type (str): Type of window to use (e.g., 'hann', 'hamming', 'boxcar').

    Returns:
        X (np.ndarray): Feature matrix of shape (n_epochs, n_channels * n_bins), normalized.
        y (np.ndarray): Labels of shape (n_epochs, 1)
        bin_edges (np.ndarray): The frequency bin edges used.
    """
    epochs = mne.read_epochs(fif_file, preload=True)
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    n_epochs, n_channels, n_times = data.shape

    # Welch parameters
    win_len = n_times // 2
    noverlap = win_len // 2

    # Frequency bins: every 2 Hz from fmin to fmax
    bin_edges = np.arange(fmin, fmax + 1, 2)  # e.g., [7, 9, 11, ..., 31]
    n_bins = len(bin_edges) - 1
    X = np.zeros((n_epochs, n_channels * n_bins))

    for i in range(n_epochs):
        features = []
        for ch in range(n_channels):
            freqs, psd = welch(data[i, ch, :], fs=sfreq, window=window_type, nperseg=win_len, noverlap=noverlap)

            # Extract PSD values within fmin-fmax
            psd = psd[(freqs >= fmin) & (freqs <= fmax)]
            freqs = freqs[(freqs >= fmin) & (freqs <= fmax)]

            # Bin PSDs into 2 Hz ranges
            binned = []
            for j in range(n_bins):
                bin_mask = (freqs >= bin_edges[j]) & (freqs < bin_edges[j + 1])
                if np.any(bin_mask):
                    binned.append(np.mean(psd[bin_mask]))
                else:
                    binned.append(0.0)
            features.extend(binned)
        X[i, :] = features

    # Normalize features per epoch
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Labels
    y = epochs.events[:, -1]
    y = np.array([0 if label == 7 else 1 for label in y]).reshape(-1, 1)

    return X.T, y
