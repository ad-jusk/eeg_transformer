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


def extract_welch_features_paper(fif_path):
    """
    Extract Welch PSD features from each epoch in a .fif file.

    Parameters:
    - fif_path: str, path to the .fif file containing MNE Epochs
    - picks: list of str, EEG channel names to include

    Returns:
    - X: np.array of shape (n_epochs, n_channels * n_freqs)
    """
    epochs = mne.read_epochs(fif_path, preload=True)

    sfreq = epochs.info["sfreq"]  # sampling frequency
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

    freqs = np.arange(7, 32, 2)  # 7, 9, ..., 31 Hz - 13 bins
    n_epochs, n_channels, n_times = data.shape
    nperseg = n_times // 2
    noverlap = nperseg // 2

    X = []
    y = epochs.events[:, -1]
    y = np.array([0 if label == 7 else 1 for label in y]).reshape(-1, 1)

    for epoch in data:
        features = []
        for ch_data in epoch:
            f, psd = welch(ch_data, fs=sfreq, window="hamming", nperseg=nperseg, noverlap=noverlap, nfft=2 * nperseg)
            psd_vals = np.interp(freqs, f, psd)
            features.extend(psd_vals)
        X.append(features)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return np.array(X).T, y
