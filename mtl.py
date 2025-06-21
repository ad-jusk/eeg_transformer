import mne
from scipy.io import loadmat
from scipy.signal import welch
from scripts.mtl.linear import MultiTaskLinear
from eeg_logger import logger
import numpy as np


def extract_basic_features_from_epochs(
    file_path: str, fmin: float = 7.0, fmax: float = 31.0, band_width: float = 2.0
) -> np.ndarray:
    # Load epochs
    epochs = mne.read_epochs(file_path, preload=True, verbose=False)
    y = epochs.events[:, -1]
    y = np.array([0 if label == 10 else 1 for label in y]).reshape(-1, 1)
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]

    n_epochs, n_channels, n_times = data.shape

    # Welch PSD for all epochs and channels
    psd_all = []
    for epoch in data:
        psd_epoch = []
        for ch_data in epoch:
            freqs, psd = welch(ch_data, fs=sfreq, nperseg=n_times)
            psd_epoch.append(psd)
        psd_all.append(psd_epoch)

    psd_all = np.array(psd_all)  # shape: (n_epochs, n_channels, n_freqs)

    # Extract features in defined frequency bands
    bands = np.arange(fmin, fmax, band_width)
    features = []

    for band_start in bands:
        band_end = band_start + band_width
        idx_band = np.where((freqs >= band_start) & (freqs < band_end))[0]
        band_power = np.mean(psd_all[:, :, idx_band], axis=2)  # (n_epochs, n_channels)
        features.append(band_power)

    # Concatenate over bands → shape: (n_epochs, n_channels * n_bands)
    features = np.concatenate(features, axis=1)
    features = np.log10(features + 1e-12)

    return features.T, y


X1, y1 = extract_basic_features_from_epochs("./preprocessed_data/BCI_IV_2b/S01/PB0101T-epo.fif")
X2, y2 = extract_basic_features_from_epochs("./preprocessed_data/BCI_IV_2b/S01/PB0102T-epo.fif")
X3, y3 = extract_basic_features_from_epochs("./preprocessed_data/BCI_IV_2b/S03/PB0301T-epo.fif")
X4, y4 = extract_basic_features_from_epochs("./preprocessed_data/BCI_IV_2b/S03/PB0302T-epo.fif")

data = np.empty(3, dtype=object)
data[0] = X1
data[1] = X2
data[2] = X3

labels = np.empty(3, dtype=object)
labels[0] = y1
labels[1] = y2
labels[2] = y3

linear = MultiTaskLinear()
linear.fit_prior(data, labels)

prior_predict_labels = linear.prior_predict(X4)
prior_acc = np.mean(prior_predict_labels == y4)
logger.info(f"Prior accuracy for new task: {prior_acc}")


# ###################################################
# data = loadmat("./testdata.mat")
# all_X = data["T_X2d"][0]
# all_y = data["T_y"][0]

# X_train, X_test = all_X[:4], all_X[4]
# y_train, y_test = all_y[:4], all_y[4]

# linear = MultiTaskLinear()

# linear.fit_prior(X_train, y_train)

# prior_predict_labels = linear.prior_predict(X_test)
# prior_acc = np.mean(prior_predict_labels == y_test)
# logger.info(f"Prior accuracy for new task: {prior_acc}")
