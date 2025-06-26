from scipy.io import loadmat
from scripts.features_extract.welch import extract_welch_features_paper, extract_welch_features
from scripts.mtl.linear import MultiTaskLinear
from eeg_logger import logger
import numpy as np
import os

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

# fitted_new_linear = linear.fit_new_task(X_test, y_test)
# new_accuracy = np.mean(fitted_new_linear["predict"](X_test) == y_test)
# logger.info(f"New task accuracy: {new_accuracy}")


def prepare_feature_matrix_physionet(epochs_dir: str) -> np.ndarray:
    if not os.path.exists(epochs_dir):
        logger.error(f"{epochs_dir} does not exist")
        return
    subject_folders = [f for f in os.listdir(epochs_dir)]

    X_all = np.empty(len(subject_folders), dtype=object)
    y_all = np.empty(len(subject_folders), dtype=object)
    for idx, subject in enumerate(subject_folders):
        epochs_file = os.path.join(epochs_dir, subject, f"PA{subject[1:]}-3s-epo.fif")
        X, y = extract_welch_features_paper(epochs_file)
        X_all[idx] = X
        y_all[idx] = y
    return X_all, y_all


def prepare_feature_matrix_bci2a(epochs_dir: str) -> np.ndarray:
    if not os.path.exists(epochs_dir):
        logger.error(f"{epochs_dir} does not exist")
        return
    subject_folders = [f for f in os.listdir(epochs_dir)]

    X_all = np.empty(len(subject_folders), dtype=object)
    y_all = np.empty(len(subject_folders), dtype=object)
    for idx, subject in enumerate(subject_folders):
        epochs_file = os.path.join(epochs_dir, subject, f"PA{subject[1:3]}T-epo.fif")
        X, y = extract_welch_features_paper(epochs_file)
        X_all[idx] = X
        y_all[idx] = y
    return X_all, y_all


X_all, y_all = prepare_feature_matrix_bci2a("./epochs/BCI_IV_2a")

X_train = np.empty(5, dtype=object)
X_train[0] = X_all[0][:, :24]
X_train[1] = X_all[0][:, 24:48]
X_train[2] = X_all[0][:, 48:72]
X_train[3] = X_all[0][:, 72:96]
X_train[4] = X_all[0][:, 96:120]

y_train = np.empty(5, dtype=object)
y_train[0] = y_all[0][:24]
y_train[1] = y_all[0][24:48]
y_train[2] = y_all[0][48:72]
y_train[3] = y_all[0][72:96]
y_train[4] = y_all[0][96:120]

X_test, y_test = X_all[0][:, 120:], y_all[0][120:]

linear = MultiTaskLinear()
linear.fit_prior(X_train, y_train)

prior_predict_labels = linear.prior_predict(X_test)
prior_acc = np.mean(prior_predict_labels == y_test)
logger.info(f"Prior accuracy for new task: {prior_acc}")

fitted_new_linear = linear.fit_new_task(X_test, y_test)
new_accuracy = np.mean(fitted_new_linear["predict"](X_test) == y_test)
logger.info(f"New task accuracy: {new_accuracy}")
