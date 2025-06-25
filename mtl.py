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

linear = MultiTaskLinear()
linear.fit_prior(X_all[:8], y_all[:8])

prior_predict_labels = linear.prior_predict(X_all[8])
prior_acc = np.mean(prior_predict_labels == y_all[8])
logger.info(f"Prior accuracy for new task: {prior_acc}")

fitted_new_linear = linear.fit_new_task(X_all[8], y_all[8])
new_accuracy = np.mean(fitted_new_linear["predict"](X_all[8]) == y_all[8])
logger.info(f"New task accuracy: {new_accuracy}")
