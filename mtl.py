from scipy.io import loadmat
from scripts.features_extract.welch import extract_welch_features_paper, extract_welch_features
from scripts.mtl.linear import MultiTaskLinear
from sklearn.model_selection import train_test_split
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


def prepare_feature_matrix_physionet(epochs_dir: str = "./epochs/Physionet") -> np.ndarray:
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


def prepare_feature_matrix_bci2a(epochs_dir: str = "./epochs/BCI_IV_2a") -> np.ndarray:
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


def prepare_feature_matrix_bci2b(epochs_dir: str = "./epochs/BCI_IV_2b") -> np.ndarray:
    if not os.path.exists(epochs_dir):
        logger.error(f"{epochs_dir} does not exist")
        return
    subject_folders = [f for f in os.listdir(epochs_dir)]

    X_all = np.empty(len(subject_folders) * 3, dtype=object)
    y_all = np.empty(len(subject_folders) * 3, dtype=object)

    for idx, subject in enumerate(subject_folders):
        subject_id = subject[1:]

        epochs_file1 = os.path.join(epochs_dir, subject, f"PB{subject_id}01T-epo.fif")
        epochs_file2 = os.path.join(epochs_dir, subject, f"PB{subject_id}02T-epo.fif")
        epochs_file3 = os.path.join(epochs_dir, subject, f"PB{subject_id}03T-epo.fif")

        X1, y1 = extract_welch_features_paper(epochs_file1)
        X2, y2 = extract_welch_features_paper(epochs_file2)
        X3, y3 = extract_welch_features_paper(epochs_file3)

        base_idx = idx * 3
        X_all[base_idx] = X1
        X_all[base_idx + 1] = X2
        X_all[base_idx + 2] = X3

        y_all[base_idx] = y1
        y_all[base_idx + 1] = y2
        y_all[base_idx + 2] = y3

    return X_all, y_all


X_all, y_all = prepare_feature_matrix_bci2b()
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=1, random_state=42, shuffle=True)
print(X_test.shape)

linear = MultiTaskLinear(num_its=200)
linear.fit_prior(X_train, y_train)

prior_predict_labels = linear.prior_predict(X_test[0])
prior_acc = np.mean(prior_predict_labels == y_test[0])
logger.info(f"Prior accuracy for new task: {prior_acc}")

fitted_new_linear = linear.fit_new_task(X_test[0], y_test[0])
new_accuracy = np.mean(fitted_new_linear["predict"](X_test[0]) == y_test[0])
logger.info(f"New task accuracy: {new_accuracy}")
