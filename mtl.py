import warnings

import numpy as np
import moabb
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

from scripts.features_extract.welch import extract_welch_features
from scripts.mtl.linear import MultiTaskLinearClassifier

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

dataset = BNCI2014_001()
dataset.subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
paradigm = LeftRightImagery(channels=["C1", "C2", "C3", "C4", "Cz", "CP1", "CP2", "FC1", "FC2", "FC3", "FC4"])

X_all, y_all, metadata = paradigm.get_data(dataset, subjects=dataset.subject_list, return_epochs=False)
subjects = metadata["subject"].unique()
sessions = metadata["session"].unique()

X_train = np.empty(len(subjects), dtype=object)
X_test = np.empty(len(subjects), dtype=object)
y_train = np.empty(len(subjects), dtype=object)
y_test = np.empty(len(subjects), dtype=object)
label_map = {"left_hand": -1, "right_hand": 1}

for idx, subject in enumerate(subjects):

    train_mask = (metadata["subject"] == subject) & (metadata["session"] == "0train")
    test_mask = (metadata["subject"] == subject) & (metadata["session"] == "1test")

    X_sess_train = X_all[train_mask]
    y_sess_train = y_all[train_mask]
    X_sess_test = X_all[test_mask]
    y_sess_test = y_all[test_mask]

    X_train_feat = extract_welch_features(X_sess_train)
    X_test_feat = extract_welch_features(X_sess_test)
    y_sess_train = np.array([label_map[label] for label in y_sess_train])
    y_sess_test = np.array([label_map[label] for label in y_sess_test])

    X_train[idx] = X_train_feat.T
    X_test[idx] = X_test_feat.T
    y_train[idx] = y_sess_train.reshape(-1, 1)
    y_test[idx] = y_sess_test.reshape(-1, 1)

clf = MultiTaskLinearClassifier(regularization=0.5, zero_mean=False, cov_flag="l2")
clf.fit_sessions(X_train, y_train)

accuracies = []
for idx, subject in enumerate(subjects):
    clf.fit(X_train[idx], y_train[idx])
    acc = clf.score(X_test[idx], y_test[idx])
    accuracies.append(acc)
    print(f"Accuracy for subject {idx}: {acc:.2f}")
print(f"Mean accuracy across subjects: {np.mean(accuracies):.2f}")
