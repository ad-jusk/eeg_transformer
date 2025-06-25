from scipy.io import loadmat
from scripts.features_extract.welch import extract_welch_features, extract_welch_features_paper
from scripts.mtl.linear import MultiTaskLinear
from eeg_logger import logger
import numpy as np

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

X1, y1 = extract_welch_features("./epochs/BCI_IV_2a/S01/PA01T-epo.fif")
X2, y2 = extract_welch_features("./epochs/BCI_IV_2a/S02/PA02T-epo.fif")
X3, y3 = extract_welch_features("./epochs/BCI_IV_2a/S03/PA03T-epo.fif")
X4, y4 = extract_welch_features("./epochs/BCI_IV_2a/S04/PA04T-epo.fif")
X5, y5 = extract_welch_features("./epochs/BCI_IV_2a/S05/PA05T-epo.fif")
X6, y6 = extract_welch_features("./epochs/BCI_IV_2a/S06/PA06T-epo.fif")
X7, y7 = extract_welch_features("./epochs/BCI_IV_2a/S07/PA07T-epo.fif")
X8, y8 = extract_welch_features("./epochs/BCI_IV_2a/S08/PA08T-epo.fif")
X9, y9 = extract_welch_features("./epochs/BCI_IV_2a/S09/PA09T-epo.fif")

X_all = np.empty(8, dtype=object)
X_all[0] = X1
X_all[1] = X2
X_all[2] = X3
X_all[3] = X4
X_all[4] = X5
X_all[5] = X6
X_all[6] = X7
X_all[7] = X8

y_all = np.empty(8, dtype=object)
y_all[0] = y1
y_all[1] = y2
y_all[2] = y3
y_all[3] = y4
y_all[4] = y5
y_all[5] = y6
y_all[6] = y7
y_all[7] = y8

linear = MultiTaskLinear()
linear.fit_prior(X_all, y_all)

prior_predict_labels = linear.prior_predict(X9)
prior_acc = np.mean(prior_predict_labels == y9)
logger.info(f"Prior accuracy for new task: {prior_acc}")

fitted_new_linear = linear.fit_new_task(X9, y9)
new_accuracy = np.mean(fitted_new_linear["predict"](X9) == y9)
logger.info(f"New task accuracy: {new_accuracy}")
