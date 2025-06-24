import mne
from scipy.io import loadmat
from scripts.mtl.linear import MultiTaskLinear
from eeg_logger import logger
import numpy as np

data = loadmat("./testdata.mat")
all_X = data["T_X2d"][0]
all_y = data["T_y"][0]

X_train, X_test = all_X[:4], all_X[4]
y_train, y_test = all_y[:4], all_y[4]

linear = MultiTaskLinear()

linear.fit_prior(X_train, y_train)

prior_predict_labels = linear.prior_predict(X_test)
prior_acc = np.mean(prior_predict_labels == y_test)
logger.info(f"Prior accuracy for new task: {prior_acc}")

fitted_new_linear = linear.fit_new_task(X_test, y_test)
new_accuracy = np.mean(fitted_new_linear["predict"](X_test) == y_test)
logger.info(f"New task accuracy: {new_accuracy}")
