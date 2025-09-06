import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from eeg_logger import logger
from scripts.multi_task.linear import MultiTaskLinear
from sklearn.metrics import accuracy_score


class MultiTaskLinearWithSelectionClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn-compatible classifier designed to handle learning from multiple tasks and apply transfer learning to a new one.
    It wraps around a base model and provides methods to:
    - Fit a prior model using multiple sessions (fit_sessions)
    - Adapt the prior to a new task with selecting weight vectors (fit)
    - Make predictions using either the prior or task-specific model (predict)
    - Evaluate the accuracy (score)
    """

    def __init__(self, num_its=100, regularization=0.5, cov_flag="l2", zero_mean=True, max_it_var=0.0001, verbose=True):
        self.base_model = MultiTaskLinear(
            num_its=num_its,
            regularization=regularization,
            cov_flag=cov_flag,
            zero_mean=zero_mean,
            max_it_var=max_it_var,
            verbose=verbose,
        )
        self.task_model = None

    def fit_sessions_with_selection(self, X_sessions, y_sessions, X_new, y_new):

        self.base_model.fit_prior(X_sessions, y_sessions)

        prior = self.base_model.prior
        mu = prior["mu"]
        selected_datasets = []
        selected_labels = []

        # Compute w_n using the current prior
        w_n, _ = self.base_model._fit_model(X_new, y_new, self.base_model.regularization, prior=prior)

        # Select weights that are close to w_n
        for i in range(X_sessions.shape[0]):
            w_s, _ = self.base_model._fit_model(
                X_sessions[i], y_sessions[i], self.base_model.regularization, prior=prior
            )
            if np.linalg.norm(w_s - w_n) <= np.linalg.norm(w_n - mu):
                selected_datasets.append(X_sessions[i])
                selected_labels.append(y_sessions[i])

        # At least 2 similar datasets are needed
        if len(selected_datasets) >= 2:
            X = np.empty(len(selected_datasets), dtype=object)
            y = np.empty(len(selected_labels), dtype=object)
            for i in range(len(selected_datasets)):
                X[i] = selected_datasets[i]
                y[i] = selected_labels[i]

            self.base_model.fit_prior(X, y)

        return self

    def __predict(self, X):
        if self.task_model:
            return self.task_model["predict"](X)
        else:
            return self.base_model.prior_predict(X)

    def score(self, X, y):
        y_p = self.__predict(X)
        return accuracy_score(y, y_p), y_p
