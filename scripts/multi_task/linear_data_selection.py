from typing import override
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
        self.base_model = MultiTaskLinearWithDataSelection(
            num_its=num_its,
            regularization=regularization,
            cov_flag=cov_flag,
            zero_mean=zero_mean,
            max_it_var=max_it_var,
            verbose=verbose,
        )
        self.task_model = None

    def fit_sessions(self, X_sessions, y_sessions):
        self.base_model.fit_prior(X_sessions, y_sessions)
        return self

    def fit(self, X, y):
        self.task_model = self.base_model.fit_new_task(X, y)
        return self

    def __predict(self, X):
        if self.task_model:
            return self.task_model["predict"](X)
        else:
            return self.base_model.prior_predict(X)

    def score(self, X, y):
        y_p = self.__predict(X)
        return accuracy_score(y, y_p), y_p


class MultiTaskLinearWithDataSelection(MultiTaskLinear):

    def __init__(
        self,
        num_its: int = 100,
        regularization: float = 0.5,
        cov_flag: str = "l2",
        zero_mean: bool = True,
        max_it_var: float = 0.0001,
        verbose: bool = True,
    ):
        self.num_its = num_its
        self.regularization = regularization
        self.cov_flag = cov_flag
        self.zero_mean = zero_mean
        self.max_it_var = max_it_var
        self.verbose = verbose
        self.prior = {}

    @override
    def fit_new_task(self, X: np.ndarray, y: np.ndarray) -> dict:
        if X.shape[0] != max(self.prior["mu"].shape):
            logger.error("Feature dimensionality of the data does not match this model")
            return {}

        prior = self.prior
        mu = prior["mu"]
        sigma = prior["sigma"]
        W = prior["W"]
        selected_weights = []

        # Compute w_n using the current prior
        w_n, _ = self._fit_model(X, y, self.regularization, prior=prior)

        # Select weights that are close to w_n
        for i in range(W.shape[1]):
            w_s = W[:, i : i + 1]
            if np.linalg.norm(w_s - w_n) <= np.linalg.norm(w_n - mu):
                selected_weights.append(w_s)

        if self.verbose:
            logger.info(f"Selected {len(selected_weights)} prior tasks out of {W.shape[1]} for new task adaptation.")

        # Update prior using selected w_s and w_n
        if selected_weights:
            selected_weights.append(w_n)
            new_W = np.concatenate(selected_weights, axis=1)
            prior = self._update_gaussian_prior(new_W)
            mu_diff = np.linalg.norm(prior["mu"] - mu)
            sigma_diff = np.linalg.norm(prior["sigma"] - sigma)

            if self.verbose:
                logger.info(f"Updated prior. mu new - mu old = {mu_diff:.4f}, sigma new - sigma old = {sigma_diff:.4f}")

        # Re-fit using updated prior
        w_n, loss = self._fit_model(X, y, self.regularization, prior=prior)

        def task_predict(X_test):
            raw_preds = np.sign(X_test.T @ w_n)
            raw_preds[raw_preds == 0] = 1
            return raw_preds

        return {
            "w": w_n,
            "loss": loss,
            "predict": task_predict,
            "train_acc": np.mean(y == task_predict(X)),
            "lambda": self.regularization,
        }
