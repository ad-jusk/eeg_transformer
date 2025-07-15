import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.linalg import sqrtm
from sklearn.metrics import accuracy_score
from eeg_logger import logger


class MultiTaskLinearClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn-compatible classifier designed to handle learning from multiple tasks and apply transfer learning to a new one.
    It wraps around a base model and provides methods to:
    - Fit a prior model using multiple sessions (fit_sessions)
    - Adapt the prior to a new task (fit)
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


class MultiTaskLinear:
    """
    This class implements a Bayesian multi-task learning framework.
    It learns a prior distribution (mean and covariance of weights) from multiple related tasks (sessions).
    It can adapt this prior to a new task using Bayesian linear regression.
    Prediction is done using either:
    - A task-specific model (fit_new_task)
    - A prior mean model (prior_predict)
    The training involves estimating a Gaussian prior over weight vectors (mu, sigma) by solving regularized linear systems for each task,
    then updating the prior iteratively until convergence
    """

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

    def fit_prior(self, X: np.ndarray, y: np.ndarray):
        try:
            self._validate_dataset(X, y)
        except ValueError as e:
            logger.error(f"Error occurred while validating dataset - {e}")
            return

        self._init_prior(X)

        its = 0
        errors = np.empty((max(X.shape), 1))
        weights = np.empty_like(X)

        while its < self.num_its:
            prev_prior = self.prior

            for i in range(max(X.shape)):
                weights[i], errors[i] = self._fit_model(X[i], y[i], self.regularization, prior=self.prior)

            temp = self.prior["regularization"]
            self._update_prior(weights)
            self.prior["regularization"] = temp
            its += 1

            is_converged, num = self._is_convergence(self.prior, prev_prior)
            if is_converged:
                if self.verbose:
                    logger.info(f"Iteration {its} converged, error = {num}")
                break
            elif self.verbose:
                logger.info(f"Iteration {its}, error = {num}")

        return self.prior

    def fit_new_task(self, X: np.ndarray, y: np.ndarray) -> dict:
        if X.shape[0] != max(self.prior["mu"].shape):
            logger.error("Feature dimensionality of the data does not match this model")
            return {}

        out = {"lambda": self.regularization}
        X_original = X
        out["w"], out["loss"] = self._fit_model(X, y, out["lambda"], prior=self.prior)

        def task_predict(X_test):
            raw_preds = np.sign(X_test.T @ out["w"])
            raw_preds[raw_preds == 0] = 1
            return raw_preds

        out["predict"] = task_predict
        out["train_acc"] = np.mean(y == out["predict"](X_original))
        return out

    def prior_predict(self, X: np.ndarray):
        mean_weights = np.mean(self.prior["W"], axis=1, keepdims=True)
        raw_preds = np.sign(X.T @ mean_weights)
        raw_preds[raw_preds == 0] = 1
        return raw_preds

    def _init_prior(self, X: np.ndarray):
        self.prior["regularization"] = self.regularization
        self.prior["mu"] = np.ones((X[0].shape[0], 1))
        self.prior["sigma"] = np.eye(X[0].shape[0])
        self.prior["W"] = np.zeros((X[0].shape[0], max(X.shape)))

    def _update_prior(self, weights: np.ndarray) -> None:
        W = np.concatenate(weights, axis=1)
        self.prior = self._update_gaussian_prior(W)

    def _update_gaussian_prior(self, W: np.ndarray) -> dict:
        new_prior = {}
        new_prior["W"] = W
        new_prior["mu"] = np.mean(W, axis=1, keepdims=True)
        temp = W - new_prior["mu"]

        raw_cov = (1 / (temp.shape[1] - 1)) * (temp @ temp.T)
        e = np.linalg.eigvalsh(raw_cov)

        if not np.any(e > 0):
            eta = 1
        else:
            eta = np.abs(np.min(e[e > 0]))

        match self.cov_flag:
            case "l1":
                eta = 1e-4
                D = sqrtm(temp @ temp.T + np.eye(temp.shape[0]) * eta)
                D = np.real_if_close(D, tol=1e10)
                C = D / np.trace(D)
            case "l2":
                C = raw_cov
            case "l2-trace":
                C = (1 / np.trace(temp @ temp.T)) * (temp @ temp.T)
            case _:
                raise ValueError("Invalid covariance estimation flag")

        if np.linalg.matrix_rank(C) < C.shape[0]:
            C = C + eta * np.eye(C.shape[0])

        new_prior["sigma"] = C
        new_prior["eta"] = eta

        if self.zero_mean:
            new_prior["mu"] = np.zeros((W.shape[0], 1))

        return new_prior

    def _validate_dataset(self, X: np.ndarray, y: np.ndarray):
        if max(X.shape) != max(y.shape):
            raise ValueError(f"Unequal data and label arrays: data - {X.shape}, labels - {y.shape}")
        if max(X.shape) <= 1:
            raise ValueError("Only one dataset provided - cancelling")
        for i in range(max(X.shape)):
            if X[i].shape[1] != max(y[i].shape):
                raise ValueError(
                    f"Number of datapoints and labels differ: : data - {X[i].shape}, labels - {y[i].shape}"
                )

    def _fit_model(self, X: np.ndarray, y: np.ndarray, regularization: float, prior=None) -> tuple[np.ndarray, float]:
        Ax = prior["sigma"] @ X
        A = (1 / regularization) * (Ax @ X.T) + np.eye(X.shape[0])
        b = (1 / regularization) * (Ax @ y) + prior["mu"]
        w = np.linalg.solve(A, b)
        loss = self._loss(w, X, y)
        return w, loss

    def _is_convergence(self, current_prior: dict, prev_prior: dict) -> tuple[bool, float]:
        W = current_prior["W"]
        W_prev = prev_prior["W"]
        norm = np.linalg.norm(W - W_prev, ord="fro")
        converged = norm < self.max_it_var * np.mean(W)
        return converged, norm

    def _loss(self, w, X, y) -> float:
        """
        Compute average squared loss for linear predictions.
        """
        residuals = X.T @ w - y
        L = np.linalg.norm(residuals, 2) ** 2 / len(y)
        return L
