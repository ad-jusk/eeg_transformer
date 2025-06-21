import numpy as np
from typing import override
from scripts.mtl.base import MultiTaskBase
from eeg_logger import logger


class MultiTaskLinear(MultiTaskBase):

    def __init__(
        self,
        num_its=100,
        regularization=0.5,
        cov_flag="l2",
        zero_mean=True,
        use_pca: bool = False,
        max_it_var: float = 0.0001,
    ):
        """
        :param int num_its: Number of iterations of the prior computation before exiting
        :param float regularization: regularization parameter
        :param str cov_flag: Flag that sets how the prior covariance is computed ['l2', 'l2-trace']
        :param bool zero_mean: If true, force the prior mean to be zero for a shrinkage effect
        :param bool use_pca: Use PCA for cross-subject dimensionality reduction
        :param float max_it_var: Maximum variation between iterations before convergence
        """
        super().__init__(num_its, regularization, cov_flag, zero_mean)
        self.use_pca = use_pca
        self.max_it_var = max_it_var
        self.labels = []
        self.class_ids = [1, -1]

    @override
    def fit_prior(self, X: np.ndarray, y: np.ndarray):
        try:
            self.__validate_dataset(X, y)
        except ValueError as e:
            logger.error(f"Error occurred while validating dataset - {e}")
            return
        self.labels = np.array([np.unique(np.concatenate(y)), self.class_ids])
        for i in range(max(y.shape)):
            y[i] = MultiTaskBase.swap_labels(y[i], self.labels, direction="to")

        if self.use_pca:
            pass
        else:
            self.prior["mu"] = np.ones((X[0].shape[0], 1))
            self.prior["sigma"] = np.eye(X[0].shape[0])
            self.prior["W"] = np.zeros((X[0].shape[0], max(X.shape)))
            return super().fit_prior(X, y)

    def __validate_dataset(self, X: np.ndarray, y: np.ndarray):
        if max(X.shape) != max(y.shape):
            raise ValueError("Unequal data and label arrays")
        if max(X.shape) <= 1:
            raise ValueError("Only one dataset provided - cancelling")
        for i in range(max(X.shape)):
            if X[i].shape[1] != max(y[i].shape):
                raise ValueError("Number of datapoints and labels differ")

    @override
    def fit_model(self, X: np.ndarray, y: np.ndarray, regularization: float) -> tuple[np.ndarray, float]:
        Ax = self.prior["sigma"] @ X
        A = (1 / regularization) * (Ax @ X.conj().T) + np.eye(X.shape[0])
        b = (1 / regularization) * (Ax @ y) + self.prior["mu"]
        w = np.linalg.solve(A, b)
        loss = self.loss(w, X, y)
        return w, loss

    @override
    def is_convergence(self, current_prior: dict, prev_prior: dict) -> tuple[bool, float]:
        W = current_prior["W"]
        W_prev = prev_prior["W"]
        norm = np.linalg.norm(W - W_prev, ord="fro")
        converged = norm < self.max_it_var * np.mean(W)
        return converged, norm

    @override
    def prior_predict(self, X: np.ndarray):
        if len(self.labels) == 0:
            logger.error("Cannot predict - model has not been trained")

        mean_weights = np.mean(self.prior["W"], axis=1, keepdims=True)
        return self.predict(mean_weights, X, self.labels)

    def predict(self, w: np.ndarray, X: np.ndarray, labels: np.ndarray):
        return MultiTaskBase.swap_labels(np.sign(X.T @ w), labels, "from")

    def loss(self, w, X, y):
        """
        Implements average squared loss: L = ||X.T @ w - y||^2 / len(y)
        """
        residuals = X.T @ w - y
        L = np.linalg.norm(residuals, 2) ** 2 / len(y)
        return L
