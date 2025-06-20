from abc import ABC, abstractmethod
from eeg_logger import logger
import numpy as np


class MultiTaskBase(ABC):

    def __init__(self, num_its: int = 100, regularization: float = 0.5, cov_flag: str = "l2", zero_mean: bool = False):
        """
        :param int num_its: Number of iterations of the prior computation before exiting
        :param float regularization: regularization parameter
        :param str cov_flag: Flag that sets how the prior covariance is computed ['l2', 'l2-trace']
        :param bool zero_mean: If true, force the prior mean to be zero for a shrinkage effect
        """
        self.num_its = num_its
        self.regularization = regularization
        self.cov_flag = cov_flag
        self.zero_mean = zero_mean
        self.prior = {"regularization": 0.5}

    @abstractmethod
    def fit_prior(self, X: np.ndarray, y: np.ndarray) -> dict:
        prev_prior = self.prior
        its = 0
        errors = np.empty((max(X.shape), 1))
        weights = [None] * len(X)

        while its < self.num_its:

            for i in range(max(X.shape)):
                weights[i], errors[i] = self.fit_model(X[i], y[i], self.regularization)

            temp = self.prior["regularization"]
            self.update_prior(np.array(weights))
            self.prior["regularization"] = temp
            its += 1

            is_converged, num = self.is_convergence(self.prior, prev_prior)
            if is_converged:
                logger.info(f"[MT_prior] Iteration {its} converged, error = {num}")
                break
            else:
                logger.info(f"[MT_prior] Iteration {its}, error = {num}")

        return self.prior

    def update_prior(self, weights: np.ndarray) -> None:
        W = np.concatenate(weights, axis=1)
        self.prior = self.update_gaussian_prior(W)

    @abstractmethod
    def fit_model(self, X: np.ndarray, y: np.ndarray, regularization: float) -> tuple[np.ndarray, float]:
        pass

    @abstractmethod
    def is_convergence(self, current_prior: dict, prev_prior: dict) -> tuple[bool, float]:
        pass

    @staticmethod
    def swap_labels(y: np.ndarray, labels: np.ndarray, direction: str) -> np.ndarray:
        """
        :param np.ndarray y: labels from one task
        :param np.ndarray labels: 2x2 matrix. First row contains labels from task, second contains labels to swap to
        :param str direction: possible values: ['to', 'from']. 'to' swaps task labels to second matrix row. 'from' swaps values from second matrix row to first matrix row
        """
        match direction:
            case "to":
                ind = 0
            case "from":
                ind = 1
        temp = np.zeros_like(y, dtype=np.int16)
        for i in range(2):
            match_val = labels[i, ind]
            replace_val = labels[i, 1 - ind]
            temp[y == match_val] = replace_val
        return temp

    def update_gaussian_prior(self, W: np.ndarray) -> dict:
        new_prior = {}
        new_prior["W"] = W
        new_prior["mu"] = np.mean(W, axis=1, keepdims=True)

        temp = W - np.tile(new_prior["mu"], (1, W.shape[1]))
        eta = np.linalg.eigvals((1 / (temp.shape[1] - 1)) * (temp @ temp.conj().T))
        eta = 1 if not np.any(eta > 0) else np.abs(np.min(eta[eta > 0]))

        match self.cov_flag:
            case "l2":  # Standard ML covariance update
                C = (1 / (temp.shape[1] - 1)) * (temp @ temp.conj().T)
            case "l2-trace":  # Trace-normalized update
                C = (1 / np.trace(temp @ temp.conj().T)) * (temp @ temp.conj().T)

        if np.linalg.matrix_rank(C) < C.shape[0]:
            C = C + eta * np.eye(C.shape[0])

        new_prior["sigma"] = C

        if self.zero_mean:
            new_prior["mu"] = np.zeros((W.shape[0], 1))

        return new_prior
