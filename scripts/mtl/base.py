from abc import ABC, abstractmethod
import numpy as np


class MultiTaskBase(ABC):

    def __init__(self, num_its: int = 1000, regularization: float = 0.5, cov_flag: str = "l2", zero_mean: bool = False):
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
    def fit_prior(self, X: np.ndarray, y: np.ndarray):
        its = 0
        errors = np.empty((max(X.shape), 1))
        weights = np.empty((max(X.shape), 1))

        while its < self.num_its:

            for i in range(max(X.shape)):
                weights[i], errors[i] = self.fit_model(X[i], y[i], self.regularization)

            temp = self.prior["regularization"]

    @abstractmethod
    def fit_model(self, X: np.ndarray, y: np.ndarray, regularization: float) -> tuple[np.ndarray, float]:
        pass

    @abstractmethod
    def update_prior(self, weights: np.ndarray) -> None:
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
        temp = np.zeros_like(y)
        for i in range(2):
            match_val = labels[i, ind]
            replace_val = labels[i, 1 - ind]
            temp[y == match_val] = replace_val
        return temp
