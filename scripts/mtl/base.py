from abc import ABC, abstractmethod
from eeg_logger import logger
import numpy as np


class MultiTaskBase(ABC):
    """
    Abstract base class for multi-task learning models, particularly suited for EEG classification.
    This class handles shared prior updates and defines a framework for model fitting across tasks.
    Task are different sessions or subjects.
    """

    def __init__(self, num_its: int = 100, regularization: float = 0.5, cov_flag: str = "l2", zero_mean: bool = True):
        """
        Initialize the multi-task learning base class.

        Args:
            num_its (int): Number of prior update iterations.
            regularization (float): Regularization coefficient.
            cov_flag (str): Covariance estimation method ('l2' or 'l2-trace').
            zero_mean (bool): Whether to enforce a zero-mean prior.
        """
        self.num_its = num_its
        self.regularization = regularization
        self.cov_flag = cov_flag
        self.zero_mean = zero_mean
        self.prior = {"regularization": 0.5}

    def update_prior(self, weights: np.ndarray) -> None:
        """
        Update the prior distribution using the given model weights from all tasks.

        Args:
            weights (np.ndarray): A list or array of model weight vectors from each task.
        """
        W = np.concatenate(weights, axis=1)
        self.prior = self.update_gaussian_prior(W)

    def update_gaussian_prior(self, W: np.ndarray) -> dict:
        """
        Compute the Gaussian prior (mean and covariance) from the weights.

        Args:
            W (np.ndarray): Weight matrix where each column represents a task's model parameters.

        Returns:
            dict: Dictionary containing the prior mean 'mu', covariance 'sigma', and regularization parameter 'eta'.
        """
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

    @abstractmethod
    def fit_prior(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fit the shared prior by iteratively updating weights across all tasks.

        Args:
            X (np.ndarray): Feature matrices for all tasks.
            y (np.ndarray): Corresponding labels for all tasks.

        Returns:
            dict: Updated prior after convergence or reaching the iteration limit.
        """
        its = 0
        errors = np.empty((max(X.shape), 1))
        weights = np.empty_like(X)

        while its < self.num_its:
            prev_prior = self.prior

            for i in range(max(X.shape)):
                weights[i], errors[i] = self._fit_model(X[i], y[i], self.regularization)

            temp = self.prior["regularization"]
            self.update_prior(weights)
            self.prior["regularization"] = temp
            its += 1

            is_converged, num = self._is_convergence(self.prior, prev_prior)
            if is_converged:
                logger.info(f"Iteration {its} converged, error = {num}")
                break
            else:
                logger.info(f"Iteration {its}, error = {num}")

        return self.prior

    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: np.ndarray, regularization: float) -> tuple[np.ndarray, float]:
        """
        Fit a single-task model given the current prior.

        Args:
            X (np.ndarray): Input features for the task.
            y (np.ndarray): Labels for the task.
            regularization (float): Regularization parameter.

        Returns:
            tuple: Tuple of fitted model weights (np.ndarray) and fitting error (float).
        """
        pass

    @abstractmethod
    def fit_new_task(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fit a model for a new task using the current shared prior.

        Args:
            X (np.ndarray): Input features for the new task.
            y (np.ndarray): Labels for the new task.

        Returns:
            dict: Task-specific model parameters.
        """
        pass

    @abstractmethod
    def _is_convergence(self, current_prior: dict, prev_prior: dict) -> tuple[bool, float]:
        """
        Check whether the prior has converged.

        Args:
            current_prior (dict): The updated prior.
            prev_prior (dict): The prior from the previous iteration.

        Returns:
            tuple: Boolean indicating convergence and the associated error/difference metric.
        """
        pass

    @abstractmethod
    def prior_predict(self, X: np.ndarray):
        """
        Predict using the learned prior.

        Args:
            X (np.ndarray): Input feature data.

        Returns:
            Any: Output depending on the implementation (e.g., predicted labels, probabilities, etc.).
        """
        pass

    @staticmethod
    def swap_labels(y: np.ndarray, labels: np.ndarray, direction: str) -> np.ndarray:
        """
        Swap class labels for a given task based on a 2-row mapping.

        Args:
            y (np.ndarray): Label array from one task.
            labels (np.ndarray): 2x2 label mapping matrix. Row 0 is source labels; Row 1 is target labels.
            direction (str): Either 'to' or 'from'. 'to' swaps from row 0 to 1, 'from' swaps from row 1 to 0.

        Returns:
            np.ndarray: Transformed label array.
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
