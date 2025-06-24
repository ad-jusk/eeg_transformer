import numpy as np
from typing import override
from scripts.mtl.base import MultiTaskBase
from eeg_logger import logger


class MultiTaskLinear(MultiTaskBase):
    """
    Concrete implementation of MultiTaskBase for linear models.
    Designed for binary classification tasks (e.g., EEG classification) using a shared Gaussian prior.

    Supports iterative training across multiple tasks and models each task using linear regression
    with regularization and shared prior parameters.

    Attributes:
        use_pca (bool): Whether to apply PCA preprocessing (currently not implemented).
        max_it_var (float): Convergence threshold for change in model weights.
        labels (list): Internal mapping of class labels across tasks.
        class_ids (list): Expected class IDs, assumed to be [1, -1].
    """

    def __init__(
        self,
        num_its: int = 100,
        regularization: float = 0.5,
        cov_flag: str = "l2",
        zero_mean: bool = True,
        use_pca: bool = False,
        max_it_var: float = 0.0001,
    ):
        """
        Initialize the linear multi-task learner with configuration options.

        Args:
            num_its (int): Maximum number of iterations for prior updates.
            regularization (float): L2 regularization strength.
            cov_flag (str): Covariance computation strategy ('l2' or 'l2-trace').
            zero_mean (bool): If True, enforce zero-mean prior.
            use_pca (bool): If True, use PCA (not yet supported).
            max_it_var (float): Threshold for convergence based on weight matrix variance.
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

        self.prior["mu"] = np.ones((X[0].shape[0], 1))
        self.prior["sigma"] = np.eye(X[0].shape[0])
        self.prior["W"] = np.zeros((X[0].shape[0], max(X.shape)))
        return super().fit_prior(X, y)

    def __validate_dataset(self, X: np.ndarray, y: np.ndarray):
        """
        Validate input shapes and consistency for multi-task data.

        Raises:
            ValueError: If any of the dataset consistency checks fail.
        """
        if max(X.shape) != max(y.shape):
            raise ValueError("Unequal data and label arrays")
        if max(X.shape) <= 1:
            raise ValueError("Only one dataset provided - cancelling")
        for i in range(max(X.shape)):
            if X[i].shape[1] != max(y[i].shape):
                raise ValueError("Number of datapoints and labels differ")

    @override
    def _fit_model(self, X: np.ndarray, y: np.ndarray, regularization: float) -> tuple[np.ndarray, float]:
        Ax = self.prior["sigma"] @ X
        A = (1 / regularization) * (Ax @ X.T) + np.eye(X.shape[0])
        b = (1 / regularization) * (Ax @ y) + self.prior["mu"]
        w = np.linalg.solve(A, b)
        loss = self.loss(w, X, y)
        return w, loss

    @override
    def fit_new_task(self, X: np.ndarray, y: np.ndarray) -> dict:
        if X.shape[0] != max(self.prior["mu"].shape):
            logger.error("Feature dimensionality of the data does not match this model")
            return {}

        out = {"lambda": self.regularization}
        X_original = X
        y_train = MultiTaskBase.swap_labels(y, self.labels, "to")
        out["w"], out["loss"] = self._fit_model(X, y_train, out["lambda"])
        out["predict"] = lambda X: self.predict(out["w"], X, self.labels)
        out["train_acc"] = np.mean(y == out["predict"](X_original))
        return out

    @override
    def _is_convergence(self, current_prior: dict, prev_prior: dict) -> tuple[bool, float]:
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

    def loss(self, w, X, y) -> float:
        """
        Compute average squared loss for linear predictions.

        Implements:
            L = ||X.T @ w - y||^2 / len(y)

        Args:
            w (np.ndarray): Weight vector.
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Ground truth labels.

        Returns:
            float: Mean squared error.
        """
        residuals = X.T @ w - y
        L = np.linalg.norm(residuals, 2) ** 2 / len(y)
        return L
