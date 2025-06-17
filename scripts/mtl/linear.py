import numpy as np
from typing import override
from scripts.mtl.base import MultiTaskBase
from eeg_logger import logger


class MultiTaskLinear(MultiTaskBase):

    def __init__(self, num_its=1000, regularization=0.5, cov_flag="l2", zero_mean=False, use_pca: bool = False):
        """
        :param bool use_pca: Use PCA for cross-subject dimensionality reduction
        """
        super().__init__(num_its, regularization, cov_flag, zero_mean)
        self.use_pca = use_pca
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

    def __validate_dataset(self, X: np.ndarray, y: np.ndarray):
        if max(X.shape) != max(y.shape):
            raise ValueError("Unequal data and label arrays")
        if max(X.shape) <= 1:
            raise ValueError("Only one dataset provided - cancelling")
        for i in range(max(X.shape)):
            if X[1].shape[1] != max(y[i].shape):
                raise ValueError("Number of datapoints and labels differ")

    @override
    def fit_model(self, X: np.ndarray, y: np.ndarray, regularization: float) -> tuple[np.ndarray, float]:
        pass
