import numpy as np
from eeg_logger import logger


class MultiTaskLinear:

    def __init__(
        self,
        num_its: int = 100,
        regularization: float = 0.5,
        cov_flag: str = "l2",
        zero_mean: bool = True,
        use_pca: bool = False,
        max_it_var: float = 0.0001,
    ):
        self.num_its = num_its
        self.regularization = regularization
        self.cov_flag = cov_flag
        self.zero_mean = zero_mean
        self.prior = {"regularization": 0.5}
        self.use_pca = use_pca
        self.max_it_var = max_it_var
        self.label_mapping = {}

    def fit_prior(self, X: np.ndarray, y: np.ndarray):
        try:
            self.__validate_dataset(X, y)
        except ValueError as e:
            logger.error(f"Error occurred while validating dataset - {e} {X.shape} {y.shape}")
            return

        self.label_mapping = {}
        for i in range(len(y)):
            y[i], mapping = self.__encode_labels(y[i])
            self.label_mapping[i] = mapping

        self.prior["mu"] = np.ones((X[0].shape[0], 1))
        self.prior["sigma"] = np.eye(X[0].shape[0])
        self.prior["W"] = np.zeros((X[0].shape[0], max(X.shape)))

        its = 0
        errors = np.empty((max(X.shape), 1))
        weights = np.empty_like(X)

        while its < self.num_its:
            prev_prior = self.prior

            for i in range(max(X.shape)):
                weights[i], errors[i] = self._fit_model(X[i], y[i], self.regularization)

            temp = self.prior["regularization"]
            self.__update_prior(weights)
            self.prior["regularization"] = temp
            its += 1

            is_converged, num = self.__is_convergence(self.prior, prev_prior)
            if is_converged:
                logger.info(f"Iteration {its} converged, error = {num}")
                break
            else:
                logger.info(f"Iteration {its}, error = {num}")

        return self.prior

    def __update_prior(self, weights: np.ndarray) -> None:
        W = np.concatenate(weights, axis=1)
        self.prior = self.__update_gaussian_prior(W)

    def __update_gaussian_prior(self, W: np.ndarray) -> dict:
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

    def __validate_dataset(self, X: np.ndarray, y: np.ndarray):
        if max(X.shape) != max(y.shape):
            raise ValueError("Unequal data and label arrays")
        if max(X.shape) <= 1:
            raise ValueError("Only one dataset provided - cancelling")
        for i in range(max(X.shape)):
            if X[i].shape[1] != max(y[i].shape):
                raise ValueError("Number of datapoints and labels differ")

    def _fit_model(self, X: np.ndarray, y: np.ndarray, regularization: float) -> tuple[np.ndarray, float]:
        Ax = self.prior["sigma"] @ X
        A = (1 / regularization) * (Ax @ X.T) + np.eye(X.shape[0])
        b = (1 / regularization) * (Ax @ y) + self.prior["mu"]
        w = np.linalg.solve(A, b)
        loss = self.loss(w, X, y)
        return w, loss

    def fit_new_task(self, X: np.ndarray, y: np.ndarray) -> dict:
        if X.shape[0] != max(self.prior["mu"].shape):
            logger.error("Feature dimensionality of the data does not match this model")
            return {}

        out = {"lambda": self.regularization}
        X_original = X
        y_internal, mapping = self.__encode_labels(y)
        out["w"], out["loss"] = self._fit_model(X, y_internal, out["lambda"])

        def task_predict(X_test):
            raw_preds = np.sign(X_test.T @ out["w"])
            return self.__decode_labels(raw_preds, mapping)

        out["predict"] = task_predict
        out["train_acc"] = np.mean(y == out["predict"](X_original))
        return out

    def __is_convergence(self, current_prior: dict, prev_prior: dict) -> tuple[bool, float]:
        W = current_prior["W"]
        W_prev = prev_prior["W"]
        norm = np.linalg.norm(W - W_prev, ord="fro")
        converged = norm < self.max_it_var * np.mean(W)
        return converged, norm

    def prior_predict(self, X: np.ndarray):

        mean_weights = np.mean(self.prior["W"], axis=1, keepdims=True)
        raw_preds = np.sign(X.T @ mean_weights)

        if self.label_mapping:
            mapping = list(self.label_mapping.values())[0]
            return self.__decode_labels(raw_preds, mapping)
        else:
            return raw_preds

    def loss(self, w, X, y) -> float:
        """
        Compute average squared loss for linear predictions.
        """
        residuals = X.T @ w - y
        L = np.linalg.norm(residuals, 2) ** 2 / len(y)
        return L

    def __encode_labels(self, y):
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("Only binary classification supported.")
        mapping = {classes[0]: -1, classes[1]: 1}
        return np.vectorize(mapping.get)(y), mapping

    def __decode_labels(self, y_internal, mapping):
        inv_mapping = {-1: list(mapping.keys())[0], 1: list(mapping.keys())[1]}
        return np.vectorize(inv_mapping.get)(y_internal)
