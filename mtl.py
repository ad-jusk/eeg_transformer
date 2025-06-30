import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne.decoding import CSP, PSDEstimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import LeftRightImagery

from scripts.mtl.linear import MultiTaskLinear

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


class MultiTaskLinearClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        num_its=100,
        regularization=0.5,
        cov_flag="l2",
        zero_mean=True,
        use_pca=False,
        max_it_var=0.0001,
    ):
        self.num_its = num_its
        self.regularization = regularization
        self.cov_flag = cov_flag
        self.zero_mean = zero_mean
        self.use_pca = use_pca
        self.max_it_var = max_it_var
        self.base_model = MultiTaskLinear(
            num_its=num_its,
            regularization=regularization,
            cov_flag=cov_flag,
            zero_mean=zero_mean,
            use_pca=use_pca,
            max_it_var=max_it_var,
        )
        self.task_model = None

    def fit_sessions(self, X_sessions, y_sessions):
        """
        This method fits the prior using multiple sessions.
        X_sessions: ndarray of shape (n_sessions,) where each element is (n_features, n_epochs)
        y_sessions: list or array of shape (n_sessions,) where each element is (n_epochs,)
        """
        y_sessions = np.array(y_sessions, dtype=object)
        self.base_model.fit_prior(X_sessions, y_sessions)
        return self

    def fit(self, X, y):
        """
        Standard scikit-learn `fit`, used to train on a single task after prior.
        Expects X shape: (n_samples, n_features) => Transposed internally to (n_features, n_samples)
        """
        X = X.T  # Convert to (n_features, n_samples)
        y = np.array(y).reshape(-1, 1)
        self.task_model = self.base_model.fit_new_task(X, y)
        return self

    def predict(self, X):
        X = X.T  # Convert to (n_features, n_samples)
        if self.task_model:
            return self.task_model["predict"](X).flatten()
        else:
            return self.base_model.prior_predict(X).flatten()

    def score(self, X, y):
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X))


dataset = BNCI2014_001()
dataset.subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
paradigm = LeftRightImagery()
X_prior, y_prior, metadata = paradigm.get_data(dataset, subjects=dataset.subject_list)
sessions = metadata.session.unique()
X_sessions = []
y_sessions = []

for session in sessions:
    session_mask = metadata.session == session
    X_sess = X_prior[session_mask].T  # shape (n_features, n_epochs)
    y_sess = y_prior[session_mask].reshape(-1, 1)
    X_sessions.append(X_sess)
    y_sessions.append(y_sess)

clf = MultiTaskLinearClassifier()
clf.fit_sessions(np.array(X_sessions, dtype=object), np.array(y_sessions, dtype=object))
eval = CrossSessionEvaluation(paradigm=paradigm, datasets=[dataset], overwrite=True)
results = eval.process(pipelines={"mtl": clf})
