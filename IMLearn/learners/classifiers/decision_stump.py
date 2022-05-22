from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics.loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m, d = X.shape

        thr_values = np.empty(d)
        thr_errors = np.empty(d)
        thr_signs = np.empty(d)

        for feature in range(d):
            feature_vec = X[:, feature]
            thr_pos, thr_err_pos = self._find_threshold(feature_vec, y, 1)
            thr_neg, thr_err_neg = self._find_threshold(feature_vec, y, -1)
            is_pos_better = thr_err_pos <= thr_err_neg
            thr_values[feature] = thr_pos if is_pos_better else thr_neg
            thr_errors[feature] = thr_err_pos if is_pos_better else thr_err_neg
            thr_signs[feature] = 1 if is_pos_better else -1

        thr_idx = np.argmin(thr_errors)
        self.threshold_, self.j_, self.sign_ = thr_values[thr_idx], thr_idx, \
                                               thr_signs[thr_idx]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        p = np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)
        return p

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # todo: check if need to check errors for majority - forum question
        # todo: check why itertools is imported

        losses = np.empty(values.size)
        for i, threshold in enumerate(values):
            predicted = np.where(values >= threshold, sign, -sign)
            where_diff = predicted != np.sign(labels)
            D = np.abs(labels)
            mult = D * where_diff
            losses[i] = np.sum(mult)
        thr_idx = np.argmin(losses)
        thr = values[thr_idx]
        thr_err = losses[thr_idx]
        return thr, thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))


