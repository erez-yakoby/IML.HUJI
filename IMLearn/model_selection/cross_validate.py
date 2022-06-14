from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    samples_sets = np.array_split(X, cv)
    responses_sets = np.array_split(y, cv)
    train_errors = np.zeros(cv)
    test_errors = np.zeros(cv)
    for k in range(cv):
        cur_fold_samples = np.concatenate(np.delete(samples_sets, k, axis=0))
        cur_fold_responses = np.concatenate(
            np.delete(responses_sets, k, axis=0))
        remainder_samples = samples_sets[k]
        remainder_responses = responses_sets[k]
        h_k = estimator.fit(cur_fold_samples, cur_fold_responses)
        fold_prediction = h_k.predict(cur_fold_samples)
        remainder_prediction = h_k.predict(remainder_samples)
        train_errors[k] = scoring(cur_fold_responses, fold_prediction)
        test_errors[k] = scoring(remainder_responses, remainder_prediction)

    return np.mean(train_errors), np.mean(test_errors)
