from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in
        `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in
        `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = \
            None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector,
        same covariance
        matrix with dependent features.


        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        self.classes_, self.pi_ = np.unique(y, return_counts=True)
        self.pi_ = self.pi_ / n_samples
        n_classes = len(self.classes_)

        self.mu_ = np.array([X[y == k].mean(axis=0) for k in self.classes_])
        self.cov_ = np.zeros((n_features, n_features))
        tiled = np.array([self.mu_[val] for val in y])
        x_minus_mean = X - tiled
        for i in range(n_samples):
            sample = x_minus_mean[i, :]
            self.cov_ = self.cov_ + sample.reshape(-1, 1) @ sample.reshape(1,
                                                                           -1)
        self.cov_ = (1 / (n_samples - 1)) * self.cov_
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        likes = self.likelihood(X)
        return np.array([self.classes_[np.argmax(like)] for like in likes])

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        # todo: check if this is what excpected
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        ret = np.empty((n_samples, n_classes))
        for k in self.classes_:
            m_k = self.mu_[k]
            for i in range(n_samples):
                d = X.shape[1]
                x_minus_mean = X[i, :].reshape((1, 2)) - m_k
                z1 = x_minus_mean @ self._cov_inv @ x_minus_mean.transpose()
                z = np.exp((-1 / 2) * z1)
                ret[i, k] = self.pi_[k] * (1 / np.sqrt(
                    ((2 * np.pi) ** d) * det(self.cov_))) * z
        return ret

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
