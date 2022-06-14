from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MIN_VAL = (-1.2)
MAX_VAL = 2


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select
    the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.linspace(MIN_VAL, MAX_VAL, n_samples)
    np.random.shuffle(X)
    epsilon = np.random.normal(0, noise, n_samples)
    true_y = f(X)
    y = true_y + epsilon

    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X),
                                                        pd.Series(y), 2 / 3)

    fig = go.Figure().add_traces([
        go.Scatter(x=X, y=true_y, mode="markers", name="Noiseless data"),
        go.Scatter(x=train_x[0], y=train_y, mode="markers", name="Train data"),
        go.Scatter(x=test_x[0], y=test_y, mode="markers", name="Test data")])
    fig.update_layout(
        title="Noiseless Data, Train and Test Data with Gaussian(0, "+ str(noise)+ ") noise",
        xaxis_title="x", yaxis_title="y")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_deg = 10
    k_vals = np.arange(max_deg + 1)
    train_scores = np.zeros(max_deg + 1)
    val_scores = np.zeros(max_deg + 1)
    for k in k_vals:
        train_scores[k], val_scores[k] = cross_validate(PolynomialFitting(k),
                                                        np.array(train_x),
                                                        np.array(train_y),
                                                        mean_square_error)
    fig = go.Figure().add_traces([
        go.Scatter(x=k_vals, y=train_scores, name="Train error"),
        go.Scatter(x=k_vals, y=val_scores, name="Validation error")])
    fig.update_layout(
        title="Train and Validation errors for 5-fold cross validation, "
              "based on polynomial degree",
        xaxis_title="Polynomial degree",
        yaxis_title="Error")
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and
    # report test error
    argmin_idx = int(np.argmin(val_scores))
    model = PolynomialFitting(argmin_idx).fit(np.array(train_x),
                                              np.array(train_y))
    error = mean_square_error(np.array(test_y),
                              model.predict(np.array(test_x)))
    print("degree: ", argmin_idx, ", mse val: ", round(error, 2))


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best
    fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the
        algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing
    # portions
    samples, responses = datasets.load_diabetes(return_X_y=True)
    train_x, train_y = samples[:n_samples], responses[:n_samples]
    test_x, test_y = samples[n_samples:], responses[n_samples:]

    # Question 7 - Perform CV for different values of the regularization
    # parameter for Ridge and Lasso regressions

    range_lim = {"Ridge": 0.04, "Lasso": 2}
    best_params = {}
    for name, model in {"Ridge": RidgeRegression, "Lasso": Lasso}.items():
        model_range = np.linspace(0.00001, range_lim[name], n_evaluations)
        model_train_score = np.zeros(n_evaluations)
        model_val_score = np.zeros(n_evaluations)
        for i, k in enumerate(model_range):
            model_train_score[i], model_val_score[i] = cross_validate(
                model(k),
                np.array(train_x), np.array(train_y),
                mean_square_error)

        fig = go.Figure().add_traces([
            go.Scatter(x=model_range, y=model_train_score, name="Train error"),
            go.Scatter(x=model_range, y=model_val_score,
                       name="Validation error")])
        fig.update_layout(
            title=name + "Regression: Train and Validation Error, based on "
                         "Regularization Parameter Value",
            xaxis_title="Regularization Parameter",
            yaxis_title="MSE")
        fig.show()

        best_params[name] = model_range[int(np.argmin(model_val_score))]
        print("best param for ", name, " model is: ", best_params[name])

    # Question 8 - Compare best Ridge model, best Lasso model and Least
    # Squares model

    models = {"Ridge": RidgeRegression(best_params["Ridge"]),
              "Lasso": Lasso(best_params["Lasso"]),
              "Least Squares": LinearRegression()}
    for name, model in models.items():
        error = mean_square_error(test_y,
                                  model.fit(train_x, train_y).predict(test_x))
        print("The validation error for ", name, " model is: ", error)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter(n_evaluations=500)
