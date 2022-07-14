import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

# from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression as LG
from IMLearn.model_selection import cross_validate

import plotly.graph_objects as go


def plot_descent_path(module,
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange]
        over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1,
    np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's
    value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding
        the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def callback(val, weight, **kwargs):
        values.append(val)
        weights.append(weight)

    return callback, values, weights


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    for name, module in {"L1": L1, "L2": L2}.items():
        conv_fig = go.Figure().update_layout(
            title=f"Convergence Rate of {name} module",
            xaxis_title="Iteration number",
            yaxis_title="Norm value")
        losses = np.zeros(len(etas))
        for i, eta in enumerate(etas):
            mod = module(init)
            fixed_lr = FixedLR(eta)
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(fixed_lr, callback=callback)
            solution = gd.fit(mod, None, None)

            # intended for last weights only!
            losses[i] = values[-1]
            path = np.array(weights)
            fig = plot_descent_path(module, path,
                                    f": {name} module with fixed LR value: {eta}")
            # fig.show()
            conv_fig.add_trace(
                go.Scatter(x=np.arange(len(values)), y=np.array(values),
                           mode="markers", name=f"eta={eta}"))
        # conv_fig.show()

        lowest_loss_idx = int(np.argmin(losses))
        print(f"lowest score for {name} module is: ", losses[lowest_loss_idx],
              "with eta value: ", etas[lowest_loss_idx])


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the
    # exponentially decaying learning rate
    conv_fig = go.Figure().update_layout(
        title=f"Convergence rate for L1 module with {eta} Exponential learning rate",
        xaxis_title="Iteration number",
        yaxis_title="Norm value"
    )
    losses = np.zeros(len(gammas))
    for i, gamma in enumerate(gammas):
        l1_mod = L1(init)
        exp_lr = ExponentialLR(eta, gamma)
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(exp_lr, callback=callback)
        solution = gd.fit(l1_mod, None, None)
        losses[i] = np.linalg.norm(solution, ord=1)

        conv_fig.add_trace(go.Scatter(x=np.arange(len(values)),
                                      y=np.array(values),
                                      mode="markers", name=f"gamma={gamma}"))
        if gamma == 0.95:
            path = np.array(weights)
            fig_l1 = plot_descent_path(L1, path,
                                       f": L1 module with exp LR values: {eta}, {gamma}")

            l2_mod = L2(init)
            exp_lr = ExponentialLR(eta, gamma)
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(exp_lr, out_type="best", callback=callback)
            gd.fit(l2_mod, None, None)
            path = np.array(weights)
            fig_l2 = plot_descent_path(L2, path,
                                       f": L2 module with exp LR values: {eta}, {gamma}")

    # Plot algorithm's convergence for the different values of gamma

    # conv_fig.show()
    lowest_loss_idx = int(np.argmin(losses))
    print("The lowest loss achieved by L1 module was: ",
          losses[lowest_loss_idx],
          "with gamma value: ", gammas[lowest_loss_idx])

    # Plot descent path for gamma=0.95
    # fig_l1.show()
    # fig_l2.show()


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train-
    and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X :
    DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X :
    DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease
    # data
    alphas = np.linspace(0, 1, 101)
    logistic_model = LogisticRegression(
        solver=GradientDescent(learning_rate=FixedLR(1e-4),
                               max_iter=20000)).fit(X_train, y_train)

    probabilities = logistic_model.predict_proba(X_train)
    fpr, tpr = np.zeros(101), np.zeros(101)
    p = np.count_nonzero(y_train)
    n = X_train.shape[0] - p

    for i, alpha in enumerate(alphas):
        prediction = np.where(probabilities > alpha, 1, 0)
        tp = np.sum(np.logical_and(prediction == 1, y_train == 1))
        tpr[i] = np.sum(tp) / p
        fp = np.sum(np.logical_and(prediction == 1, y_train == 0))
        fpr[i] = np.sum(fp) / n

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="."),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=alphas,
                         name="", showlegend=False, marker_size=5,

                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=f"ROC curve for Logistic Model",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    # fig.show()

    # question 9
    opt_alpha = alphas[np.argmax(tpr - fpr)]

    print("best alpha is: ", opt_alpha)
    model_of_opt_alpha = LogisticRegression(alpha=opt_alpha,
                                            solver=GradientDescent(
                                                learning_rate=FixedLR(1e-4),
                                                max_iter=20000))
    model_of_opt_alpha.fit(X_train, y_train)
    opt_alpha_loss = model_of_opt_alpha.loss(X_test, y_test)
    print("its loss is: ", opt_alpha_loss)

    # Fitting l1- and l2-regularized logistic regression models, using
    # cross-validation to specify values
    # of regularization parameter

    lambdas = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    for name, module in {"l1": L1, "l2": L2}.items():
        train_scores, val_scores = np.zeros(len(lambdas)), np.zeros(
            len(lambdas))
        for i, lam in enumerate(lambdas):
            train_scores[i], val_scores[i] = cross_validate(
                LogisticRegression(
                    solver=GradientDescent(learning_rate=FixedLR(1e-4),
                                           max_iter=20000),
                    penalty=name, lam=lam, alpha=0.5),
                np.array(X_train), np.array(y_train),
                misclassification_error)

        best_lambda_idx = np.argmin(val_scores)
        best_lambda = lambdas[best_lambda_idx]
        print(f"Best lambda val for {name} module is: {best_lambda}")
        new_model = LogisticRegression(
            solver=GradientDescent(learning_rate=FixedLR(1e-4),
                                   max_iter=20000),
            penalty=name, lam=best_lambda, alpha=0.5)
        new_model.fit(X_train, y_train)
        loss = new_model.loss(X_test, y_test)
        print(f"Loss over test set using {name} module is: {loss}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
