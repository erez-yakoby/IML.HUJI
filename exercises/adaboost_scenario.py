import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaboost.fit(train_X, train_y)

    learners_range = np.arange(1, n_learners + 1)
    train_err, test_err = np.empty(n_learners), np.empty(n_learners)
    for t in learners_range:
        train_err[t - 1] = adaboost.partial_loss(train_X, train_y, t)
        test_err[t - 1] = adaboost.partial_loss(test_X, test_y, t)

    fig = go.Figure().add_traces(
        [go.Scatter(x=learners_range, y=train_err, name="Train samples"),
         go.Scatter(x=learners_range, y=test_err, name="Test samples")])
    fig.update_layout(
        title="Adaboost loss according to amount of learners",
        xaxis_title="Number of learners", yaxis_title="Loss")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    symbols = np.array(["circle", "x"])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$\textbf{{Learners amount: {m}}}$"
                                        for m in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(
            lambda x: adaboost.partial_predict(x, T=t), lims[0], lims[1],
            showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                       showlegend=False,
                       marker=dict(color=test_y.astype(int), symbol="circle",
                                   colorscale=[custom[0], custom[-1]],
                                   line=dict(color="black", width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(
        title="Decision Boundaries according to learners amount",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_ensemble_idx = np.argmin(test_err)
    best_ensemble_size = best_ensemble_idx + 1
    accuracy = 1 - test_err[best_ensemble_idx]

    fig = go.Figure().add_traces(
        [decision_surface(
            lambda x: adaboost.partial_predict(x, T=best_ensemble_size),
            lims[0], lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                       showlegend=True, name="Test samples",
                       marker=dict(color=test_y.astype(int),
                                   symbol="circle",
                                   colorscale=[custom[0], custom[-1]],
                                   line=dict(color="black", width=1)))])
    title = "Decision boundary of the best ensemble.\n" + \
            "Ensemble size:{}, Accuracy: {}"
    fig.update_layout(title=title.format(best_ensemble_size, accuracy))
    fig.show()

    # Question 4: Decision surface with weighted samples
    s = 5 * adaboost.D_ / np.max(adaboost.D_)
    fig = go.Figure().add_traces(
        [decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                    showlegend=True, name="Train samples",
                    marker=dict(color=train_y.astype(int), size=s,
                                symbol=symbols[train_y.astype(int)],
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))])
    title = "Decision boundary of the last iteration with markers size " \
            "proportional to weight"
    fig.update_layout(title=title)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)




