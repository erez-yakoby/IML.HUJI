from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers.
    File is assumed to be an ndarray of shape (n_samples, 3) where the first
    2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
    linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss
    values (y-axis) as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        design_mat, responses = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        loss_callback = lambda fit, x, y: losses.append(
            fit.loss(design_mat, responses))
        Perceptron(callback=loss_callback).fit(design_mat, responses)

        # Plot figure of loss as function of fitting iteration
        fig = px.line(x=np.arange(len(losses)), y=np.array(losses), title=n)
        fig.update_layout(xaxis_title="Iteration number",
                          yaxis_title="Training loss value")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified
    covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and
    gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda_model = LDA().fit(X, y)
        lda_prediction = lda_model.predict(X)
        gnb_model = GaussianNaiveBayes().fit(X, y)
        gnb_prediction = gnb_model.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot
        # titles should specify algorithm and accuracy

        # Create subplots
        from IMLearn.metrics import accuracy
        lda_accuracy = '{:.3f}'.format(round(accuracy(y, lda_prediction), 3))
        gnb_accuracy = '{:.3f}'.format(round(accuracy(y, gnb_prediction), 3))
        symbols = np.array(["circle", "square", "diamond"])

        plot = make_subplots(1, 2, subplot_titles=(
            rf"$\text{{LDA model - Accuracy: {lda_accuracy}}}$",
            rf"$\text{{GNB model - Accuracy: {gnb_accuracy}}}$"))
        plot.add_traces(
            [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                        marker=dict(color=lda_prediction, symbol=symbols[y],
                                    line=dict(color="black", width=1),
                                    colorscale=[custom[0], custom[-1]]),
                        showlegend=False),
             go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                        marker=dict(color=gnb_prediction, symbol=symbols[y],
                                    line=dict(color="black", width=1),
                                    colorscale=[custom[0], custom[-1]]),
                        showlegend=False)],
            rows=[1, 1], cols=[1, 2])
        plot.update_layout(
            title=rf"$\textbf{{LDA and GBA models over {f} data set}}$",
            margin=dict(t=100))

        # Add `X` dots specifying fitted Gaussians' means
        plot.add_traces(
            [go.Scatter(x=lda_model.mu_[:, 0], y=lda_model.mu_[:, 1],
                        mode="markers", marker=dict(color="black", symbol="x",
                                                    line=dict(color="black",
                                                              width=1)),
                        showlegend=False),
             go.Scatter(x=gnb_model.mu_[:, 0], y=gnb_model.mu_[:, 1],
                        mode="markers", marker=dict(color="black", symbol="x",
                                                    line=dict(color="black",
                                                              width=1)),
                        showlegend=False)],
            rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        n_classes = lda_model.classes_.shape[0]
        plot.add_traces(
            [get_ellipse(lda_model.mu_[k, :], lda_model.cov_) for k in
             range(n_classes)], rows=1, cols=1)
        plot.add_traces(
            [get_ellipse(gnb_model.mu_[k, :],
                         np.diag(gnb_model.vars_[k, :])) for k in
             range(n_classes)], rows=1, cols=2)

        plot.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
