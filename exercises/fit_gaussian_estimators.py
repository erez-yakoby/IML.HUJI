import pandas as pd

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import plotly.express as px

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    samples = np.random.normal(10, 1, 1000)
    uni_estimator = UnivariateGaussian().fit(samples)
    print("({mu}, {var})".format(mu=uni_estimator.mu_, var=uni_estimator.var_))

    # Question 2 - Empirically showing sample mean is consistent

    sample_sizes = np.arange(1, (1000 / 10) + 1) * 10

    # array containing absolut distances between estimated and real expectation
    differences = []
    for size in sample_sizes:
        expectation_val = UnivariateGaussian().fit(samples[:int(size)]).mu_
        differences.append(abs(expectation_val - 10))

    fig = go.Figure().add_trace(
        go.Scatter(x=sample_sizes, y=differences, mode="lines")).update_layout(
        title_text="Distance between estimated and real expectation")
    fig.update_xaxes(title_text="Sample size")
    fig.update_yaxes(title_text="Expectation difference")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model

    x_axis_name_2, y_axis_name_2 = "Sample value", "PDF value"
    graph_title_2 = "Empirical PDF function"

    sorted_samples = np.sort(samples)
    pdf_values = uni_estimator.pdf(sorted_samples)
    pdf_df = pd.DataFrame(np.array([sorted_samples, pdf_values]).transpose(),
                          columns=[x_axis_name_2, y_axis_name_2])
    px.scatter(pdf_df, x=x_axis_name_2, y=y_axis_name_2,
               title=graph_title_2).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov_matrix = np.array([[1, 0.2, 0, 0.5],
                           [0.2, 2, 0, 0],
                           [0, 0, 1, 0],
                           [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mean, cov_matrix, 1000)
    multy_estimator = MultivariateGaussian().fit(samples)
    print(multy_estimator.mu_, multy_estimator.cov_, sep="\n")

    # Question 5 - Likelihood evaluation

    lin_space = np.linspace(-10, 10, 200)
    f1_vector = np.repeat(lin_space, 200)
    f3_vector = np.tile(lin_space, 200)
    mean_matrix = np.zeros((40000, 4))
    mean_matrix[:, 0] = f1_vector
    mean_matrix[:, 2] = f3_vector

    max_val = MultivariateGaussian.log_likelihood(
        mean_matrix[0], cov_matrix, samples)
    arg_max = (mean_matrix[0][0], mean_matrix[0][2])

    log_likelihood_vector = np.empty(40000)
    for i in range(40000):
        log_val = MultivariateGaussian.log_likelihood(
            mean_matrix[i], cov_matrix, samples)
        if log_val > max_val:
            max_val = log_val
            arg_max = (mean_matrix[i][0], mean_matrix[i][2])
        log_likelihood_vector[i] = log_val

    z_ = log_likelihood_vector.reshape(200, 200)
    go.Figure(go.Heatmap(x=lin_space, y=lin_space, z=z_), layout=go.Layout(
        title="Log-likelihood based on different mean vectors",
        height=600, width=700)) \
        .update_xaxes(title_text="f3 value") \
        .update_yaxes(title_text="f1 value").show()

    # Question 6 - Maximum likelihood
    # answer in the pdf file!


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
