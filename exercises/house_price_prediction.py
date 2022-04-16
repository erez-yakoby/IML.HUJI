import scipy.stats

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)

    # drop rows by values that are probably a mistake in my opinion
    df.drop(df[(df.bathrooms < 1) | (df.price < 0) |
               (df.bedrooms < 1) | (df.sqft_lot15 < 0)].index, inplace=True)
    df = df.dropna()

    # this new column will hold the year of the latest physical house change
    df["yr_changed"] = df[["yr_renovated", "yr_built"]].max(axis=1)

    # those columns are temporarily droped so that the filtering by std
    # will not remove unnecessary rows
    water_front_copy = df["waterfront"]
    zipcode_copy = df["zipcode"]
    price_copy = df["price"]

    df.drop(columns=["id", "date", "lat", "long", "yr_renovated", "yr_built"],
            inplace=True)
    df.drop(columns=["zipcode", "waterfront", "price"], inplace=True)

    # filtering by std
    z_score = scipy.stats.zscore(df)
    abs_z_scores = np.abs(z_score)
    filtered = (abs_z_scores < 3).all(axis=1)

    # adding the columns that were removed for the filtering
    df = pd.concat((df[filtered], water_front_copy[filtered],
                    zipcode_copy[filtered], price_copy[filtered]), axis=1)

    df = pd.get_dummies(df, columns=["zipcode"])

    return df.drop(columns=["price"]), df["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    sigma_x = np.std(X)
    sigma_y = np.std(y)

    cov = np.dot(y.T - y.mean(), X - X.mean(axis=0)) / (y.shape[0] - 1)
    corr = cov / (sigma_x * sigma_y)

    for col in X.columns:
        # I chose to not create the graphs for the zipcode dummies because it
        # takes a lot of computation time. I added to the pdf, a graph with all
        # the features correlations (including zipcode).
        if "zipcode" in col:
            continue
        feature = X[col]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feature, y=y, mode="markers"))
        fig.update_layout(
            title="Feature: " + col + ". Pearson Correlation = " + str(
                corr[col]),
            xaxis_title="Feature value",
            yaxis_title="Price")
        file_name = col + "_to_price.png"
        fig.write_image(output_path + file_name)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    design_matrix, responses = load_data("house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(design_matrix, responses)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(design_matrix,
                                                        responses)

    # Question 4 - Fit model over increasing percentages of the overall
    # training data For every percentage p in 10%, 11%, ..., 100%, repeat the
    # following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of
    # size (mean-2*std, mean+2*std)

    percentages = np.arange(10, 101)

    mean_loss = np.empty((91))
    std_loss = np.empty((91))

    for p in percentages:
        cur_loss = np.empty((10, 1))
        for i in range(10):
            current_sample = train_x.sample(frac=(p / 100), random_state=i)
            current_response = train_y.sample(frac=(p / 100), random_state=i)
            model = LinearRegression()
            model.fit(np.array(current_sample), np.array(current_response))
            cur_loss[i] = model.loss(np.array(test_x), np.array(test_y))

        mean_loss[p - 10] = np.mean(cur_loss)
        std_loss[p - 10] = np.std(cur_loss)

    fig = go.Figure()
    fig.add_traces([go.Scatter(x=percentages, y=mean_loss, name="mean loss"),
                    go.Scatter(x=percentages, y=mean_loss - 2 * std_loss,
                               fill=None, mode="lines",
                               line=dict(color="lightgrey"), showlegend=True,
                               name="mean +- (2 * std)"),
                    go.Scatter(x=percentages, y=mean_loss + 2 * std_loss,
                               fill="tonexty", mode="lines",
                               line=dict(color="lightgrey"),
                               showlegend=False)])

    fig.update_layout(
        title="Mean of square loss as a function of train data percentage",
        xaxis_title="Percentage",
        yaxis_title="Average loss")
    fig.show()
