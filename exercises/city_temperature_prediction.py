import calendar

import IMLearn.metrics.loss_functions
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df.dropna()
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df.drop(df[(df.DayOfYear > 365) | (df.DayOfYear < 1)].index, inplace=True)
    df.drop(df[(df.Temp < -20)].index, inplace=True)
    return df


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = data.groupby("Country").get_group("Israel")

    year_seperated = israel_data.groupby("Year")
    temp_by_day_of_year_fig = go.Figure()
    for year in range(1995, 2008):
        current_year_df = year_seperated.get_group(year)
        temp_by_day_of_year_fig.add_trace(
            go.Scatter(x=current_year_df["DayOfYear"],
                       y=current_year_df["Temp"],
                       name=str(year), mode="markers"))

    temp_by_day_of_year_fig.update_layout(
        title="Temperature based on Day of year",
        xaxis_title="Day of year",
        yaxis_title="Temperature")
    temp_by_day_of_year_fig.show()

    grouped_by_months = israel_data.groupby("Month")
    months = np.array(calendar.month_name[1:])
    temp_std = [np.std(grouped_by_months.get_group(month)["Temp"]) for month in
                range(1, 13)]
    months_std_fig = go.Figure().add_trace(go.Bar(x=months, y=temp_std))
    months_std_fig.update_layout(
        title="Temperature's std per month",
        xaxis_title="Month",
        yaxis_title="Temperature's std")
    months_std_fig.show()

    # Question 3 - Exploring differences between countries

    grouped_by_country = data.groupby("Country")
    months_avg_temp = go.Figure()
    countries = data["Country"].unique()
    for country in countries:
        grouped = grouped_by_country.get_group(country).groupby("Month")
        mean_temp = grouped.mean()["Temp"]
        std_temp = grouped.std()["Temp"]
        months_avg_temp.add_trace(
            go.Scatter(x=months, y=mean_temp, name=str(country),
                       mode="lines+markers",
                       error_y=dict(array=std_temp)))

    months_avg_temp.update_layout(
        title="Average temperature based on month",
        xaxis_title="Month",
        yaxis_title="Average Temperature")
    months_avg_temp.show()

    # Question 4 - Fitting model for different values of `k`

    train_x, train_y, test_x, test_y = split_train_test(
        israel_data["DayOfYear"], israel_data["Temp"])
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    loss_for_k = np.empty(10)
    for k in range(1, 11):
        model = PolynomialFitting(k).fit(train_x, train_y)
        loss_for_k[k - 1] = round(model.loss(test_x, test_y), 2)
        print("loss for model with polynomial degree k =", k, "is:",
              loss_for_k[k - 1])
    k_deg_mse_fig = go.Figure().add_trace(
        go.Bar(x=np.arange(1, 11), y=loss_for_k))
    k_deg_mse_fig.update_layout(
        title="MSE of a model as a function of it's polynomial degree",
        xaxis_title="Polynomial degree",
        yaxis_title="MSE")
    k_deg_mse_fig.show()

    # Question 5 - Evaluating fitted model on different countries
    israel_model = PolynomialFitting(5).fit(
        israel_data["DayOfYear"].to_numpy(), israel_data["Temp"].to_numpy())
    countries_loss = []
    countries_without_israel = countries[countries != "Israel"]
    for country in countries_without_israel:
        country_df = grouped_by_country.get_group(country)
        countries_loss.append(
            israel_model.loss(country_df["DayOfYear"].to_numpy(),
                              country_df["Temp"].to_numpy()))

    countries_mse_fig = go.Figure().add_trace(
        go.Bar(x=countries_without_israel, y=np.array(countries_loss)))
    countries_mse_fig.update_layout(
        title="MSE of each country based on Israel's best model",
        xaxis_title="Country",
        yaxis_title="MSE")
    countries_mse_fig.show()
