import os
from warnings import warn

import numpy as np
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformers.single_series.boxcox import BoxCoxTransformer
from sktime.transformers.single_series.detrend import ConditionalDeseasonalizer
from sktime.transformers.single_series.detrend import Detrender
from xgboost import XGBRegressor

from sktime.performance_metrics.forecasting import mase_loss
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.validation.forecasting import check_sp
from sktime.utils.validation.forecasting import check_y
from statsmodels.tsa.stattools import acf

SEASONAL_MODEL = "multiplicative"

ses = ExponentialSmoothing()
holt = ExponentialSmoothing(trend="add", damped=False)
damped = ExponentialSmoothing(trend="add", damped=True)


def M4_owa_loss(mase, smape, naive2_mase, naive2_smape):
    """overall weighted average of sMAPE and MASE loss used in M4 competition
    References
    ----------
    ..[1]  https://github.com/Mcompetitions/M4-methods/blob/master
    /Benchmarks%20and%20Evaluation.R
    """
    return ((np.nanmean(smape) / np.mean(naive2_smape)) + (
            np.nanmean(mase) / np.mean(naive2_mase))) / 2


def seasonality_test_R(y, sp):
    """Seasonality test used in M4 competition
    R and Python versions were inconsistent [2], this is the Python
    translation of the R version [1].
    References
    ----------
    ..[1]  https://github.com/Mcompetitions/M4-methods/blob/master
    /Benchmarks%20and%20Evaluation.R
    ..[2]  https://github.com/Mcompetitions/M4-methods/issues/25
    """
    y = check_y(y)
    y = np.asarray(y)
    n_timepoints = len(y)

    sp = check_sp(sp)
    if sp == 1:
        return False

    if n_timepoints < 3 * sp:
        warn(
            "Did not perform seasonality test, as `y`` is too short for the "
            "given `sp`, returned: False")
        return False

    else:
        coefs = acf(y, nlags=sp, fft=False)  # acf coefficients
        coef = coefs[sp]  # coefficient to check

        tcrit = 1.645  # 90% confidence level
        limits = tcrit / np.sqrt(n_timepoints) * np.sqrt(
            np.cumsum(np.append(1, 2 * coefs[1:] ** 2)))
        limit = limits[sp - 1]  # zero-based indexing
        return np.abs(coef) > limit


def seasonality_test_Python(y, sp):
    """Seasonality test used in M4 competition
    R and Python versions were inconsistent [2], this is a copy of the
    Python version [1].
    References
    ----------
    ..[1]  https://github.com/M4Competition/M4-methods/blob/master
    /ML_benchmarks.py
    ..[2]  https://github.com/Mcompetitions/M4-methods/issues/25
    """

    if sp == 1:
        return False

    def _acf(data, k):
        m = np.mean(data)
        s1 = 0
        for i in range(k, len(data)):
            s1 = s1 + ((data[i] - m) * (data[i - k] - m))

        s2 = 0
        for i in range(0, len(data)):
            s2 = s2 + ((data[i] - m) ** 2)

        return float(s1 / s2)

    s = _acf(y, 1)
    for i in range(2, sp):
        s = s + (_acf(y, i) ** 2)

    limit = 1.645 * (np.sqrt((1 + 2 * s) / len(y)))

    return (abs(_acf(y, sp))) > limit


def make_pipeline(*estimators):
    """Helper function to make pipeline"""
    steps = [(estimator.__class__.__name__, estimator) for estimator in
             estimators]
    return TransformedTargetForecaster(steps)


def deseasonalise(forecaster, seasonality_test=seasonality_test_R, **kwargs):
    return make_pipeline(
        ConditionalDeseasonalizer(seasonality_test, **kwargs),
        forecaster
    )


def boxcox(forecaster):
    return make_pipeline(
        BoxCoxTransformer(bounds=(0, 1)),
        forecaster
    )


def deseasonalise_boxcox(forecaster, seasonality_test=seasonality_test_R,
                         **kwargs):
    return make_pipeline(
        ConditionalDeseasonalizer(seasonality_test, **kwargs),
        BoxCoxTransformer(bounds=(0, 1)),
        forecaster
    )


def construct_M4_forecasters(sp, fh):
    kwargs = {"model": SEASONAL_MODEL, "sp": sp} if sp > 1 else {}

    theta_bc = make_pipeline(
        ConditionalDeseasonalizer(seasonality_test=seasonality_test_R,
                                  **kwargs),
        BoxCoxTransformer(bounds=(0, 1)),
        ThetaForecaster(deseasonalise=False)
    )
    """
    MLP = make_pipeline(
        ConditionalDeseasonalizer(seasonality_test=seasonality_test_Python,
                                  **kwargs),
        Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
        RecursiveRegressionForecaster(
            regressor=MLPRegressor(hidden_layer_sizes=6, activation="identity",
                                   solver="adam", max_iter=100,
                                   learning_rate="adaptive",
                                   learning_rate_init=0.001),
            window_length=3)
    )
    RNN = make_pipeline(
        ConditionalDeseasonalizer(seasonality_test=seasonality_test_Python,
                                  **kwargs),
        Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
        RecursiveTimeSeriesRegressionForecaster(
            regressor=SimpleRNNRegressor(nb_epochs=100),
            window_length=3)
    )
    """
    forecasters = {
        "Naive": NaiveForecaster(strategy="last"),
        "sNaive": NaiveForecaster(strategy="seasonal_last", sp=sp),
        "Naive2": deseasonalise(NaiveForecaster(strategy="last"), **kwargs),
        "SES": deseasonalise(ses, **kwargs),
        "Holt": deseasonalise(holt, **kwargs),
        "Damped": deseasonalise(damped, **kwargs),
        "Theta": deseasonalise(ThetaForecaster(deseasonalise=False), **kwargs),
        "ARIMA": AutoARIMA(suppress_warnings=True, error_action="ignore",
                           sp=sp),
        "Com": deseasonalise(EnsembleForecaster(
            [("ses", ses), ("holt", holt), ("damped", damped)]), **kwargs),
        # "MLP": MLP,
        # "RNN": RNN,
        "260": theta_bc,
    }
    return forecasters
