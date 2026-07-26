---
name: Time Series Analysis
description: Analyze temporal data patterns including trends, seasonality, autocorrelation, and forecasting for time series decomposition, trend analysis, and forecasting models
---

# Time Series Analysis

## Overview

Time series analysis examines data points collected over time to identify patterns, trends, and seasonality for forecasting and understanding temporal dynamics.

## When to Use

- Forecasting future values based on historical trends
- Detecting seasonality and cyclical patterns in data
- Analyzing trends over time in sales, stock prices, or website traffic
- Understanding autocorrelation and temporal dependencies
- Making time-based predictions with confidence intervals
- Decomposing data into trend, seasonal, and residual components

## Core Components

- **Trend**: Long-term directional movement
- **Seasonality**: Repeating patterns at fixed intervals
- **Cyclicity**: Long-term oscillations (non-fixed periods)
- **Stationarity**: Constant mean, variance over time
- **Autocorrelation**: Correlation with past values

## Key Techniques

- **Decomposition**: Separating trend, seasonal, residual components
- **Differencing**: Making data stationary
- **ARIMA**: AutoRegressive Integrated Moving Average models
- **Exponential Smoothing**: Weighted average of past values
- **SARIMA**: Seasonal ARIMA models

## Implementation with Python

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Create sample time series data
dates = pd.date_range('2020-01-01', periods=365, freq='D')
values = 100 + np.sin(np.arange(365) * 2*np.pi / 365) * 20 + np.random.normal(0, 5, 365)
ts = pd.Series(values, index=dates)

# Decomposition
decomposition = seasonal_decompose(ts, model='additive', period=30)

# Test for stationarity (Augmented Dickey-Fuller)
result = adfuller(ts)
print(f"ADF p-value: {result[1]:.6f}")
if result[1] <= 0.05:
    print("Time series is stationary")
else:
    print("Time series is non-stationary - differencing needed")

# ARIMA Model
arima_model = ARIMA(ts, order=(1, 1, 1))
arima_result = arima_model.fit()

# Forecast
forecast_steps = 30
forecast = arima_result.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
```

## Stationarity

- **Stationary**: Mean, variance, autocorrelation constant over time
- **Non-stationary**: Trend or seasonal patterns present
- **Solution**: Differencing, log transformation, or detrending

## Model Selection

- **ARIMA**: Good for univariate forecasting
- **SARIMA**: Includes seasonal components
- **Exponential Smoothing**: Simpler, good for trends
- **Prophet**: Handles holidays and changepoints

## Evaluation Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
