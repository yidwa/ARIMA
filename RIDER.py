import matplotlib.pyplot as plt
import datetime
import pandas as pd
import matplotlib.dates as dates
import numpy as np
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import Test
import statsmodels.api as sm

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

data = pd.read_csv('Portland.csv' , index_col='Month', date_parser=dateparse)


# print data.head()
# plt.plot(data.index.to_pydatetime(), data['Portland'])


ts = data['Portland']
# Test.test_stationary(ts)


# apply transformation which penalize higher values more than smaller values
data_log = np.log(ts)
# plt.plot(data_log.index.to_pydatetime(), data_log)
# plt.show()


# remove noise
# moving average
# take average of k consecutive values depending on the frequency of time series

# moving_avg = ts_log.rolling(window=12, center=False).mean()
# plt.plot(ts_log)
# plt.plot(moving_avg, color='red')
# plt.show()

# moving average only define rolling mean for the last month of each year
# ts_log_moving_avg_diff = ts_log - moving_avg
# print ts_log_moving_avg_diff.head(12)


# drop NAN values
# ts_log_moving_avg_diff.dropna(inplace=True)
# test_stationary(ts_log_moving_avg_diff)


# exponentially weighted moving average where weights are assigned to all the previous values with a decay factor
#  no missing values as all values from starting are given weights

# expwighted_avg = ts_log.ewm(halflife=12, ignore_na=False, min_periods=0, adjust=True).mean()
# plt.plot(ts_log)
# plt.plot(expwighted_avg, color='red')
# plt.show()

# ts_log_ewm_diff = ts_log - expwighted_avg
# test_stationary(ts_log_ewm_diff)

# Differencing - take the difference with a particular time lag
# ts_log_diff = ts_log - ts_log.shift()
# plt.plot(ts_log_diff)
# plt.show()
#
# ts_log_diff.dropna(inplace=True)
# test_stationary(ts_log_diff)

# Decomposition , remove both trend and seasonal

decomposition = seasonal_decompose(data_log)
#
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
#

def showing_decomposition():
    plt.subplot(411)
    plt.plot(data_log.index.to_pydatetime(), data_log, label='Original')
    plt.legend(loc='best')
#
    plt.subplot(412)
    plt.plot(trend.index.to_pydatetime(), trend, label='Trend')
    plt.legend(loc='best')
#
    plt.subplot(413)
    plt.plot(seasonal.index.to_pydatetime(), seasonal, label='Seasonality')
    plt.legend(loc = 'best')
#
    plt.subplot(414)
    plt.plot(residual.index.to_pydatetime(), residual, label='Residuals')
    plt.legend(loc ='best')
    plt.tight_layout()

    plt.grid(True)
    plt.show()



# showing_decomposition()


# model the residuals
data_log_decompose = residual
data_log_decompose.dropna(inplace=True)

data_first_diff = data_log_decompose - data_log_decompose.shift(1)
data_first_diff.dropna(inplace=True)

data_season_diff = data_first_diff - data_first_diff.shift(12)
data_season_diff.dropna(inplace=True)

# Test.test_stationary(data_season_diff)


# ARIMA model (p,d,q)
# p - number of AR terms
# q - number of moving average terms
# d - number of differences

# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(data_season_diff.iloc[13:], lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(data_season_diff.iloc[13:], lags=40, ax=ax2)

# plt.show()


