import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
import patsy
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import Test

rcParams['figure.figsize'] = 15, 6


# data = pd.read_csv('AirPassengers.csv')
#
# print data.head()c
#
# print '\n Data Types:'
#
# print data.dtypes

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

data = pd.read_csv('AirPassengers.csv' , parse_dates=[0], index_col='Month', date_parser=dateparse)

# print data.head()

# print data.index

# print ts.head(10)

# print ts['1949-01-01' : '1949-05-01']
#
# #print ts[datetime(1949,1,1)]
#
# print ts[:'1949-05-01']

#
ts = data['#Passengers']

# plt.plot(ts)

# # showing the plot
# plt.show()


# test the stationary of time series data
# def test_stationary(timeseries):
#
#     # Determin rolling statics
#     rolmean = timeseries.rolling(window=12, center=False).mean()
#     rolstd = timeseries.rolling(window=12, center=False).std()
#     #
#     # Plot rolling statics
#     orig = plt.plot(timeseries, color='blue', label ='Original')
#     mean = plt.plot(rolmean, color='red', label = 'Rolling Mean')
#     std = plt.plot(rolstd, color='black', label = 'Rolling Std')
#     plt.legend(loc='best')
#     plt.title('Rolling mean & Standard Deviation')
#
#     #Perform Dicker-Fuller test
#
#     print "Result of Dicker-Fuller test:"
#     dftest = adfuller(timeseries, autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
#     for key, value in dftest[4].items():
#         dfoutput['Critical Value (%s) ' %key] = value
#
#     print dfoutput
#     plt.show()
Test.test_stationary(ts)


# apply transformation which penalize higher values more than smaller values
# ts_log = np.log(ts)
# plt.plot(ts_log)
# plt.show()


# remove noise
# moving average
# take average of k consecutive values depending on the frequency of time series

#moving_avg = ts_log.rolling(window=12, center=False).mean()
# plt.plot(ts_log)
# plt.plot(moving_avg, color='red')
# plt.show()

# moving average only define rolling mean for the last month of each year
#ts_log_moving_avg_diff = ts_log - moving_avg
# print ts_log_moving_avg_diff.head(12)


# drop NAN values
#ts_log_moving_avg_diff.dropna(inplace=True)
#test_stationary(ts_log_moving_avg_diff)


# exponentially weighted moving average where weights are assigned to all the previous values with a decay factor
#  no missing values as all values from starting are given weights

# expwighted_avg = ts_log.ewm(halflife=12, ignore_na=False, min_periods=0, adjust=True).mean()
# plt.plot(ts_log)
# plt.plot(expwighted_avg, color='red')
# plt.show()

# ts_log_ewm_diff = ts_log - expwighted_avg
# test_stationary(ts_log_ewm_diff)


# ???? Differencing
# ts_log_diff = ts_log - ts_log.shift()
# # plt.plot(ts_log_diff)
# # plt.show()
#
# ts_log_diff.dropna(inplace=True)
# test_stationary(ts_log_diff)

# Decomposition , remove both trend and seasonal

# decomposition = seasonal_decompose(ts_log)
#
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid
#
# plt.subplot(411)
# plt.plot(ts_log, label='Original')
# plt.legend(loc='best')
#
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
#
# plt.subplot(413)
# plt.plot(seasonal, label='Seasonality')
# plt.legend(loc = 'best')
#
# plt.subplot(414)
# plt.plot(residual, label='Residuals')
# plt.legend(loc ='best')
# plt.tight_layout()

# plt.show()

# model the residuals
# ts_log_decompose = residual
# ts_log_decompose.dropna(inplace=True)
# test_stationary(ts_log_decompose)


# ARIMA model (p,d,q)
# p - number of AR terms
# q - number of moving average terms
# d - number of differences

# acf & pcf
# lag_acf = acf(ts_log_diff, nlags=20)
# lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#
# #plot acf
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.axhline(y=0,linesyte='--', color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
# plt.title('Autocorrelation Function')
#
# #plot pacf
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0,linesyte='--', color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()
#
# plt.show()

# AR MODEL
# model = ARIMA(ts_log, order=(2,1,0))
# result_AR = model.fit(disp=-1)
# plt.plot(ts_log_diff)
# plt.plot(result_AR.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((result_AR.fittedvalues-ts_log_diff)**2))
# plt.show()

# MA model
# model = ARIMA(ts_log, order=(0,1,2))
# result_MA = model.fit(disp=-1)
# plt.plot(ts_log_diff)
# plt.plot(result_MA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((result_MA.fittedvalues-ts_log_diff)**2))
# plt.show()

#Combines Model
# model = ARIMA(ts_log, order=(2,1,2))
# result_ARIMA = model.fit(disp=-1)
# plt.plot(ts_log_diff)
# plt.plot(result_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((result_ARIMA.fittedvalues-ts_log_diff)**2))
# plt.show()

# prediction_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues, copy=True)
# print prediction_ARIMA_diff.head()

# prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
#print prediction_ARIMA_diff_cumsum.head()

#
# prediction_ARIMA_log = pd.Series(ts_log.ix[0], index= ts_log.index)
# prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum, fill_value=0)
# print prediction_ARIMA_log.head()

# prediction_ARIMA = np.exp(prediction_ARIMA_log)
# plt.plot(ts)
# plt.plot(prediction_ARIMA)
# plt.title('RMSE: %.4f' % np.sqrt(sum((prediction_ARIMA-ts))**2/len(ts)))
# plt.show()