import matplotlib.pyplot as plt
import datetime
import pandas as pd
import matplotlib.dates as dates
import numpy as np
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
import Test
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

rcParams['figure.figsize'] = 15, 8

dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')

data = pd.read_csv('grid_training.csv', parse_dates=[0], index_col='Time', date_parser=dateparse)

# print data.index

# remove three data record with 3000/2000/1700 count

# print data.head()
# plt.plot(data.index.to_pydatetime(), data['Count'])
# plt.show()

# ts = data['#Count']
# print ts.head(10)
# # Test.test_stationary(ts)
data_log = np.log(data)
# print data_log.index

decomposition = seasonal_decompose(data_log,freq = 12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid



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
#
data_first_diff = data_log_decompose - data_log_decompose.shift(1)
data_first_diff.dropna(inplace=True)
#
data_season_diff = data_first_diff - data_first_diff.shift(12)
data_season_diff.dropna(inplace=True)
#
# Test.test_stationary(data_log_decompose)


# ARIMA model (p,d,q)c
# p - number of AR terms
# q - number of moving average terms
# d - number of differences
#
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(data_season_diff.iloc[1:], lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(data_season_diff.iloc[1:], lags=40, ax=ax2)

# plt.show()
# acf & pcf
lag_acf = acf(data_log_decompose, nlags=20)
lag_pacf = pacf(data_log_decompose, nlags=20, method='ols')

data_log_diff = data_log-data_log.shift()
#plot acf
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.axhline(y=0, color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(data_log_diff)), color='gray')
# plt.axhline(y=1.96/np.sqrt(len(data_log_diff)),  color='gray')
# plt.title('Autocorrelation Function')
#
# #plot pacf
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0, color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(data_log_diff)), color='gray')
# plt.axhline(y=1.96/np.sqrt(len(data_log_diff)),  color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()
#
# plt.show()

# Combines Model
model = ARIMA(data_log, order=(2,1,2))
result_ARIMA = model.fit(disp=-1)
plt.plot(data_log_diff)
plt.plot(result_ARIMA.fittedvalues, color='red')
# # plt.title('RSS: %.4f'% sum((result_ARIMA.fittedvalues-data_log)**2))
plt.show()

# prediction_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues, copy=True)
# print prediction_ARIMA_diff.head()
#
# prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
# print prediction_ARIMA_diff_cumsum.head()
#
#
# prediction_ARIMA_log = pd.Series(data_log.ix[0], index= data_log.index)
# prediction_ARIcMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum, fill_value=0)
# # print prediction_ARIMA_log.head()
#
# prediction_ARIMA = np.exp(prediction_ARIMA_log)
# plt.plot(data.index.to_pydatetime(), data['Count'])
# plt.plot(prediction_ARIMA.index.to_pydatetime(), prediction_ARIMA, color='red')
# plt.title('RMSE: %.4f'% np.sqrt(sum((prediction_ARIMA-data['Count']))**2/len(data)))
# plt.show()