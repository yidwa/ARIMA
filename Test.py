
import matplotlib.pylab as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def test_stationary(timeseries):
    # Determine rolling statics
    rolmean = timeseries.rolling(window=12, center=False).mean()
    rolstd = timeseries.rolling(window=12, center=False).std()
    #
    # Plot rolling statics
    orig = plt.plot(timeseries.index.to_pydatetime(), timeseries,color='blue', label ='Original')
    mean = plt.plot(rolmean.index.to_pydatetime(), rolmean, color='red', label = 'Rolling Mean')
    std = plt.plot(rolstd.index.to_pydatetime(), rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling mean & Standard Deviation')

    #Perform Dicker-Fuller test

    print "Result of Dicker-Fuller test:"
    dftest = adfuller(timeseries.unstack(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s) ' %key] = value

    print dfoutput
    plt.grid(True)
    plt.show()