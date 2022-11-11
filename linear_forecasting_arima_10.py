import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

df1=pd.read_csv("/home/dan/VScode/Python/DF.csv")
print(df1.head())
print(df1.tail())

df = df1.dropna()
df.columns=["date","size"]
print(df1.head())
print(df.tail())
df['date']=pd.to_datetime(df['date'])
df.set_index('date',inplace=True)
print()
df.describe()
print()

df.plot()
plt.show()

from statsmodels.tsa.stattools import adfuller
print('adfuller test')
test_result=adfuller(df['size'])
#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(size):
    result=adfuller(size)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    
adfuller_test(df['size'])
print()

df['Size difference'] = df['size'] - df['size'].shift(1)
df['size'].shift(1)

df['Seasonal First Difference']=df['size']-df['size'].shift(12)
df.head(14)

## Again test dickey fuller test
adfuller_test(df['Seasonal First Difference'].dropna())
df['Seasonal First Difference'].plot()
plt.show()

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['size'])
plt.show()

#from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
#import statsmodels.api as sm
#fig = plt.figure(figsize=(12,8))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)

from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(df['size'],order=(1,1,1))
model_fit=model.fit()
print(model_fit.summary())

df['forecast']=model_fit.predict(start=80,end=120,dynamic=True)
df[['size','forecast']].plot(figsize=(12,8))
plt.show()

import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(df['size'],order=(3, 1, 1),seasonal_order=(3,1,0,12))
results=model.fit()
df['forecast']=results.predict(start=80,end=120,dynamic=True)
df[['size','forecast']].plot(figsize=(12,8))
plt.show()

from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
print(future_datest_df.tail())

future_df=pd.concat([df,future_datest_df])
future_df['forecast'] = results.predict(start = 120, end = 150, dynamic= True)  
future_df[['size', 'forecast']].plot(figsize=(12, 8)) 
plt.show()

print(df)


#https://github.com/krishnaik06/ARIMA-And-Seasonal-ARIMA/blob/master/Untitled.ipynb