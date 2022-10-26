# AR: Авторегрессия. Модель, которая использует зависимую связь между наблюдением и некоторым количеством запаздывающих наблюдений.
# I: Интегрированный. Использование дифференцирования необработанных наблюдений (например, вычитание наблюдения из наблюдения на предыдущем временном шаге), чтобы сделать временной ряд стационарным.
# MA: скользящая средняя. Модель, которая использует зависимость между наблюдением и остаточной ошибкой из модели скользящего среднего, применяется к запаздывающим наблюдениям.

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pmdarima as pm
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
import platform
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from pmdarima.arima.stationarity import ADFTest
import statsmodels.api as sm

# версии
print('Python: ', platform.python_version())
print('pandas: ', pd.__version__) #инструмент анализа данных
#print('matplotlib: ', plt.__version__) #инструмент визуализации данных
print('pmdarima: ', pm.__version__) #автоматизированный инструмент прогнозирования
print('statsmodels: ', statsmodels.__version__) #инструмент статистических моделей
print('NumPy: ', np.__version__) #инструмент для научных вычислений
print('sklearn: ', sklearn.__version__) #машинное обучение

# Upload and assign to pandas
data1 = pd.read_csv("/home/dan/VScode/Python/data_forecast.csv", index_col = 0)
#Remove Nan values
data = data1.dropna()
data.index = pd.to_datetime(data.index) # Переводим индекс в дату
data.info()

# Очистка данных
size_dif = data.loc['2022-07-15':'2022-10-12']

# Настройки таблиц
fig_size = plt.rcParams["figure.figsize"]
 
# table sizing
fig_size[0] = 30
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size

# резет индекса для построения диаграммы
size_dif.reset_index().plot(x='date', y='size', kind='line', grid=1)
plt.show()

# Возвращаем индекс к дате
data.index = pd.to_datetime(data.index)

# Тест ADF на стационарность дат
adf_test = ADFTest(alpha=0.05)
p_val, should_diff = adf_test.should_diff(data['size']) 

if p_val < 0.05:
    print('Time Series is stationary. p-value is ',  p_val)
else:
    print('Time Series is not stationary. p-value is ',  p_val, '. Differencing is needed: ', should_diff)
# Временной ряд не стационарен.
# Оцениваем различия АРИМА при нестационарном ряде
from pmdarima.arima.utils import ndiffs

# Оценка кол-ва различий метод ADF
n_adf = ndiffs(data['size'], test='adf') 
print('n_adf:', n_adf)
print()

# и тестов KPSS (auto_arima default):
n_kpss = ndiffs(data['size'], test='kpss') 
print('n_kpss:', n_kpss)
if n_adf == 1 & n_kpss == 1:
    print('Use differencing value of 1 when creating ARIMA model')    
else:
    print('Differencing is not needed when creating ARIMA model')
n_adf: 1
n_kpss: 1

total_data = data.filter(items=['size'])
print('All: ', total_data.shape)
print()

# Разделение данных для прогноза и тестов 80/20
data_total_train = total_data.loc['2022-07-15':'2022-09-23']
data_total_test = total_data.loc['2022-09-23':]
print( 'Train: ', data_total_train.shape)
print( 'Test: ', data_total_test.shape)
print()

# Поиск оптимальных параметров перебором
data_total_fit = pm.auto_arima(data_total_train, 
                           m=3, # Данные за 3 месяца 
                           seasonal=False, # Не имеет сезонности
                           d=1, # Данные не являются стационарными и требуется дифференциация 1 (посчитано выше)
                           trace=True, 
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=False) # Поиск методом перебора, поскольку набор дат небольшой, а не интеллектуальный поиск
print()
# Передача гиперпараметров (другие библоитечки)
mod = sm.tsa.statespace.SARIMAX(data,
order=(1, 1, 1),
seasonal_order=(0, 0, 0, 0),
enforce_stationarity=False,
enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
# вывод плотов
results.plot_diagnostics(figsize=(15, 12))
plt.show()
# прогноз
pred = results.get_prediction(start=pd.to_datetime('2022-07-15'), dynamic=False) # пошаговое прогнозирование (прогнозы в каждой точке генерируются с использованием полной истории)
pred_ci = pred.conf_int()
ax = data['2022-07-15':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
pred_ci.iloc[:, 0],
pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('date')
ax.set_ylabel('size')
plt.legend()
plt.show()

data_forecasted = pred.predicted_mean
data_truth = data['2022-07-15':] # Compute the mean square error
mse = ((data_forecasted - data_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# Получить прогноз на 500 шагов вперёд
pred_uc = results.get_forecast(steps=100)
# Получить интервал прогноза
pred_ci = pred_uc.conf_int()

ax = data.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
pred_ci.iloc[:, 0],
pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('date')
ax.set_ylabel('size')
plt.legend()
plt.show()
