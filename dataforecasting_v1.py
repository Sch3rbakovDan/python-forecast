# AR: Авторегрессия. Модель, которая использует зависимую связь между наблюдением и некоторым количеством запаздывающих наблюдений.
# I: Интегрированный. Использование дифференцирования необработанных наблюдений (например, вычитание наблюдения из наблюдения на предыдущем временном шаге), чтобы сделать временной ряд стационарным.
# MA: скользящая средняя. Модель, которая использует зависимость между наблюдением и остаточной ошибкой из модели скользящего среднего, применяется к запаздывающим наблюдениям.

import pandas as pd
import matplotlib as plt
import pmdarima as pm
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
import platform
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from pmdarima.arima.stationarity import ADFTest

# версии
print('Python: ', platform.python_version())
print('pandas: ', pd.__version__) #инструмент анализа данных
print('matplotlib: ', plt.__version__) #инструмент визуализации данных
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
plt.pyplot.show()

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
# Сводка выбранных оптимальных гиперпараметров
print('Сводка выбранных оптимальных гиперпараметров')
print (data_total_fit.summary())
print()

# Передача гиперпараметров
data_total_fit.fit(data_total_train)
# Построение прогноза на 20 дней
data_forecast = data_total_fit.predict(n_periods=20)
# Сравнение исторических / фактических данных с прогнозом
data_total_validate = pd.DataFrame(data_forecast, index = data_total_test.index, columns=['Prediction'])
data_total_validate = pd.concat([data_total_test, data_total_validate], axis=1)
print('Сравнение исторических и фактических данных с прогнозом')
print (data_total_validate)
print()

# Построение графика различий
data_total_validate.plot()
plt.pyplot.show()

# Вычисление абсолютной разницы между фактом и прогнозом
data_total_validate['Abs Diff'] = (data_total_validate['size'] - data_total_validate['Prediction']).abs()
data_total_validate['Abs Diff %'] = (data_total_validate['size'] - data_total_validate['Prediction']).abs()/data_total_validate['size']
data_total_validate.loc['Average Abs Diff %'] = pd.Series(data_total_validate['Abs Diff %'].mean(), index = ['Abs Diff %'])
data_total_validate.loc['Min Abs Diff %'] = pd.Series(data_total_validate['Abs Diff %'].min(), index = ['Abs Diff %'])
data_total_validate.loc['Max Abs Diff %'] = pd.Series(data_total_validate['Abs Diff %'].max(), index = ['Abs Diff %'])
print('Вычисление абсолютной разницы между фактом и прогнозом')
print (data_total_validate)
print()
# Сравнение всех данных
data_total = total_data.filter(items=['size'])
data_total_1 = total_data.loc['2022-07-15':'2022-09-23']
data_total_2 = total_data.loc['2022-09-24':]
data_forecast_all = data_total_fit.predict(n_periods=90)
print('Сравнение всех данных')
print (data_forecast_all)
print()

#Построение графика
data_validate_all = pd.DataFrame(data_forecast_all, index = data_total_2.index, columns=['Prediction'])
data_validate_all = pd.concat([data_total_2, data_validate_all], axis=1)
data_validate_all = data_total_1.append(data_validate_all, sort=True)
data_validate_all.plot()
plt.pyplot.show()
