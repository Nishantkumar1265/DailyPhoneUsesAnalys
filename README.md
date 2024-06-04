# Dataset loading
import pandas as pd
data = pd.read_csv('/content/Nishant dataset.csv')
# head
data.head()

# data describe
data.describe()


# convert the 'Date column to a datatime type
data['Date'] = pd.to_datetime(data['Date'])

# Set the 'Date column as the index
data.set_index('Date', inplace=True)
data

import matplotlib.pyplot as plt
# create a line plot
plt.plot(data.index,data['PhoneUsesINhours'])
# add label and title
plt.xlabel('Date')
plt.ylabel('PhoneUsesINhours')
plt.title('Daily Phone Uses Time Data')
# display the plot
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition =seasonal_decompose(data['PhoneUsesINhours'] ,period = 90)
trend = decomposition.trend
seasonality=decomposition.seasonal
residuals= decomposition.resid
decomposition.resid

plt.figure(figsize=(10,8))
plt.subplot(411)
plt.plot(data.index,data['PhoneUsesINhours'], label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(data.index,trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(data.index,seasonality, label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(data.index,residuals, label='Residuals')
plt.legend(loc='best')


plt.tight_layout()
plt.show()


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# plot ACF
plt.figure(figsize=(10,4))
plot_acf(data['PhoneUsesINhours'], lags=50)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function(ACF)')
plt.show()

# plot PACF
plt.figure(figsize=(10,4))
plot_pacf(data['PhoneUsesINhours'], lags=50)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function(PACF)')
plt.show()

from statsmodels.tsa.stattools import adfuller
result=adfuller(data['PhoneUsesINhours'])
adf_statistic = result[0]
p_value=result[1]
print(f'ADF Statistic: {adf_statistic:.4f}')
print(f'p_value: {p_value:.4f}')
# define a function to interpret the test results
def interpret_adf_results(p_value):
  if p_value < 0.05: #compared to 5% error
    print('The time series is stationary')
  else:
    print('The time series is non-stationary')
# interpret the test results
interpret_adf_results(p_value)


!pip install pycaret

df=data['PhoneUsesINhours']



from pycaret.time_series import TSForecastingExperiment
fig_kwargs={'renderer': 'notebook'}
forecast_horizon= 10
fold=3
exp=TSForecastingExperiment()
exp.setup(data=df,fh=forecast_horizon,fold=fold, session_id=123,fig_kwargs=fig_kwargs)
exp.check_stats()

exp.models()

arima = exp.create_model('arima')
arima

tuned_arima= exp.tune_model(arima)
tuned_arima

best=exp.compare_models()

for model in best_models:
    exp.predict_model(model)




import matplotlib.pyplot as plt

# Model names and RMSE values
model_names = []
rmse_values = []

# Add model names and RMSE values from best_models
for model in best_models:
    model_names.append(model)
    rmse_values.append(best_models[model]['RMSE'])

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(model_names, rmse_values, color='blue')

# Set labels and title
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Comparison of Models based on RMSE')

# Show the plot
plt.show()


best_models

import matplotlib.pyplot as plt

# Model names and RMSE values
model_names = []
rmse_values = []

# Add model names and RMSE values from exp.models
for model in best_models:
    model_names.append(exp.models[model].name)
    rmse_values.append(best_models[model]['RMSE'])

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(model_names, rmse_values, color='blue')

# Set labels and title
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Comparison of Models based on RMSE')

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

# Show the plot
plt.show()


import matplotlib.pyplot as plt

# Model names and RMSE values
model_names = []
rmse_values = []

# Add model names and RMSE values from exp.models
for model in best_models:
    model_names.append(model)
    rmse_values.append(best_models[model]['RMSE'])

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(model_names, rmse_values, color='blue')

# Set labels and title
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Comparison of Models based on RMSE')

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

# Show the plot
plt.show()
