#### DEVELOPED BY: GOWRISANKAR P
#### REG NO: 212222230041
#### Date:
# Ex.No: 6               HOLT WINTERS METHOD
 
### AIM:
To forecast future prices using Holt-Winters exponential smoothing by analyzing GOLD_PRICE_PREDICTION prices. The goal is to predict closing prices for next year.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np


file_path = 'FINAL_USO.csv'
data = pd.read_csv(file_path)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data_resampled = data['USO_Close'].resample('MS').mean()

data_clean = data_resampled.ffill()

train_size = len(data_clean) - 12  # Keeping the last 12 months for testing
train_data = data_clean[:train_size]
test_data = data_clean[train_size:]

model = ExponentialSmoothing(train_data, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()

test_predictions = fit.forecast(steps=len(test_data))

rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
print(f'Test RMSE: {rmse}')

plt.figure(figsize=(10, 6))
plt.plot(data_clean, label='USO Close Price (Monthly)')
plt.plot(test_data.index, test_predictions, label='Test Predictions', color='orange')
plt.title('Test USO Monthly Closing Price')
plt.xlabel('Date')
plt.ylabel('USO Close Price')
plt.legend()
plt.show()

n_steps = 12
final_predictions = fit.forecast(steps=n_steps)

plt.figure(figsize=(10, 6))
plt.plot(data_clean.index, data_clean, label='Original Data')

plt.plot(pd.date_range(start=data_clean.index[-1], periods=n_steps+1, freq='MS')[1:], final_predictions, label='Final Forecast', color='green')
plt.xlabel('Date')
plt.ylabel('USO Closing Price')
plt.title('Final Predictions for USO Closing Prices')
plt.legend()
plt.show()

```

### OUTPUT:

### TEST_PREDICTION:
![image](https://github.com/user-attachments/assets/0d55198f-322f-447c-b647-5790428586f7)


### FINAL_PREDICTION:
![image](https://github.com/user-attachments/assets/834b6541-9d02-48c8-a010-9cdea83d6272)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
