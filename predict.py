# Import necessary libraries
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Assume that df is your DataFrame and 'number1' is the column with the numbers
df = pd.read_csv('pathto.csv')  # replace with your actual csv file

# Prepare a DataFrame to store the predictions
predictions = pd.DataFrame()

# Fit model and make predictions for each column
for i in range(6):
    series = df.iloc[:, i]
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast(steps=3)
    yhat = output.apply(round)  # round to the nearest whole number
    predictions[f'Predicted future values for column {i+1}'] = yhat

# Print and save the predictions
print(predictions)
predictions.to_csv('predictions.csv', index=False)
