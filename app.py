import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

# Specify the path to your Excel file here
file_path = 'data.xlsx'

# Read the Excel file
data = pd.read_excel(file_path)

# Clean the data (remove NULLs and unnecessary columns)
data.dropna(inplace=True)

# Check the data
print(data.head())

# Convert the date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Set the date as the index
data.set_index('date', inplace=True)

# Separate X and y for the next day prediction
X = np.arange(len(data)).reshape(-1, 1)
y = data['shift_name_column']  # Replace 'shift_name_column' with your actual column name

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Plot the predictions
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Shift Name')
plt.title('Shift Name Prediction')
plt.legend()
plt.show()

# Prepare ARIMA model
arima_model = ARIMA(y, order=(5, 1, 0))
arima_fit = arima_model.fit()

# Make the next day's prediction
forecast = arima_fit.forecast(steps=1)
print(f'Next Day Prediction: {forecast[0]}')

