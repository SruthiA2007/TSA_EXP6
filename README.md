# Ex.No: 6               HOLT WINTERS METHOD
### Date: 16/05/2026



### AIM:

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
# Importing Necessary Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ---------------------------------------
# Load Salary Prediction Dataset
# ---------------------------------------
data = pd.read_csv('/content/salary_prediction_data.csv')

# Display First 5 Rows
print(data.head())

# ---------------------------------------
# Create Index Column
# ---------------------------------------
data['Index'] = pd.RangeIndex(start=1, stop=len(data)+1)

# Set Index
data.set_index('Index', inplace=True)

# ---------------------------------------
# Select Salary Column
# ---------------------------------------
salary_data = data[['Salary']]

# ---------------------------------------
# Plot Original Salary Data
# ---------------------------------------
salary_data.plot(figsize=(12,6))

plt.title('Original Salary Data')

plt.xlabel('Index')

plt.ylabel('Salary')

plt.show()

# ---------------------------------------
# Scale the Data
# ---------------------------------------
scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(
        salary_data.values.reshape(-1,1)
    ).flatten()
)

# Plot Scaled Data
scaled_data.plot(figsize=(12,6))

plt.title('Scaled Salary Data')

plt.xlabel('Index')

plt.ylabel('Scaled Salary')

plt.show()

# ---------------------------------------
# Seasonal Decomposition
# ---------------------------------------
decomposition = seasonal_decompose(
    salary_data,
    model='additive',
    period=12
)

decomposition.plot()

plt.show()

# ---------------------------------------
# Split Train and Test Data
# ---------------------------------------
scaled_data = scaled_data + 1

train_data = scaled_data[:int(len(scaled_data) * 0.8)]

test_data = scaled_data[int(len(scaled_data) * 0.8):]

# ---------------------------------------
# Holt-Winters Model
# ---------------------------------------
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# ---------------------------------------
# Forecast Test Data
# ---------------------------------------
test_predictions_add = model_add.forecast(
    steps=len(test_data)
)

# ---------------------------------------
# Visual Evaluation
# ---------------------------------------
ax = train_data.plot(figsize=(12,6))

test_predictions_add.plot(ax=ax)

test_data.plot(ax=ax)

ax.legend([
    "train_data",
    "test_predictions_add",
    "test_data"
])

ax.set_title('Visual Evaluation')

plt.show()

# ---------------------------------------
# RMSE Performance Metric
# ---------------------------------------
rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions_add
    )
)

print("RMSE Performance Metric")
print(rmse)

# Standard Deviation and Mean
print("\nStandard Deviation and Mean")

print(
    np.sqrt(scaled_data.var()),
    scaled_data.mean()
)

# ---------------------------------------
# Final Model
# ---------------------------------------
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# ---------------------------------------
# Predict Future Salary Data
# ---------------------------------------
final_predictions = final_model.forecast(
    steps=int(len(scaled_data)/4)
)

# ---------------------------------------
# Plot Final Prediction
# ---------------------------------------
ax = scaled_data.plot(figsize=(12,6))

final_predictions.plot(ax=ax)

ax.legend([
    "salary_data",
    "final_predictions"
])

ax.set_xlabel('Index')

ax.set_ylabel('Salary')

ax.set_title('Salary Prediction')

plt.show()
```

### OUTPUT:
ORGINAL SALARY DATA:

<img width="966" height="512" alt="image" src="https://github.com/user-attachments/assets/5df7bce9-1abd-43a1-8a3a-1f0e820bfa68" />

SCALED SALARY DATA:

<img width="928" height="510" alt="image" src="https://github.com/user-attachments/assets/73d4578d-8518-4a1b-a823-d1afa6d3b2ba" />


DECOMPOSED PLOT:

<img width="587" height="450" alt="image" src="https://github.com/user-attachments/assets/b7c280ef-05ad-47dc-ad72-4e14be60c84f" />



TEST_PREDICTION

<img width="917" height="487" alt="image" src="https://github.com/user-attachments/assets/61b46b90-80c4-4678-913a-b3b36ac850c6" />

```
RMSE Performance Metric
0.15663280338070712

Standard Deviation and Mean
0.17715293386416495 1.4516936807918916
```

FINAL_PREDICTION


<img width="940" height="517" alt="image" src="https://github.com/user-attachments/assets/d4c42f2a-c287-4852-94be-ca409af3757e" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
