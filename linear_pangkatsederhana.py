#Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

#File path  
file_path = 'D:/PROJECT VSCODE/PYTHON/Regresi/Student_Performance.csv'

#Load the dataset
data = pd.read_csv(file_path)

#Data
x = data ['Hours Studied'].values.reshape(-1, 1)
y = data ['Performance Index'].values

#Linear Regression
linear_model = LinearRegression()
linear_model.fit(x, y)
linear_predictions = linear_model.predict(x)

#Plot Linear Regression
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot (x, linear_predictions, color='red', label = 'Linear regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Linear Regression')
plt.legend()
plt.show()

#RMS error for Linear Regression
rms_error_linear = np.sqrt (mean_squared_error(y, linear_predictions))
print (f'RMS Error (Linear Regression) : {rms_error_linear}')

#Power Function: y = a * x^b
def power_function(z, a, b):
    return a * np.power(z, b)

#Fit Power Function to Data
params, _ = curve_fit(power_function, data['Hours Studied'], data['Performance Index'])
a, b = params
linear_predictions = power_function(data['Hours Studied'], a, b)

#Plot Power Regression
plt.scatter(data['Hours Studied'], data['Performance Index'],
color='blue', label='Data points')
plt.plot(data['Hours Studied'], linear_predictions, color='green',
label='Power regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Power Regression')
plt.legend()
plt.show()

#RMS error for Power Regression
rms_error_power = np.sqrt(mean_squared_error(data['Performance Index'], linear_predictions))
print(f'RMS Error (Power Regression): {rms_error_power}')
