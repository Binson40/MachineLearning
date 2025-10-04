import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print('#section for the Import the data set')
dataset = pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values
print(x)
print(y)
print('#section for the Tarin and test data split')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
print('#created Liner Regression')
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(x_train)
print(y_train)
print('#Predit Training test ')
y_pred=regressor.predict(x_test)
print('#Visualizing the training set results')
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
print('#Visualizing the test set results')
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
print('#Vishualization for singel predit')
print(regressor.predict([[30]]))
