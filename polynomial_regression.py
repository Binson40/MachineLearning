import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)
print('#section for the Tarin and test data split')
#x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=0)

print('#created Liner Regression')
lin_reg = LinearRegression()
lin_reg.fit(x,y)
print('#created polynomial Regression')
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)
print('#Visualizing the training set results for linear regression')
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('True vs Bluff (Liner Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
print('#Visualizing the training set results for polynomial regression')
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(x_poly), color='blue')
plt.title('True vs Bluff (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
lin_reg.predict([[6.5]])
print(lin_reg.predict([[6.5]]))
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))






