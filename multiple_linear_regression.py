import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



print('#section for the Import the data set')
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

print(x)
print(y)
print('#Encode the categorical variable')
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [3])], remainder='passthrough')
x=np.array(ct.fit_transform(x))
print(x)
print(y)
print('#section for the Tarin and test data split')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
print('#created Liner Regression')
regressor= LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))





