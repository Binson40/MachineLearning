import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#section for the Import the data set
print('#section for the Import the data set')
dataset = pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset  .iloc[:,-1].values
print(x)
print('hai how are u')
print(y)
#section for the clear the missing data
print('#section for the clear the missing data')
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print('after replacement')
print(x)
print('hai how are u')
print(y)
#section for the encoding  Independent variable
print('#section for the encoding Independent variable')
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x=np.array(ct.fit_transform(x))
print(x)
#section for the encoding dependent variable
print('#section for the encoding dependent  variable')
le= LabelEncoder()
y=np.array(le.fit_transform(y))
print(y)
#Tarin and test data split
print('#section for the Tarin and test data split')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print('#section for the X_Tarin and X_test data split')
print(X_train)
print(X_test)
print('#section for the Y_Tarin and Y_Test data split')
print(y_train)
print(y_test)
#Feature scaling Standard
print('#section for the Feature scaling Standard  ')
sc = StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
print( X_train)
print( 'x-test----')
X_test[:,3:]=sc.fit_transform(X_test[:,3:])
print(X_test)







