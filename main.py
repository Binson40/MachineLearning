import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
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





