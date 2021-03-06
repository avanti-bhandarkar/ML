
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

df1 = pd.read_csv('Data_preprocess_1.csv')
df1

print('\n X variable \n')
X = df1.iloc[:,:-1].values
X

print('\n y variable \n')
y = df1.iloc[:,-1].values
y

#handling 

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:,1:])
X[:,1:] = imputer.transform(X[:,1:])
X

#categorical values >>> numerical values using label encoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y

labelencoder_z = LabelEncoder()
z = df1.iloc[:,0]
z

z = labelencoder_z.fit_transform(z)
z

df2 = df1
for r in range(0,df1.shape[0]):
  df2.iloc[r,0] = z[r]
  df2.iloc[r,-1] = y[r]

df2
