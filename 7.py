
#Implementing KNN using sklearn library


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sn
df = pd.read_csv('IRIS_dataset.csv')
df.head()

print('\n X variable \n')
X = df.iloc[:,:-1].values
X.shape
X

print('\n y variable \n')
y = df.iloc[:,4].values
y.shape
y

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 10)

#feature scaling

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train
X_test

#training

classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(X_train,y_train)

#predictions

y_pred = classifier.predict(X_test)

#performance evaluation

confusion_matrix = pd.crosstab(y_test,y_pred, rownames = ['actual'],colnames = ['predicted'])
sn.heatmap(confusion_matrix, annot = True)

#training for various k values in a loop

from sklearn import metrics
scores = {}
scores_list = []
for k in range(1,30):
  classifier1 = KNeighborsClassifier(n_neighbors=k)
  classifier1.fit(X_train,y_train)
  y_pred1 = classifier1.predict(X_test)
  temp = metrics.accuracy_score(y_test,y_pred1)
  scores[k] = temp
  scores_list.append(temp)

k_range = range(1,30)
plt.plot(k_range,scores_list)
plt.title('accuracy of classifier for various values of k')
plt.xlabel('value of k')
plt.ylabel('accuracy')
